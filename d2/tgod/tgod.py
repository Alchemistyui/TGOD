# ------------------------------------------------------------------------
# TGOD structure
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from deformable_models.backbone import Joiner
from deformable_models.tgod_detr import DeformableTGOD, SetCriterionTGOD, PostProcessTGOD
from deformable_models.tgod_matcher import HungarianMatcherTGOD
from deformable_models.position_encoding import PositionEmbeddingSine
from deformable_models.tgod_transformer import DeformableTGODTransformer
from util.misc import NestedTensor, reduce_dict
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels

        # if cfg.
        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        self.strides = [8, 16, 32]
        self.num_channels = cfg.MODEL.DETR.NUM_CHANNELS

    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


@META_ARCH_REGISTRY.register()
class TGOD(nn.Module):
    """
    Implement Detr
    """
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.mask_on = cfg.MODEL.MASK_ON
        hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        self.num_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES
        self.word_len = cfg.MODEL.DETR.MAX_WORD_LEN
        # Transformer parameters:
        nheads = cfg.MODEL.DETR.NHEADS
        dropout = cfg.MODEL.DETR.DROPOUT
        dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        pre_norm = cfg.MODEL.DETR.PRE_NORM
        # Deformable DETR parameters:
        feat_level = cfg.MODEL.DETR.FEATLEVEL
        
        # Loss parameters:
        giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        class_weight = cfg.MODEL.DETR.CLASS_WEIGHT
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        contrastive_weight = cfg.MODEL.DETR.CONTASTIVE_WEIGHT
        contrastive_loss = cfg.MODEL.DETR.CONTRASTIVE_LOSS
        contrastive_hdim = cfg.MODEL.DETR.CONTRASTIVE_HDIM

        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels

        transformer = DeformableTGODTransformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            return_intermediate_dec=deep_supervision,
        )

        self.detr = DeformableTGOD(
            backbone, transformer, num_classes=self.num_classes, num_queries=self.num_queries, 
            num_feature_levels=feat_level,
            aux_loss=deep_supervision,
            max_word_len=self.word_len,
            d_model=hidden_dim,
            is_contrastive=contrastive_loss,
            contrastive_hdim=contrastive_hdim)

        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcherTGOD(cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight}
        weight_dict["loss_giou"] = giou_weight
        if contrastive_loss:
            weight_dict['loss_contrastive'] = contrastive_weight
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)


        losses = ["labels", "boxes", "cardinality"]
        if contrastive_loss:
            losses.append('contrastive')
        if self.mask_on:
            losses += ["masks"]
        self.criterion = SetCriterionTGOD(
            self.num_classes, matcher=matcher, weight_dict=weight_dict, losses=losses
        )
        self.criterion.to(self.device)
        self.postprocessors = {'bbox': PostProcessTGOD()}

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def exclude_text_token(self, output):
        """
        exclude the output for the text token inputs, 
        only remain object query prediction
        """
        obj_query_len = self.num_queries-self.word_len-1
        output['pred_logits'] = output['pred_logits'][:, :obj_query_len]
        output['pred_boxes'] = output['pred_boxes'][:, :obj_query_len]
        if 'proj_queries' in output.keys():
            output['proj_queries'] = output['proj_queries'][:, :obj_query_len]
        aux = []
        for lvl in output['aux_outputs']:
            if 'proj_queries' in lvl.keys():
                aux.append({
                    'pred_logits': lvl['pred_logits'][:, :obj_query_len],
                    'pred_boxes': lvl['pred_boxes'][:, :obj_query_len],
                    'proj_queries': lvl['proj_queries'][:, :obj_query_len],
                    'proj_tokens': lvl['proj_tokens']
                })
            else:
                aux.append({
                    'pred_logits': lvl['pred_logits'][:, :obj_query_len],
                    'pred_boxes': lvl['pred_boxes'][:, :obj_query_len]
                })
        output['aux_outputs'] = aux
        return output

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        text_token, text_pad_mask = self.preprocess_text_token(batched_inputs)
        output = self.detr(images, text_token, text_pad_mask)
        output = self.exclude_text_token(output)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            targets, batch_positive_map = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets, batch_positive_map)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}

            return loss_dict_reduced_scaled, losses
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]

            proj_queries = output['proj_queries'] if 'proj_queries' in output.keys() else None
            proj_tokens = output['proj_tokens'] if 'proj_tokens' in output.keys() else None
            mask_pred = output["pred_masks"] if self.mask_on else None
            processed_results = []

            for idx, (input_per_image, image_size) in enumerate(zip(batched_inputs, images.image_sizes)):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                ori_size = [height, width]

                dict_result = {'pred_logits': box_cls[idx].unsqueeze(0), 'pred_boxes': box_pred[idx].unsqueeze(0)}
                if proj_queries != None:
                    dict_result['proj_queries'] = proj_queries[idx].unsqueeze(0)
                r = self.postprocessors['bbox'](dict_result, torch.tensor(ori_size).unsqueeze(0).to(box_cls.device))

                result = Instances(ori_size)
                result.pred_boxes = Boxes(r[0]['boxes'])
                result.scores = r[0]['scores']
                result.pred_classes = r[0]['labels']
                if 'proj_queries' in r[0].keys():
                    result.proj_queries = r[0]['proj_queries'] 
                if 'word_labels' in r[0].keys():
                    result.word_labels = r[0]['word_labels']
                if proj_tokens is not None:
                    processed_results.append({"instances": result, 'proj_tokens': proj_tokens[idx]})
                else:
                    processed_results.append({"instances": result})

            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        batch_positive_map = []

        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})

            
            pos_map = targets_per_image.positive_map
            batch_positive_map.append(pos_map)
        return new_targets, torch.cat(batch_positive_map)

    def inference(self, box_cls, box_pred, mask_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
            scores, labels, box_pred, image_sizes
        )):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                mask = F.interpolate(mask_pred[i].unsqueeze(0), size=image_size, mode='bilinear', align_corners=False)
                mask = mask[0].sigmoid() > 0.5
                B, N, H, W = mask_pred.shape
                mask = BitMasks(mask.cpu()).crop_and_resize(result.pred_boxes.tensor.cpu(), 32)
                result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images
    
    def preprocess_text_token(self, batched_inputs):
        
        text_feat = [x['qa_embed'] for x in batched_inputs]
        padded = torch.zeros((len(text_feat), self.word_len, text_feat[0].shape[-1]))
        key_padding_mask = torch.zeros((len(text_feat), self.word_len), dtype=torch.bool)
        for idx, x in enumerate(text_feat):
            assert x.shape[0] <= self.word_len, 'sentence is longer than the max word len!'
            padded[idx, :x.shape[0], :x.shape[1]] = torch.from_numpy(x)
            key_padding_mask[idx, x.shape[0]:] = True

        return padded, key_padding_mask
