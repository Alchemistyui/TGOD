# ------------------------------------------------------------------------
# TGOD: dataset mapper for TVQA+ dataset
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import copy
import os
import logging

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, BoxMode, Boxes

__all__ = ["TVQADatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


def tvqa_annotations_to_instances(annos, image_size, max_qa_len):
    boxes = [BoxMode.convert(torch.Tensor(obj["bbox"]), obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)

    target.gt_boxes = Boxes(torch.stack(boxes,0))
    classes = torch.zeros(len(target), max_qa_len)
    target.gt_classes = torch.tensor([0 for obj in annos])
    for i, idx in enumerate([obj['box_idx'] for obj in annos]):
        classes[i][idx] = 1

    target.cls_weight = torch.tensor([1 for x in annos])
    target.positive_map = classes
    
    return target


class TVQADatasetMapper:
    def __init__(self, cfg, is_train=True, return_val_anno=False, is_extract=False):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.is_extract = is_extract
        self.return_val_anno = return_val_anno
        self.max_qa_len = cfg.MODEL.DETR.MAX_WORD_LEN
        self.mask_on = cfg.MODEL.MASK_ON

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )
        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

        if self.is_extract:
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST)
            self.qa_embed = torch.load(self.metadata.qa_embed_path)
        else:
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) if is_train else MetadataCatalog.get(cfg.DATASETS.TEST[0])
            self.qa_embed = torch.load(self.metadata.qa_embed_path)
            
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(os.path.join(self.metadata.img_base_dir, dataset_dict["file_name"]), format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if (not self.is_train) and (not self.is_extract) and (not self.return_val_anno):
            # only select the correct answer
            if isinstance(list(self.qa_embed.keys())[0], int):
                dataset_dict['qa_embed'] = self.qa_embed[dataset_dict['qid']]['qa_feat'][int(self.qa_embed[dataset_dict['qid']]['answer_idx'])]
            else:
                dataset_dict['qa_embed'] = self.qa_embed[str(dataset_dict['qid'])]['qa_feat'][int(self.qa_embed[str(dataset_dict['qid'])]['answer_idx'])]
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if self.is_extract: 
            if isinstance(list(self.qa_embed.keys())[0], int):
                dataset_dict['qa_embed'] = self.qa_embed[int(dataset_dict['qid'])]['qa_feat']
                if 'qa_words' not in dataset_dict.keys():
                    dataset_dict['qa_words'] = self.qa_embed[int(dataset_dict['qid'])]['valid_words']
                    dataset_dict['qas'] = self.qa_embed[int(dataset_dict['qid'])]['qa_sentence']
                # only use for visualization
                dataset_dict['ca_idx'] = self.qa_embed[int(dataset_dict['qid'])]['answer_idx']
            else:
                dataset_dict['qa_embed'] = self.qa_embed[str(dataset_dict['qid'])]['qa_feat']
                dataset_dict['qa_words'] = self.qa_embed[str(dataset_dict['qid'])]['qa_words']

            if not self.return_val_anno:
                return dataset_dict
        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = tvqa_annotations_to_instances(annos, image_shape, self.max_qa_len)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

            if not self.is_extract:
                try:
                    dataset_dict['qa_embed'] = self.qa_embed[int(dataset_dict['qid'])]['qa_feat'][int(self.qa_embed[int(dataset_dict['qid'])]['answer_idx'])]
                except:
                    dataset_dict['qa_embed'] = self.qa_embed[str(dataset_dict['qid'])]['qa_feat'][int(self.qa_embed[str(dataset_dict['qid'])]['answer_idx'])]

        return dataset_dict
