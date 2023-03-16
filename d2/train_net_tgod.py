# ------------------------------------------------------------------------
# TGOD: training script
# ------------------------------------------------------------------------
# Modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import os
import sys
import itertools

# fmt: off
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from d2.tgod import add_tgod_config
from d2.tvqa import TVQADatasetMapper
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results, print_csv_format
from detectron2.evaluation import inference_on_dataset
from detectron2.solver.build import maybe_add_gradient_clipping

import logging
from collections import OrderedDict


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = TVQADatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = TVQADatasetMapper(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        lr_backbone_names = ["backbone.0"]
        lr_linear_proj_names = ['reference_points', 'sampling_offsets']
        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out
        param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()
                    if not match_name_keywords(n, lr_backbone_names) and not match_name_keywords(n, lr_linear_proj_names) and p.requires_grad],
            "lr": cfg.SOLVER.BASE_LR,
        },
        {
            "params": [p for n, p in model.named_parameters() if match_name_keywords(n, lr_backbone_names) and p.requires_grad],
            "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.BACKBONE_MULTIPLIER,
        },
        {
            "params": [p for n, p in model.named_parameters() if match_name_keywords(n, lr_linear_proj_names) and p.requires_grad],
            # "lr": args.lr * args.lr_linear_proj_mult,
            "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.BACKBONE_MULTIPLIER,
        }
    ]
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.SOLVER.BASE_LR,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)

        return optimizer

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

    def run_step(self):
        """
        !!Hack!! for the run_step method in SimpleTrainer to adjust the loss
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict, losses = self.model(data)

        self.optimizer.zero_grad()
        losses.backward()
        max_norm = 0.1
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        else:
            grad_total_norm = torch.nn.utils.get_total_grad_norm(self.model.parameters(), max_norm)


        self.optimizer.step()

        with torch.cuda.stream(torch.cuda.Stream()):
            metrics_dict = loss_dict
            self._write_metrics(metrics_dict)
            self._detect_anomaly(losses, metrics_dict)
        

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_tgod_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)


    if args.debug:
        setup_seed(20)
        trainer.debug_train(cfg)
    
    
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
