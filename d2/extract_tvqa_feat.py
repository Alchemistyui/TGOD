# ------------------------------------------------------------------------
# TGOD: region features extraction script.
# ------------------------------------------------------------------------
import argparse
import os
import json
import h5py
import time
import datetime
import numpy as np
from fvcore.common.file_io import PathManager

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from d2.tgod import add_tgod_config
from d2.tvqa import TVQADatasetMapper

def extract_feature_argument_parser():
    parser = argparse.ArgumentParser(description="Region feature extraction")
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument('--mode', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--model_weight', default='')
    parser.add_argument('--out_dir')

    return parser

def select_valid_object(outs, features, data, max_obj_len = 25):
    objects, out_feat, out_boxes, out_scores = [], [], [], []
    
    for idx, (out, feat) in enumerate(zip(outs, features)):        
        word_labels = out['instances'].word_labels
        scores = out['instances'].scores
        boxes = out['instances'].pred_boxes.tensor
        
        
        thresh = 0.3
        keep = torch.tensor(range(len(boxes)))
        # -------- filt by threshold --------
        if (scores[keep].squeeze()>thresh).any():
            keep = keep[torch.nonzero(scores[keep].squeeze()>thresh).squeeze(1)]
        keep = keep[word_labels[keep]<len(data[0]['qa_words'][idx])]
        keep = keep[:max_obj_len]

        out_scores.append([round(i, 4) for i in scores[keep].tolist()])
        out_feat.append(feat[keep])
        word_labels = word_labels[keep]

        objects.append([data[0]['qa_words'][idx][x] for x in word_labels])
        new_boxes = []
        for b in boxes[keep].tolist():
            new_boxes.append([round(i, 2) for i in b])
        out_boxes.append(new_boxes)

    counts = [len(x) for x in out_boxes]
    
    out_features=torch.zeros(len(out_feat), max([i.shape[0] for i in out_feat]), out_feat[0].shape[-1])
    for feat_idx, feat in enumerate(out_feat):
        out_features[feat_idx, :feat.shape[0], :feat.shape[1]] = feat.detach()
    return {'counts': counts, 'object': objects, 'boxes': out_boxes}, out_scores, out_features


def do_feature_extraction_tgod(cfg, model, mode, debug):
    dump_folder = os.path.join(
        cfg.OUTPUT_DIR, "extract_vfeat_vcpt/"
    )
    PathManager.mkdirs(dump_folder)
    model.eval()
    
    mapper = TVQADatasetMapper(cfg, False, is_extract=True)
    loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST, mapper=mapper)

    if debug:
        out_file_vcpt = os.path.join(dump_folder, f'debug_{mode}.json')
        h5f = h5py.File(os.path.join(dump_folder, f'debug_{mode}.h5'), "w")
    else:
        out_file_vcpt = os.path.join(dump_folder, f'extract_vcpt_per_img_{mode}.json')
        h5f = h5py.File(os.path.join(dump_folder, f'extract_vfeat_per_img_{mode}.h5'), "w")
    print(f'---- annotation will be write to : {out_file_vcpt}----')
    vis_count = 0
    for idx, data in enumerate(loader):
        qid = data[0]['qid']
        tic = time.time()
        # image_name_idx in TVQA+ Dataset start from 1
        img_name_idx = int(data[0]['file_name'].split('/')[-1].split('.')[0])
        conv_features = []

        # generate box for each qa pairs
        input_imgs = [{'image': data[0]['image'], 'height': data[0]['height'], 'width': data[0]['width'], 'qa_embed': embed} for embed in data[0]['qa_embed']]
        outs = model(input_imgs)

        feature_pooled = [o['instances'].proj_queries for o in outs]

        # select valid object prediction
        annos, scores, out_features = select_valid_object(outs, feature_pooled, data)

        vcpt = {'qid': int(qid), 'img_name_idx': img_name_idx}
        vcpt.update(annos)
        with open(out_file_vcpt, 'a') as f:
            json.dump(vcpt, f)
            f.write('\n')
        if qid in h5f:
            qid_dset = h5f[qid]
        else:
            qid_dset = h5f.create_group(qid)
        qid_dset.create_dataset(str(img_name_idx), data=out_features.numpy(), dtype=np.float32)
        if idx % 1000 == 0:
            rest_time = int((time.time()-tic)*(len(loader)-idx))
            print(str(idx)+'/'+str(len(loader))+':', 'rest time:', datetime.timedelta(seconds=rest_time))

    h5f.close()
    print(f'---- finish extraction and write to : {out_file_vcpt}----')
    return dump_folder

def read_json_data(file):
    with open(file) as f:
        for line in f.readlines():
            item = json.loads(line)

def transform_vfeat(feat_path, out_file):
    out_feat = {}

    print('loading feature from {}'.format(feat_path.split('/')[-1]))
    cur_time = time.time()
    feat_h5 = h5py.File(feat_path, "r", driver=None)
    
    qids = list(feat_h5.keys())
    for qid in qids:
        img_ids = sorted(list(feat_h5[qid].keys()), key = lambda i:int(i))
        for img_id in img_ids:
            feat = feat_h5[qid][img_id][:]

            ori_shape = feat.shape
            if ori_shape[1] == 0:
                feat_data = torch.zeros(ori_shape[0],1,300)
            else:
                feat_data = torch.tensor(feat).half()
                if torch.isinf(feat_data).any():
                    feat_data = torch.tensor(feat)
            if qid in out_feat.keys():
                out_feat[qid].extend([feat_data])
            else:
                out_feat[qid] = [feat_data]

    torch.save(out_feat, out_file, _use_new_zipfile_serialization=False)
    print('*Saving feature after pca done to: ', out_file)

def transform_vcpt(vcpt_path, out_file):
    items = []
    with open(vcpt_path) as f:
        for line in f.readlines():
            item = json.loads(line)
            items.append(item)

    vcpt = {'counts': {}, 'object': {}, 'boxes':{}}
    for item in items:
        if item['qid'] in vcpt['counts'].keys():
            vcpt['counts'][item['qid']].update({item['img_name_idx']: item['counts']})
            vcpt['object'][item['qid']].update({item['img_name_idx']: item['object']})
            vcpt['boxes'][item['qid']].update({item['img_name_idx']: item['boxes']})
        else:
            vcpt['counts'][item['qid']] = {item['img_name_idx']: item['counts']}
            vcpt['object'][item['qid']] = {item['img_name_idx']: item['object']}
            vcpt['boxes'][item['qid']] = {item['img_name_idx']: item['boxes']}
    out_vcpt = {}
    for qid in vcpt['counts'].keys():
        img_ids = list(vcpt['counts'][qid].keys())
        img_ids.sort()
        out_vcpt[qid] = {'counts': [vcpt['counts'][qid][i] for i in img_ids], 
                        'object': [vcpt['object'][qid][i] for i in img_ids], 
                        'boxes':[vcpt['boxes'][qid][i] for i in img_ids]}

    torch.save(out_vcpt, out_file, _use_new_zipfile_serialization=False)
    print('*Saving vcpt after pca done to: ', out_file)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_tgod_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    cfg.defrost()
    # modify the dataset info in cfg to extract mode
    if args.mode == 'train':
        # tvqa_extract contain other info: qa_words and qa_idx, and all the qa pairs
        cfg.DATASETS.TEST = ('tvqa_extract_train_pos_tag')
    if args.mode == 'val':
        cfg.DATASETS.TEST = ('tvqa_extract_val_pos_tag')
    if args.mode == 'test':
        cfg.DATASETS.TEST = ('tvqa_extract_test_pos_tag')

    cfg.freeze()
    model = build_model(cfg)
    if args.model_weight == '':
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
    else:
        print('---- load from args model weight ----')
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            args.model_weight, resume=False
        )

    dump_folder = do_feature_extraction_tgod(cfg, model, args.mode, args.debug)
    return dump_folder


if __name__ == "__main__":
    args = extract_feature_argument_parser().parse_args()
    print("Command Line Args:", args)
    tic = time.time()
    dump_folder = main(args)
    used_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))
    print(f'finished extraction, time spent: {used_time}')


    print('transfer to STAGE input format')
    in_base_path = dump_folder
    raw_vfeat_path = os.path.join(in_base_path, 'extract_vfeat_per_img_{}.h5'.format(args.mode))
    raw_vcpt_path = os.path.join(in_base_path, 'extract_vcpt_per_img_{}.json'.format(args.mode))
    out_dir = args.out_dir
    transform_vfeat(raw_vfeat_path, os.path.join(out_dir,"tvqa_tensor_feature_{}.pt".format(args.mode)))
    transform_vcpt(raw_vcpt_path, os.path.join(out_dir,"tvqa_vcpt_{}.pt".format(args.mode)))