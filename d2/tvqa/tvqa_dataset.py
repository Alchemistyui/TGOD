# ------------------------------------------------------------------------
# TGOD: TVQA+ dataset registration
# ------------------------------------------------------------------------
import os
import json
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

def get_tvqa_dicts(data_file, mode):
    with open(data_file) as f:
        imgs_anns = json.load(f)

    for x in imgs_anns:
        for anno in x['annotations']:
            anno['category_id'] = 0
            anno['bbox_mode'] = BoxMode.XYWH_ABS

    return imgs_anns


# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")

# only select possible object word as text token
for d in ["train", "val"]:
    DatasetCatalog.register("tvqa_" + d + '_pos_tag', lambda d=d: get_tvqa_dicts(f'/DATA_DIR/tgod_anno/img_anno_{d}.json', d))
    MetadataCatalog.get("tvqa_" + d + '_pos_tag').set(thing_classes=['matched', 'not matched'])
    MetadataCatalog.get("tvqa_" + d + '_pos_tag').set(qa_embed_path=f'/DATA_DIR/tgod_anno/qid_qafeat_{d}.pt')
    MetadataCatalog.get("tvqa_" + d + '_pos_tag').set(img_base_dir='/DATA_DIR/frames_hq/bbt_frames/')

# extraction version
for d in ["train", "val", 'test']:
    DatasetCatalog.register("tvqa_extract_" + d + '_pos_tag', lambda d=d: get_tvqa_dicts(f'/DATA_DIR/tgod_anno/extract_img_anno_{d}.json', d))
    MetadataCatalog.get("tvqa_extract_" + d + '_pos_tag').set(thing_classes=['matched', 'not matched'])
    MetadataCatalog.get("tvqa_extract_" + d + '_pos_tag').set(qa_embed_path=f'/DATA_DIR/tgod_anno/qid_qafeat_{d}.pt')
    MetadataCatalog.get("tvqa_extract_" + d + '_pos_tag').set(img_base_dir='/DATA_DIR/frames_hq/bbt_frames/')

