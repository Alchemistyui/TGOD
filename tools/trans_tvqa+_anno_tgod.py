# ------------------------------------------------------------------------
# TGOD
# Transform TVQA+ annotation into TGOD input format for training
# ------------------------------------------------------------------------
import os
import json
import torch
import h5py
import numpy as np
import cv2
import time
import datetime
import spacy
from detectron2.structures import BoxMode


DATA_DIR = None
mode = 'val'
# input path
anno_dir = DATA_DIR+'tvqa_plus_stage_features/'
qa_bert_path = os.path.join(anno_dir, 'bbt_qa_s_tokenized_bert_sub_qa_tuned_new_qid.h5')

if mode == 'train':
    data_anno_path = os.path.join(anno_dir, 'tvqa_plus_train_preprocessed.json')
elif mode == 'val':
    data_anno_path = os.path.join(anno_dir, 'tvqa_plus_valid_preprocessed.json')
elif mode == 'test':
    data_anno_path = os.path.join(anno_dir, 'tvqa_plus_test_preprocessed_no_anno.json')
all_img_name_path = os.path.join(anno_dir.rsplit('/', 2)[0], 'coco_style_anno',f'qid_vid_imgidxs_{mode}.json')
word2idx_path = os.path.join(anno_dir, 'word2idx.json')
img_base_dir = DATA_DIR+'frames_hq/bbt_frames/'

# out path
out_dir = DATA_DIR+'tgod_anno/extract/'
out_qa_path = os.path.join(out_dir, f'qid_qafeat_{mode}.pt')
out_img_path = os.path.join(out_dir, f'img_anno_{mode}.json')
extract_img_path = os.path.join(out_dir, f'extract_img_anno_{mode}.json')


with open(data_anno_path) as f:
    cur_data_dict = json.load(f)
qa_bert_h5 = h5py.File(qa_bert_path, "r", driver=None)

# word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
with open(word2idx_path) as f:
    word2idx = json.load(f)

def numericalize(sentence, eos=True, match=False):
    """convert words to indices, match stanford tokenizer"""
    if match:
        sentence = match_stanford_tokenizer(sentence)
    sentence_indices = [word2idx[w] if w in word2idx else word2idx["<unk>"]
                        for w in line_to_words(sentence, eos=eos)]  # 1 is <unk>, unknown
    return sentence_indices

def line_to_words(line, eos=True, downcase=True):
    eos_word = "<eos>"
    words = line.lower().split() if downcase else line.split()

    words = [w for w in words]
    words = words + [eos_word] if eos else words
    return words


'''
for qas: {'qid': {'vid_name':, 'qa_feat': [qa1, qa2, qa3, qa4, qa5], 'answer_idx': int}, ...}

'''
def transfer_qa_paires(sp):
    out_qa_anno = {}

    start_time = time.time()
    max_pos_select_words = 0
    for index in range(len(cur_data_dict)):
        qid = cur_data_dict[index]['qid']
        answer_keys = ["a0", "a1", "a2", "a3", "a4"]
        q_a_sentence = [cur_data_dict[index]["q"]
                        + " " + cur_data_dict[index][k] for k in answer_keys]
        qa_numerical = [numericalize(k, eos=False) for k in q_a_sentence]
        
        q_a_words = [x.lower().split() for x in q_a_sentence]

        valid_words = []
        valid_word_bert = []
        for idx, (qa, k) in enumerate(zip(q_a_words, answer_keys)):
            word2idx = get_pos_tag(sp, cur_data_dict[index]['q'], 
                    cur_data_dict[index][k],
                    qa, cur_data_dict[index]['q_len'])
            pos_select_words = list(word2idx.keys())
            valid_words.append(pos_select_words)
            max_pos_select_words = len(pos_select_words) if len(pos_select_words) > max_pos_select_words else max_pos_select_words

            sentence_word_bert = np.concatenate([qa_bert_h5[str(qid) + "_q"], qa_bert_h5[str(qid) + "_" + k]], axis=0)
            assert len(qa) == len(qa_numerical[idx]) == sentence_word_bert.shape[0]
            if pos_select_words == []:
                print(cur_data_dict[index]['bbox'])
                if list(cur_data_dict[index]['bbox'].values())[0] != []:
                    import pdb; pdb.set_trace()
                words_bert = np.zeros([1,sentence_word_bert.shape[-1]])
            else:
                words_bert = np.concatenate([np.expand_dims(sentence_word_bert[w_idx].mean(0),axis=0) 
                                for w_idx in word2idx.values()])
            valid_word_bert.append(words_bert)
            
        out_qa_anno[qid] = {'vid_name': cur_data_dict[index]['vid_name'],
                            'qa_feat': valid_word_bert,
                            'valid_words': valid_words,
                            'qa_sentence': q_a_sentence,
                            'answer_idx': cur_data_dict[index]['answer_idx']}
        
        if index % 500 == 1:
            rest_time = int((time.time()-start_time)/index*(len(cur_data_dict)-index))
            print(str(index)+'/'+str(len(cur_data_dict))+':', 'rest time:', datetime.timedelta(seconds=rest_time))

    print('max number of label (vcpt):', max_pos_select_words)
    return out_qa_anno

'''
for img: coco-style format
    [{"qid":, file_name":, "image_id":, "height":, "width":, "annotations": {"bbox": , "label": , 'box_idx': }}, ...]
'''
def get_pos_tag(sp, q, a, qa_words, q_len):
    q_sen = sp(q)
    ca_sen = sp(a)
    
    word2idx = {}
    for i, word in enumerate(qa_words):   
        pos_tag = q_sen[i].pos_ if i < q_len else ca_sen[i-q_len].pos_
        if pos_tag in ['PROPN', 'NOUN']:
            if word in word2idx.keys():
                word2idx[word].append(i)
            else:
                word2idx[word] = [i]
    return word2idx

def transfer_img(sp):
    out_img_anno = []
    count = 0
    max_pos_select_words = 0
    start_time = time.time()
    for index in range(len(cur_data_dict)):
        box_annos = cur_data_dict[index]['bbox']
        q_ca_sentence = cur_data_dict[index]['q'] + ' ' + cur_data_dict[index]['a'+cur_data_dict[index]['answer_idx']]
        q_ca_words = q_ca_sentence.lower().split()
        
        word2idx = get_pos_tag(sp, cur_data_dict[index]['q'], 
                    cur_data_dict[index]['a'+cur_data_dict[index]['answer_idx']],
                    q_ca_words, cur_data_dict[index]['q_len'])
        pos_select_words = list(word2idx.keys())
        max_pos_select_words = len(pos_select_words) if len(pos_select_words) > max_pos_select_words else max_pos_select_words

        for k in box_annos.keys():
            if len(box_annos[k]) == 0:
                continue
            info = {}
            info['qid'] = cur_data_dict[index]['qid']
            info['file_name'] = os.path.join(cur_data_dict[index]['vid_name'], k.zfill(5)+'.jpg')
            info['image_id'] = count

            height, width = cv2.imread(os.path.join(img_base_dir, info['file_name'])).shape[:2]
            info['height'] = height
            info['width'] = width
            info['q_ca_sentence'] = q_ca_sentence

            objs = []
            for anno in box_annos[k]:
                label = anno['label'].lower()
                if '’' in label:
                    label = label.replace('’', "'")
                
                if label in pos_select_words:
                    label_idx = pos_select_words.index(label)
                else:
                    print(anno, ':', pos_select_words)
                obj = {
                    # xywh
                    "bbox": [anno["left"], anno["top"], anno["width"], anno["height"]],
                    "label": anno['label'],
                    "box_idx": label_idx,
                }
                objs.append(obj)


            info['annotations'] = objs
            count += 1
            out_img_anno.append(info)

        if index % 500 == 1:
            rest_time = int((time.time()-start_time)/index*(len(cur_data_dict)-index))
            print(str(index)+'/'+str(len(cur_data_dict))+':', 'rest time:', datetime.timedelta(seconds=rest_time))

    return out_img_anno

def transfer_img_extract():
    out_img_anno = []
    count = 0
    with open(all_img_name_path) as f:
        img_series_dict = json.load(f)

    for q_idx, qid in enumerate(img_series_dict.keys()):
        data = img_series_dict[qid]
        for k in data['image_indices']:

            info = {}
            info['qid'] = qid
            info['file_name'] = os.path.join(data['vid_name'], str(k).zfill(5)+'.jpg')
            info['image_id'] = count

            height, width = cv2.imread(os.path.join(img_base_dir, info['file_name'])).shape[:2]
            info['height'] = height
            info['width'] = width

            objs = []
            info['annotations'] = objs
            count += 1
            out_img_anno.append(info)

    return out_img_anno
    
if __name__ == '__main__':
    sp = spacy.load('en_core_web_sm')
    dataset_dicts = transfer_img(sp)
    with open(out_img_path, 'w') as f:
        json.dump(dataset_dicts, f)
    print(mode, 'img info saved...length:', len(dataset_dicts))

    out_qa_anno = transfer_qa_paires(sp)
    torch.save(out_qa_anno, out_qa_path)
    print(mode, 'qa feat saved...length:', len(out_qa_anno))

    # generate image annotation for visual feature extraction
    extract_dicts = transfer_img_extract()
    with open(extract_img_path, 'w') as f:
        json.dump(extract_dicts, f)
    print(mode, 'extract img info saved...length:', len(extract_dicts))




