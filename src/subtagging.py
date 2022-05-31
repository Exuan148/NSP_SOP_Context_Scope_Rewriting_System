import argparse
import os
import logging
from others.logging import set_logger, init_logger
import torch
import glob
import numpy as np
import random
logger = logging.getLogger(__name__)
def subtagging_batch(batch,scope,sels,context=0.5):
    """
    trim and sub-tag document after knowing about the index of the extracted sentences
    :param batch:
    :param scope:
    :param sels:
    :param context:
    :return:
    """
    GPUdivice=batch.src.device

    def isvacuum(l):
        for i in l:
            if (i != 0): return False
        return True
    def padding(l,length):
        for i in range(len(l)):
            ini_length=len(l[i])
            if(ini_length>=length):
                for p in range(ini_length-length):
                    l[i].pop()
            else:
                l[i].extend([0 for p in range(length - ini_length)])
        return l

    def padding_tag(l,length):
        for i in range(len(l)):
            ini_length = len(l[i])
            # print(ini_length)
            number=len(l[i][0])
            if(ini_length>=length):
                for p in range(ini_length-length):
                    l[i].pop()
            else:
                for p in range(length-ini_length):

                    l[i].append([0 for k in range(number)])
        return l


    radius = int(scope / 2)

    src_tags_set=[]
    # detract operation: token-level
    detr_src_set = []
    detr_tag_set = []
    detr_msk_src_set = []
    # detract operation: sentence-level
    detr_cls_set = []
    detr_src_str_set = []
    detr_seg_set = []
    detr_msk_cls_set = []
    detr_msk_label_set = []

    for b_i in range(batch.batch_size):

        # adding subtags operation
        sen_length = len(batch.src_str[b_i])
        word_length = len(batch.src[b_i])
        src_sent_labels = [0 for i in range(sen_length)]
        src_sent_labels_point = [0 for i in range(sen_length)]
        src_tags = batch.tag_src[b_i].to(torch.device("cpu")).numpy().tolist()
        try:
            sels=sels.numpy().tolist()
        except:
            pass
        for index, idx in enumerate(sels[b_i]):
            flag = 0  # right scope
            for i in range(batch.clss[b_i][idx], word_length):
                if (batch.src[b_i][i] == 101): flag += 1
                if (flag >= radius + 2):
                    break
                elif (flag > 1):
                    src_tags[i][index] = context
            flag = 0  # left scope
            i = batch.clss[b_i][idx] - 1
            while (i >= 0):
                if (batch.src[b_i][i] == 102): flag += 1
                if (flag >= radius + 1):
                    break
                else:
                    src_tags[i][index] = context
                i -= 1

            src_sent_labels_point[idx] = 1
            flag = 0  # right scope
            for i in range(idx, sen_length):
                if (flag > radius): break
                src_sent_labels_point[i] = 1
                flag += 1
            flag = 0  # left scope
            i = idx
            while (i >= 0):
                if (flag > radius): break
                src_sent_labels_point[i] = 1
                flag += 1
                i -= 1

        src_tags_set.append(src_tags.copy())

        # detract operation: token-level
        detr_src = []
        detr_tag = []
        detr_msk_src=[]
        number = len(sels[b_i])

        for i in range(len(batch.tag_src[b_i])):
            if(isvacuum(src_tags[i]) is True):continue
            detr_src.append(batch.src.to(torch.device("cpu")).numpy().tolist()[b_i][i])
            # detr_seg.append(dataset['segs'][i])
            detr_tag.append(src_tags[i])
            detr_msk_src.append(1)
        detr_src_set.append(detr_src.copy())
        detr_tag_set.append(detr_tag.copy())
        detr_msk_src_set.append(detr_msk_src.copy())

        # detract operation: sentence-level
        detr_cls = []
        detr_src_str = []
        detr_seg = []
        detr_msk_cls = []
        detr_msk_label = []
        seg_idx=0
        for i in range(len(detr_src)):
            if (detr_src[i] == 101):
                detr_cls.append(i)
                detr_msk_cls.append(1)
                detr_msk_label.append(1)
                seg_idx+=1
            detr_seg.append(seg_idx)
        for i in range(sen_length):
            # print(i)
            if (src_sent_labels_point[i] != 0):

                detr_src_str.append(batch.src_str[b_i][i])
        detr_cls_set.append(detr_cls.copy())
        detr_src_str_set.append(detr_src_str.copy())
        detr_seg_set.append(detr_seg.copy())
        detr_msk_cls_set.append(detr_msk_cls.copy())
        detr_msk_label_set.append(detr_msk_label.copy())

    detr_cls_set = padding(detr_cls_set, 25)
    batch.clss=torch.tensor(np.array(detr_cls_set),dtype=torch.int64).to(GPUdivice)

    detr_msk_cls_set = padding(detr_msk_cls_set, 25)
    batch.mask_cls=torch.tensor(np.array(detr_msk_cls_set),dtype=torch.uint8).to(GPUdivice)

    detr_msk_label_set = padding(detr_msk_label_set, 25)
    batch.mask_labels=torch.tensor(np.array(detr_msk_label_set),dtype=torch.uint8).to(GPUdivice)

    detr_msk_src_set = padding(detr_msk_src_set, 512)
    batch.mask_src=torch.tensor(np.array(detr_msk_src_set),dtype=torch.uint8).to(GPUdivice)

    # msk_tgt
    detr_seg_set = padding(detr_seg_set,512)
    batch.segs=torch.tensor(np.array(detr_seg_set),dtype=torch.int64).to(GPUdivice)

    detr_src_set=padding(detr_src_set,512)
    batch.src=torch.tensor(np.array(detr_src_set),dtype=torch.int64).to(GPUdivice)

    # src_sent_label

    detr_tag_set=padding_tag(detr_tag_set,512)
    batch.tag_src=torch.tensor(np.array(detr_tag_set),dtype=torch.float32).to(GPUdivice)

    # tag_tgt
    # tgt

    batch.src_str=detr_src_str_set

    return batch

def subtagging(args,corpus_type,context=1.1,sent_dropout=0.2):
    def find_one(l):
        for i in range(len(l)):
            if(l[i]==1): return i
    assert corpus_type in ["train", "valid", "test"]
    pts = sorted(glob.glob(args.source_bert_data_path + '.' + corpus_type + '.[0-9]*.pt'))
    n=len(pts)
    radius=int(args.scope/2)
    count=1
    for pt in pts:
        print(count,n)
        datasets=torch.load(pt)
        new_datasets=[]
        for data_i in range(len(datasets)):
            # adding subtags operation
            sen_length = len(datasets[data_i]['clss'])
            word_length=len(datasets[data_i]['src'])
            src_sent_labels = [0 for i in range(sen_length)]
            src_sent_labels_point=[0 for i in range(sen_length)]
            src_tags=datasets[data_i]['src_tags'].copy()

            for index,idx in enumerate(datasets[data_i]['abs_art_idx']):
                src_sent_labels[idx] = 1
                flag=0#right scope
                for i in range(datasets[data_i]['clss'][idx], word_length):
                    if (datasets[data_i]['src'][i] == 101): flag += 1
                    if(flag>=radius+2):break
                    elif(flag>1):src_tags[i][index] = context
                flag=0#left scope
                i=datasets[data_i]['clss'][idx]-1
                while(i>=0):
                    if (datasets[data_i]['src'][i] == 102): flag += 1
                    if(flag>=radius+1):break
                    else:src_tags[i][index] = context
                    i-=1

                src_sent_labels_point[idx]=1
                flag = 0  # right scope
                for i in range(idx, sen_length):
                    if(flag>radius):break
                    src_sent_labels_point[i]=1
                    flag+=1
                flag = 0  # left scope
                i = idx
                while (i >= 0):
                    if (flag > radius): break
                    src_sent_labels_point[i]=1
                    flag+=1
                    i -= 1

            datasets[data_i]['src_sent_labels'] = src_sent_labels.copy()
            datasets[data_i]['src_tags'] = src_tags.copy()

            # detract operation: token-level
            detr_src=[]
            detr_tag=[]
            number=len(datasets[data_i]['abs_art_idx'])
            vacuum=[0 for num in range(number)]
            for i in range(len(datasets[data_i]['src_tags'])):
                if(src_tags[i]==vacuum):
                    continue
                detr_src.append(datasets[data_i]['src'][i])
                # detr_seg.append(dataset['segs'][i])
                detr_tag.append(datasets[data_i]['src_tags'][i])
            datasets[data_i]['src']=detr_src
            datasets[data_i]['src_tags']=detr_tag

            # detract operation: sentence-level
            detr_cls=[]
            detr_src_txt=[]
            detr_seg = []
            detr_src_sent_labels=[]
            flag=-1
            for i in range(len(datasets[data_i]['src'])):
                if(datasets[data_i]['src'][i]==101):
                    detr_cls.append(i)
                    flag=-flag
                if(flag==1):detr_seg.append(0)
                else: detr_seg.append(1)
            for i in range(sen_length):
                if(src_sent_labels_point[i]!=0):
                    detr_src_txt.append(datasets[data_i]['src_txt'][i])
                    detr_src_sent_labels.append(datasets[data_i]['src_sent_labels'][i])
            datasets[data_i]['clss'] = detr_cls
            datasets[data_i]['src_txt'] = detr_src_txt
            datasets[data_i]['segs'] = detr_seg
            datasets[data_i]['src_sent_labels']=detr_src_sent_labels

            poi=[]
            ord=[]
            for i in range(len(detr_src_sent_labels)):
                if(detr_src_sent_labels[i]==1):
                    poi.append(i)
                    ord.append(find_one(detr_tag[detr_cls[i]]))
            detr_abs_art_idx=[0 for i in range(len(datasets[data_i]['abs_art_idx']))]
            for i in range(len(ord)):
                detr_abs_art_idx[ord[i]]=poi[i]
            if(len(detr_abs_art_idx)<len(datasets[data_i]['abs_art_idx'])):
                for a in range(1,len(datasets[data_i]['abs_art_idx'])):
                    if(datasets[data_i]['abs_art_idx'][a]==datasets[data_i]['abs_art_idx'][a-1]):
                        detr_abs_art_idx[a]=detr_abs_art_idx[a-1]

            datasets[data_i]['abs_art_idx']=detr_abs_art_idx
            # sent_dropout==0.2
            for s_i, val in enumerate(datasets[data_i]['src_sent_labels']):
                if(val==1):
                    ran_n=random.randint(1,10)
                    if(ran_n>=10-sent_dropout*10+1):datasets[data_i]['src_sent_labels'][s_i]=0

            new_datasets.append(datasets[data_i])
        count += 1
        torch.save(new_datasets,args.target_bert_data_path+pt.split("/")[-1].replace("cnndm",''))



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-scope", default=5)
    parser.add_argument("-source_bert_data_path", default=r"")
    parser.add_argument("-target_bert_data_path", default=r"../bert_data/cnndm")
    parser.add_argument("-log_file", default='../logs/subtagging.log')
    args = parser.parse_args()

    set_logger(logger, args.log_file)

    subtagging(args, "train", context=0.5, sent_dropout=0.2)
    subtagging(args, "valid", context=0.5, sent_dropout=0.2)


