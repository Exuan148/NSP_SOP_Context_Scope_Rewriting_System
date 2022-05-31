import argparse
import logging
from others.logging import set_logger, init_logger
import torch
import random
import json
import csv
logger = logging.getLogger(__name__)

def To_make_SOPGold(args,type,surfix,data_len):
    assert type in ["test", "train", "dev"]
    data = torch.load(args.source_bert_data_path + "\\" + surfix)
    split_num = data_len // 2

    def pos_scratch(l):
        assert len(l) > 1
        l.append("[EOS]")
        sent1 = random.randint(0, len(l) - 2)
        return l[sent1], l[sent1 + 1]

    idxs = [i for i in range(data_len)]
    random.shuffle(idxs)
    with open(args.target_bert_data_path + '\\' + type + '.tsv', 'a+', newline='', encoding="utf-8") as f:
        tsv_w = csv.writer(f, delimiter='\t')
        for i in idxs[0:split_num]:
            l=data[i]["tgt_txt"].split("<p>")
            sentence1, sentence2 = pos_scratch(l)
            tsv_w.writerow(['1', "XX", 'XX', sentence1, sentence2])

        for i in idxs[split_num:split_num * 2]:
            l = data[i]["tgt_txt"].split("<p>")
            sentence2, sentence1 = pos_scratch(l)
            tsv_w.writerow(['0', "XX", 'XX', sentence1, sentence2])

def To_make_SOP(args,type,surfix,data_len):
    assert type in ["test", "train", "dev"]
    data = torch.load(args.source_bert_data_path + "\\" + surfix)
    split_num = data_len // 2

    def pos_scratch(l):
        assert len(l) > 1
        l.append("[EOS]")
        sent1 = random.randint(0, len(l) - 2)
        return l[sent1], l[sent1 + 1]

    idxs = [i for i in range(data_len)]
    random.shuffle(idxs)
    with open(args.target_bert_data_path+'\\'+type+'.tsv', 'a+', newline='',encoding="utf-8") as f:
        tsv_w = csv.writer(f, delimiter='\t')
        for i in idxs[0:split_num]:
            sentence1, sentence2 = pos_scratch(data[i]["src_txt"])
            tsv_w.writerow(['1', "XX", 'XX', sentence1, sentence2])

        for i in idxs[split_num:split_num * 2]:
            sentence2, sentence1 = pos_scratch(data[i]["src_txt"])
            tsv_w.writerow(['0', "XX", 'XX', sentence1, sentence2])

def To_make(args,type,surfix,data_len):
    assert type in ["test","train","dev"]
    data=torch.load(args.source_bert_data_path+"\\"+surfix)
    split_num=data_len//2
    data_dict={"id": 0, "sentence1": "", "sentence2": "", "label": ""}
    data_list=[]
    def pos_scratch(l):
        assert  len(l)>1
        sent1=random.randint(0, len(l)-2)
        return l[sent1],l[sent1+1]
    def neg_scratch(l):
        assert  len(l) > 2
        sent1 = random.randint(0, len(l) - 2)
        sentence1=l[sent1]
        del l[sent1]
        del l[sent1]
        sent2=random.randint(0, len(l)-1)
        return sentence1,l[sent2]
    idxs=[i for i in range(data_len)]
    random.shuffle(idxs)
    for i in idxs[0:split_num]:
        data_dict["id"]=i
        sentence1,sentence2=pos_scratch(data[i]["src_txt"])
        data_dict["sentence1"]=sentence1
        data_dict["sentence2"]=sentence2
        data_dict["label"]="1"
        data_list.append(data_dict.copy())

    for i in idxs[split_num:split_num*2]:
        data_dict["id"] = i
        sentence2, sentence1 = pos_scratch(data[i]["src_txt"])
        data_dict["sentence1"] = sentence1
        data_dict["sentence2"] = sentence2
        data_dict["label"] = "0"
        data_list.append(data_dict.copy())

    array = sorted(data_list,key=lambda x:x["id"])
    filename = args.target_bert_data_path+"\\"+type+'.json'
    with open(filename, 'w') as f_obj:
        json.dump(array, f_obj,indent=1, separators=(',', ': '))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # split bert data
    # train:1,dev:2,test:3
    # parser.add_argument("-data_len", default=1600)
    parser.add_argument("-source_bert_data_path", default=r"E:\PycharmProject\paper2\Presum\PreSumm-master\bert_data")
    parser.add_argument("-target_bert_data_path", default=r"E:\PycharmProject\paper2\BERT-with-SOP-master\BERT-with-SOP-master\glue\ISNEXTGold")
    # parser.add_argument("-target_bert_data_path", default=r"E:\PycharmProject\paper2\BERT-with-SOP-master\BERT-with-SOP-master\glue\ISNEXT")
    # parser.add_argument("-target_bert_data_path", default=r"E:\PycharmProject\paper2\src\NSPrectifier-BERT-main\datasets\cnndm_isnext")
    parser.add_argument("-log_file", default='../logs/SOP_datamaking.log')
    parser.add_argument("-self",default=True,help="whether the negative sets are chosen from the same sentence")
    args = parser.parse_args()
    set_logger(logger, args.log_file)
    # To_make(args,"train","cnndm.train.1.bert.pt",200)
    # To_make(args,"dev","cnndm.train.2.bert.pt",200)
    # To_make(args,"test","cnndm.train.3.bert.pt",1600)
    for i in range(25):
        print(i)
        To_make_SOP(args,"train","cnndm.train."+str(i)+".bert.pt",200)
    for i in range(5):
        print(25+i)
        To_make_SOP(args,"test","cnndm.train."+str(25+i)+".bert.pt",200)