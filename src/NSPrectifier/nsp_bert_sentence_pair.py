from tqdm import tqdm
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from NSPrectifier.utils import *
import time
import copy
import json
import itertools
import torch
import numpy as np
time_start = time.time()
maxlen = 256  # The max length 128 is used in our paper
batch_size = 40  # Will not influence the results
dataset_name = 'cnndm_isnext'
model_name = 'uer-mixed-bert-base'
is_pre = False
bert_model = Model(model_name=model_name)
dataset = Datasets(dataset_name=dataset_name)

tokenizer = Tokenizer(bert_model.dict_path, do_lower_case=True)
model = build_transformer_model(config_path=bert_model.config_path, checkpoint_path=bert_model.checkpoint_path, with_nsp=True,)
class data_generator(DataGenerator):
    """Data Generator"""
    def __init__(self, is_pre=True, is_mask=False, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.is_pre = is_pre
        self.is_mask = is_mask

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            text_1, text_2 = text.split('[SEP]')

            if (self.is_pre):
                token_ids, segment_ids = tokenizer.encode(first_text=text_2, second_text=text_1, maxlen=maxlen)
            else:
                token_ids, segment_ids = tokenizer.encode(first_text=text_1, second_text=text_2, maxlen=maxlen)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def pro(data_generator, data, label_num, note=""):
    logits = []
    id = 0
    for (x, _) in tqdm(data_generator):
        outputs = model.predict(x)
        for out in outputs:
            logit_pos = out[0].T
            logits.append(logit_pos)
    return logits

def NSP_pre(sentenceset,srctagset,batch_src):
    windows = len(srctagset[0][0])
    sentenceset=[item.split('<q>') for item in sentenceset]
    ordered_sentenceset=[]
    ordered_srctagset=[]
    for sentences,srctags in zip(sentenceset,srctagset):
        array = []
        number = len(sentences)
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                array.append(
                    {"id": str(i) + str(j), "sentence1": sentences[i], "sentence2": sentences[j], "label": "0"})
            array.append({"id": str(i) + str(j), "sentence1": sentences[i], "sentence2": "[EOS]", "label": "0"})
        with open("temp.json", 'w') as f_obj:
            json.dump(array, f_obj, indent=1, separators=(',', ': '))

        dev_data = dataset.load_data("temp.json", sample_num=-1)  # -1 for all the samples
        dev_data = sample_dataset(dev_data, -1)
        dev_generator = data_generator(is_pre=is_pre, data=dev_data, batch_size=batch_size)
        res = pro(dev_generator, dev_data, 2, note="Dev Set")
        res_ = []
        for i in range(number):
            res_.append(res[i * (number + 1):(i + 1) * (number + 1)])
        res_.append([0 for i in range(number + 1)])
        s = [i for i in range(number)]
        search = list(itertools.permutations(s, number))
        for i in range(len(search)):
            temp = list(search[i])
            temp.append(number)
            search[i] = tuple(temp)
        outcome = {}
        for combination in search:
            prob = 1
            if (combination[0] == number - 1): continue
            for i in range(len(combination) - 1):
                prob *= res_[combination[i]][combination[i + 1]]
            outcome[combination] = prob

        outcome = sorted(outcome.items(), key=lambda x: x[1], reverse=True)
        rec_order = list(outcome[0][0])
        rec_order.pop(-1)

        temp_sentences=[]
        pointer={}
        for idx in rec_order:
            temp_sentences.append(sentences[idx])
        ordered_sentenceset.append(temp_sentences.copy())

        temp_srctags=[]
        for i in range(len(srctags)):
            t = [0 for i in range(windows)]
            for j in range(number):
                if(srctags[i][j]==1):
                    t[rec_order[j]]=1
            temp_srctags.append(t.copy())
        ordered_srctagset.append(temp_srctags.copy())

    temp_exts = []
    for i in range(len(ordered_sentenceset)):
        temp_ext = ""
        for j in range(len(ordered_sentenceset[i])):
            temp_ext += ordered_sentenceset[i][j]
            if (j < len(ordered_sentenceset[i]) - 1):
                temp_ext += "<q>"
        temp_exts.append(temp_ext)
    ordered_sentenceset = temp_exts.copy()

    return ordered_sentenceset,torch.tensor(np.array(ordered_srctagset),dtype=torch.float32).to(batch_src.device)










# if __name__ == "__main__":
#     sentence1 = "I've never see a diamond in the flesh, I cut my teeth on wedding rings in the movies."
#     sentence2 = "And I'm not proud of my address, in the torn up town, no post code envy."
#     sentence3 = "But every songs like gold teeth, bald goons, trashing in the hotel room."
#     sentence4 = "We don't care, we're driving Cadillac in our dream."
#     sentence5 = "You'll never be royal."
#     NSP_pre([sentence1, sentence2, sentence3, sentence4, sentence5])

