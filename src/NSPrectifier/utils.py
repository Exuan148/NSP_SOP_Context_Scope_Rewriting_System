import csv
import json
import random

class Datasets():
    def __init__(self, dataset_name=""):
        self.dataset_name = dataset_name
        self.patterns = []
        self.train_path, self.dev_path, self.test_path = "", "", ""

        if (dataset_name == "cnndm_isnext"):
            # self.train_path = r"datasets/cnndm_isnext/train.json"
            # self.dev_path = r"datasets/cnndm_isnext/dev.json"
            # self.test_path = r"datasets/cnndm_isnext/test.json"
            self.metric = 'Acc'
            self.labels = [0, 1]

    def load_data(self, filename, sample_num=-1, is_train=False, is_shuffle=False):
        D = []

        if (self.dataset_name == "cnndm_isnext"):
            with open(filename, encoding='utf-8') as f:
                array=eval(f.read())
                for l in array:
                    sentence1 = l['sentence1']
                    sentence2 = l['sentence2']
                    label = l['label']
                    text = "{}[SEP]{}".format(sentence1, sentence2)
                    D.append((text, int(label)))

        # Shuffle the dataset.
        if (is_shuffle):
            random.seed(1)
            random.shuffle(D)

        # Set the number of samples.
        if (sample_num == -1):
            # -1 for all the samples
            return D
        else:
            return D[:sample_num + 1]

class Model():
    def __init__(self, model_name=""):
        self.model_name = model_name
        self.config_path, self.checkpoint_path, self.dict_path = "", "", ""

        if (model_name == "uer-mixed-bert-base"):
            # self.config_path = './models/uer_mixed_corpus_bert_base/bert_config.json'
            self.config_path = 'NSPrectifier/models/uer_mixed_corpus_bert_base/bert_config.json'
            # self.checkpoint_path = './models/uer_mixed_corpus_bert_base/bert_model.ckpt'
            self.checkpoint_path = 'NSPrectifier/models/uer_mixed_corpus_bert_base/bert_model.ckpt'
            # self.dict_path = './models/uer_mixed_corpus_bert_base/vocab.txt'
            self.dict_path = 'NSPrectifier/models/uer_mixed_corpus_bert_base/vocab.txt'


def read_labels(label_file_path):
    labels_text = []
    text2id = {}
    with open(label_file_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f.readlines()):
            label = line.strip('\n')
            labels_text.append(label)
            text2id[label] = index
    return labels_text, text2id


def sample_dataset(data: list, k_shot: int, label_num=-1):
    if(k_shot==-1):
        return data
    label_set = set()
    label2samples = {}
    for d in data:
        (text, label) = d
        label_set.add(label)
        if (label in label2samples):
            label2samples[label].append(d)
        else:
            label2samples[label] = [d]
    if (label_num != -1):
        assert len(label_set) == label_num
    new_data = []
    for label in label_set:
        if (isinstance(label, float)):
            random.seed(0)
            new_data = random.sample(data, k_shot)
            random.shuffle(new_data)
            return new_data
        random.seed(0)
        new_data += random.sample(label2samples[label], k_shot)
    random.seed(0)
    random.shuffle(new_data)
    return new_data

# if __name__ == "__main__":
#     print()
