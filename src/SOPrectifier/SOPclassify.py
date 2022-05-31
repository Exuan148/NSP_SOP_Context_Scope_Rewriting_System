import fire
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from SOPrectifier import  SOPtokenization
from SOPrectifier import SOPtokenization
from SOPrectifier import SOPmodels
from SOPrectifier import SOPoptim
from SOPrectifier import SOPtrain
from SOPrectifier.SOPdata_reader_helper import *
from SOPrectifier.SOPdata_tokenizer import *
from SOPrectifier.SOPutils import set_seeds, get_device
import pandas as pd
import numpy as np

class Classifier(nn.Module):
    """
    Classifier with Transformer
    """
    def __init__(self, cfg, n_labels):
        super().__init__()
        self.transformer = SOPmodels.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ(self.fc(h[:, 0]))
        logits = self.classifier(self.drop(pooled_h))
        return logits
cfg = SOPtrain.Config.from_json('SOPrectifier/SOP_config/train_mrpc.json')
model_cfg = SOPmodels.Config.from_json('SOPrectifier/SOP_config/bert_base.json')
set_seeds(cfg.seed)
tokenizer = SOPtokenization.FullTokenizer(vocab_file='SOPrectifier/uncased_L-12_H-768_A-12/vocab.txt', do_lower_case=True)
TaskDataset = dataset_to_class_mapping('mrpc')
pipeline = [Tokenize_data(tokenizer.convert_to_unicode, tokenizer.tokenize),
            Tokenizer_helper(128),
            Indexing_the_tokens(tokenizer.convert_tokens_to_ids,
            TaskDataset.labels, 128)]
def SOP_pre(sentenceset,srctagset,batch_src,SOPmodelname):
    windows=len(srctagset[0][0])
    sentenceset = [item.split('<q>') for item in sentenceset]
    ordered_sentenceset = []
    ordered_srctagset = []
    def batch_prob(model, batch):
        input_ids, segment_ids, input_mask, label_id = batch
        logits = model(input_ids, segment_ids, input_mask)
        _, label_pred = logits.max(1)
        probabilities=torch.sigmoid(_)
        probabilities=probabilities*(-torch.pow(-1.0,label_pred).float())
        return probabilities

    for sentences, srctags in zip(sentenceset, srctagset):
        number = len(sentences)
        with open(r'../temp.tsv', 'w', newline='') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerow(['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String'])
            # res=combination(sentences)
            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    tsv_w.writerow(['0', str(i), str(j), sentences[i], sentences[j]])
                tsv_w.writerow(['0', str(i), 'E', sentences[i], "<EOS>"])

        dataset = TaskDataset("temp.tsv", pipeline)
        data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
        model = Classifier(model_cfg, len(TaskDataset.labels))
        criterion = nn.CrossEntropyLoss()
        trainer = SOPtrain.Trainer(cfg, model, data_iter, SOPoptim.optim4GPU(cfg, model), None, get_device())
        probabilities = trainer.get_prob(batch_prob,SOPmodelname, True)
        res_ = []
        for i in range(number):
            res_.append(probabilities[i * (number + 1):(i + 1) * (number + 1)])
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

        temp_sentences =""
        pointer = {}
        for i in range(number):
            idx=rec_order[i]
            temp_sentences+=sentences[idx]
            if i<number-1: temp_sentences+="<q>"

        ordered_sentenceset.append(temp_sentences)

        temp_srctags = []
        for i in range(len(srctags)):
            t = [0 for i in range(windows)]
            for j in range(number):
                if (srctags[i][j] == 1):
                    t[rec_order[j]] = 1
            temp_srctags.append(t.copy())
        ordered_srctagset.append(temp_srctags.copy())

    return ordered_sentenceset, torch.tensor(np.array(ordered_srctagset),dtype=torch.float32).to(batch_src.device)

def main(task='mrpc', train_cfg='config/train_mrpc.json',
         model_cfg='config/bert_base.json',
         data_file='glue/ISNEXTGold/test.tsv',
         model_file="E:\Dataset\SOPrec.model_steps_15000.pt",
         pretrain_file='uncased_L-12_H-768_A-12/bert_model.ckpt',
         data_parallel=True, vocab='uncased_L-12_H-768_A-12/vocab.txt',
         save_dir='exp/ISNEXT', max_len=128, mode='eval'
         ):
# def main(task='mrpc', train_cfg='config/train_mrpc.json',
#         model_cfg='config/bert_base.json',
#         data_file='glue/ISNEXTGold/train.tsv',
#         model_file="E:\Dataset\SOPrec.model_steps_9452.pt",
#         pretrain_file="E:\Dataset\SOPrec.model_steps_9452.pt",
#         data_parallel=True, vocab='uncased_L-12_H-768_A-12/vocab.txt',
#         save_dir='exp/ISNEXTcontinue', max_len=128, mode='train'
#         ):
    """
    :param task:            dataset for which you want to run
    :param train_cfg:       json file containing params for the classification run
    :param model_cfg:       json file containing the details about BERT baase model
    :param data_file:       csv file realted the the data set
    :param model_file:
    :param pretrain_file:   pretrained model weights checkpoint
    :param data_parallel:   if we want to run data parallel
    :param vocab:           vocab file "uses the vocab file which came along with the bet uncased weights"
    :param save_dir:        path to save the checkpoints
    :param max_len:         maximum sequence length
    :param mode:            train, validation or test
    :return:
    """

    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)
    set_seeds(cfg.seed)
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    TaskDataset = dataset_to_class_mapping(task)
    pipeline = [Tokenize_data(tokenizer.convert_to_unicode, tokenizer.tokenize),
                Tokenizer_helper(max_len),
                Indexing_the_tokens(tokenizer.convert_tokens_to_ids,
                                    TaskDataset.labels, max_len)]
    dataset = TaskDataset(data_file, pipeline)
    data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = Classifier(model_cfg, len(TaskDataset.labels))
    criterion = nn.CrossEntropyLoss()

    trainer = train.Trainer(cfg, model, data_iter, optim.optim4GPU(cfg, model), save_dir, get_device())

    if mode == 'train':
        def get_loss(model, batch, global_step):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits, label_id)
            return loss

        trainer.train(get_loss, model_file, pretrain_file, data_parallel)

    elif mode == 'eval':
        def evaluate(model, batch):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            _, label_pred = logits.max(1)
            result = (label_pred == label_id).float()
            accuracy = result.mean()
            return accuracy, result

        results = trainer.eval(evaluate, model_file, data_parallel)
        total_accuracy = torch.cat(results).mean().item()
if __name__ == '__main__':
    fire.Fire(main)

