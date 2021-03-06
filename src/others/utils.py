import os
import re
import shutil
import time

from others import pyrouge

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def process(params):
    temp_dir, data = params
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}-{}".format(current_time, pool_id))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


# def test_rouge(temp_dir, cand, ref):
#     candidates = [line.strip() for line in open(cand, encoding='utf-8')]
#     references = [line.strip() for line in open(ref, encoding='utf-8')]
#     # print(candidates,references)
#     assert len(candidates) == len(references)
#
#     cnt = len(candidates)
#     current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
#     tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
#     if not os.path.isdir(tmp_dir):
#
#         os.mkdir(tmp_dir)
#         os.mkdir(tmp_dir + "/candidate")
#         os.mkdir(tmp_dir + "/reference")
#         print(68)
#     try:
#
#         total = 0
#         for i in range(cnt):
#             if len(references[i]) < 1:
#                 continue
#             with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
#                       encoding="utf-8") as f:
#                 f.write(candidates[i])
#             with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
#                       encoding="utf-8") as f:
#                 f.write(references[i])
#             total += 1
#         r = pyrouge.Rouge155(temp_dir=temp_dir)
#         r.model_dir = tmp_dir + "/reference/"
#         r.system_dir = tmp_dir + "/candidate/"
#         r.model_filename_pattern = 'ref.#ID#.txt'
#         r.system_filename_pattern = r'cand.(\d+).txt'
#         rouge_results = r.convert_and_evaluate()
#         # print(rouge_results)
#         results_dict = r.output_to_dict(rouge_results)
#         results_dict['samples'] = total
#         results_dict['results_str'] = str(rouge_results)
#         print(91)
#     finally:
#         pass
#         if os.path.isdir(tmp_dir):
#             shutil.rmtree(tmp_dir)
#         print(98)
#     return results_dict
from sumeval.metrics.rouge import RougeCalculator
def test_rouge(temp_dir, cand, ref):
    candidates = [line.strip() for line in open(cand, encoding='utf-8')]
    references = [line.strip() for line in open(ref, encoding='utf-8')]

    rouge = RougeCalculator(stopwords=True, lang="en")
    results_dict={"rouge_1_f_score":0,"rouge_2_f_score":0,"rouge_l_f_score":0}
    ct=0
    for can,ref in zip(candidates,references):
        ct+=1
        results_dict["rouge_1_f_score"]+=rouge.rouge_n(summary=can,references=[ref],n=1)
        results_dict["rouge_2_f_score"]+=rouge.rouge_n(summary=can,references=[ref],n=2)
        results_dict["rouge_l_f_score"]+=rouge.rouge_l(summary=can,references=[ref])
    results_dict["rouge_1_f_score"]/=ct
    results_dict["rouge_2_f_score"]/=ct
    results_dict["rouge_l_f_score"]/=ct
    results_dict["samples"]=ct
    return results_dict




def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

def rouge_results_to_str(results_dict):
    return ">> Samples: {:d}\nROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\n\n".format(
        results_dict["samples"],
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100
        # results_dict['results_str']
    )

def avg_rouge_f1(results_dict):
    r1 = results_dict["rouge_1_f_score"] * 100
    r2 = results_dict["rouge_2_f_score"] * 100
    rl = results_dict["rouge_l_f_score"] * 100
    return (r1 + r2 + rl) / 3
