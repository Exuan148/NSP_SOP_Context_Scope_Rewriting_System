# Data Preparation

Download the [pre-processed CNN/DM](https://drive.google.com/file/d/173_3qIV_A0pURh130dDfL-P1A4L_KFEE/view) and extract it into bert_data folder.

Or you can pre-process the data by yourself following below steps.

Steps: 
1) Following [PreSumm](https://github.com/nlpyang/PreSumm) for preparing initial data and put it into folder ./source_bert_data.
2) Run the command to convert ./source_bert_data to ./bert_data:
```
    python src/prepro/data_builder.py
``` 

# Data Sub-tagging

After Data Preparation, we trim and sub-tag both the training and validation datasets to fit our Context-Scope model. You can run the following 
command or download our extant data via given link.

Steps:
Either Run the command:
```
   python src/subtagging.py --scope=5 --source_data_path='path1' 

```
where the path1 should be the CNN/DM data

OR Download :
   scope-3 data's link :(https://pan.baidu.com/s/1WFNwIQ_GyumHNEE2-FDnyw) verification code: aaaa 
   scope-5 data's link :(https://pan.baidu.com/s/1sPKVYwPMsCHJFsj9iGYIAQ) verification code: aaaa 
   scope-7 data's link :(https://pan.baidu.com/s/1rFowLqmoHouZe1_EIOAG3w) verification code: aaaa 
   scope-9 data's link :(https://pan.baidu.com/s/1Z4r-wSqAaIapp4N86NS5DA) verification code: aaaa 

 