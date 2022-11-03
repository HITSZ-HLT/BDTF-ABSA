[**中文说明**](https://github.com/HITSZ-HLT/BDTF-ASTE/) | [**English**](https://github.com/HITSZ-HLT/BDTF-ASTE/blob/master/README_EN.md)

# BDTF-ASTE

This repository releases the code of the following paper:

- Title: Boundary-Driven Table-Filling for Aspect Sentiment Triplet Extraction
- Authors: Yice Zhang∗, Yifan Yang∗, Yihui Li, Bin Liang, Shiwei Chen, Yixue Dang, Ming Yang, and Ruifeng Xu
- Conference: EMNLP-2022 Main (Long)

## Introduction

### The ASTE Task

The task that this paper addresses is Aspect Sentiment Triplet Extraction (ASTE), which is an important task in Aspect-Based Sentiment Analysis(ABSA).
As shown in the figure below, ASTE aims to extract the aspect terms along with the corresponding opinion terms and the expressed sentiments in the review.
Specifically, a triplet is defined as (aspect term, opinion term, sentiment polarity):
- **Aspect term**: the target of an opinion, usually an aspect of an entity (a restaurant or product).
- **Opinion term**: the word or phrase that specifically expresses the sentiment.
- **Sentiment polarity**: a specific category in `{POS, NEG, NEU}`.

<div align="center"> <img src="https://user-images.githubusercontent.com/9134454/199022562-2cca1c06-b91e-4e4b-8bf0-20273a16821e.png" alt="ASTE" width="50%" /></div>

### Previous Methods and Limitations

Previous methods tackle the ASTE task through a table-filling approach, where the triplets are represented by a two-dimensional (2D) table of word-pair relations. 
In this approach, aspect terms and opinion terms are extracted through the diagonal elements of the table, and sentiments are treated as relation tags that are represented by the non-diagonal elements of the table.
This formalization enables joint learning of different subtasks in ASTE, achieving superior performance over the pipeline approach.

However, the previous table formalization suffers from relation inconsistency and boundary insensitivity when dealing with multi-word aspect terms and opinion terms. It decomposes the relation between an aspect term and an opinion term into the relations between the corresponding aspect words and opinion words.
In other words, a term-level relation is represented by several wordlevel relation tags. The relation tags in the table are assigned independently, which leads to potential inconsistencies in the predictions of the wordlevel relations. 
In addition, when there are minor boundary errors in the aspect term or opinion term, the voting result for the term-level relation may stay unchanged, encouraging the model to produce wrong predictions. Reseachers try to solve this problem through a span-based method, but their method discards fine-grained word-level information, which is the advantage of the table-filling approach.

<div align="center"> <img src="https://user-images.githubusercontent.com/9134454/199043065-86775e70-6027-4732-99b3-c49c0fd30e30.png" alt="GTS" width="50%" /></div>

### Proposed Approach

This paper proposes a Boundary-Driven Table-Filling (BDTF) approach for ASTE to overcome the above issues.
In BDTF, a triplet is represented as a relation region in the 2D table, which is shown in the figure below.
In this way, it extracts triplets by directly detecting and classifying the relation regions in a 2D table. 
Classification over the entire relation region ensures relation consistency, and those relation regions with boundary errors can be removed by being classified as invalid.

<div align="center"> <img src="https://user-images.githubusercontent.com/9134454/199046656-e45f508e-b196-4ce4-a649-19cd4582dee0.png" alt="BDTF" width="50%" /></div>

In addition, this paper also develops an effective relation representation learning approach to learn the table representation. 
This consists of three parts: 
- We first learn the word-level contextualized representations of the input review through a pre-trained language model. 
- Then we adopt a tensor-based operation to  construct the relation-level representations to fully exploit the word-to-word interactions. 
- Finally, we model relation-to-relation interactions through a multi-layer convolution-based encoder to enhance the relation-level representations. 

The relation representations of each two words in the review together form a 2D relation matrix, which serves as the table representation for BDTF.

The proposed approach is briefly present in the figure below.

<div align="center"> <img src="https://user-images.githubusercontent.com/9134454/199048478-82c2c1ff-1b10-41aa-8071-5f4ad6197559.png" alt="Model" width="40%" /></div>

### Experimentual Results

The main results are listed in the table below. See the paper for a detailed analysis.

<div align="center"> <img src="https://user-images.githubusercontent.com/9134454/199048765-b85e7c6a-04f2-4d40-aec5-2ccf73709f81.png" alt="Result" width="80%" /></div>

## How to Run

### Requirements

- transformers==4.15.0
- pytorch==1.7.1
- einops=0.4.0
- torchmetrics==0.7.0
- tntorch==1.0.1
- pytorch-lightning==1.3.5

### Files

```
├── code
│   ├── utils
│   │   ├── __init__.py
│   │   ├── aste_datamodule.py
|   |   └── aste_result.py
│   ├── model
│   │   ├── seq2mat.py
│   │   ├── table.py
│   │   ├── table_encoder
│   │   |   └── resnet.py
|   |   └── bdtf_model.py
|   ├── aste_train.py
|   └── bash
│       ├── aste.sh
│       ├── aste_14res.sh
│       ├── aste_14lap.sh
│       ├── aste_15res.sh
|       └── aste_16res.sh
└── data
    └── aste_data_bert
        ├── V1
        │   ├── 14res
        |   │   ├── train.json
        |   │   ├── dev.json
        |   │   └── test.json
        │   ├── 14lap/...
        │   ├── 15res/...
        |   └── 16res/...
        └── V2/...
```

### Run our code!

Enter `code` and
- execute `chmod +x bash/*`,
- execute `bash/aste_14lap.sh`.

Result of aste_14lap.sh (Random seed is set to be 40 and the computing device is A100):

<div align="center"> <img src="https://user-images.githubusercontent.com/9134454/199077758-21eeedc2-c4f2-49e4-a332-813a000d9047.png" alt="Result" width="60%" /></div>

Result of aste_14lap.sh (Random seed is set to be 40 and the computing device is V100):

<div align="center"> <img src="https://user-images.githubusercontent.com/9134454/199708850-5d1ff9a0-4fa2-4c51-afff-813377415ae1.png" alt="Result2" width="75%" /></div>

Note that the performance posted in the paper is the average results of 5 run with 5 different random seeds, which has some differences from a single run.

## If you have any questions, please raise an `issue` or contact me

- email: `zhangyc_hit@163.com`




