# BDTF-ASTE

本仓库开源了以下论文的代码：

- 标题：Boundary-Driven Table-Filling for Aspect Sentiment Triplet Extraction
- 作者：Yice Zhang∗, Yifan Yang∗, Yihui Li, Bin Liang, Shiwei Chen, Yixue Dang, Ming Yang, and Ruifeng Xu
- 会议：EMNLP-2022 Main (Long)

## 方法简介

### ASTE任务

本文要解决的是Aspect-Based Sentiment Analysis(ABSA)问题中的Aspect Sentiment Triplet Extraction(ASTE)任务。
如下图所示，ASTE的目的是抽取用户评论中表达观点的方面情感三元组，一个元组包含三个部分
- Aspect Term: 情感所针对的目标对象，一般是被评价实体（餐馆或者产品）的某个方面项，常被称作方面术语、方面词、属性词等
- Opinion Term: 具体表达情感的词或短语，常被称作情感术语、情感词等
- Sentiment Polarity: 用户针对Aspect Term所表达的情感倾向，类别空间为`{POS, NEG, NEU}`

<div align="center"> <img src="https://user-images.githubusercontent.com/9134454/199022562-2cca1c06-b91e-4e4b-8bf0-20273a16821e.png" alt="ASTE" width="50%" /></div>

### 本方法的动机

以往的方法将本任务建模为一个表格填充问题（table-filling problem）。如下图所示，二维表中的每个元素为词与词之间的关系。

<div align="center"> <img src="[https://user-images.githubusercontent.com/9134454/199022562-2cca1c06-b91e-4e4b-8bf0-20273a16821e.png](https://user-images.githubusercontent.com/9134454/199043065-86775e70-6027-4732-99b3-c49c0fd30e30.png)" alt="GTS" width="50%" /></div>


## 如何运行
### Requirements
### 


<!-- ## Citation -->
