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

### 以往方法的问题

以往的方法将本任务建模为一个表格填充问题（table-filling problem）。如下图所示，二维表中的每个元素为词与词之间的关系。该方法首先通过对角线抽取aspect和opinion，然后通过aspect和opinion定位对应的关系区域，通过投票的方法是确定aspect和opinion之间的关系。该方法存在诸多问题。比较明显的问题有两个：
1. 关系不一致：将aspect和opinion之间的关系分解为词与词之间的关系，这会带来潜在的关系不一致问题。
2. 边界不敏感：如果aspect或者opinion的边界出现了小错误，二者的关系预测的结果大概率不会改变，这就是的模型产生了边界错误的输出，如('dogs', 'top notch', `POS`)。

<div align="center"> <img src="https://user-images.githubusercontent.com/9134454/199043065-86775e70-6027-4732-99b3-c49c0fd30e30.png" alt="GTS" width="50%" /></div>

以往的工作尝试使用Span-based的方法来解决关系不一致的问题。这是一种可行的思路。但是该方法忽略了细粒度的词级别的信息，这正是表格填充方法的优点。

## 本文提出方法的简介

本文为了解决上述的两个问题，提出了边界驱动的表格填充方法（Boundary-Driven Table-Filling）。如下图所示，该方法将方面关系三元组转为二维表中的一个关系区域，因而将ASTE任务转化为关系区域的定位和分类。对关系区域整体进行分类可以解决了关系不一致的问题，那些边界错误的关系区域也可以通过将其分类为Invaild而移除。

<div align="center"> <img src="https://user-images.githubusercontent.com/9134454/199046656-e45f508e-b196-4ce4-a649-19cd4582dee0.png" alt="BDTF" width="50%" /></div>

此外，本文还提出了一种关系学习的方法来学习一个二维的表示。该方法包含三个部分：
- 首先，将评论文本输入到`BERT`中学习词级别的上下文表示。
- 然后，通过基于张量的操作，根据词表示构建关系表示。文本中所有词之间的关系表示构成一个二维的表，表中的元素为一个向量。
- 最后，使用CNN对二维表进行建模。
该方法学习到的二维表示将被用到关系区域的定位和分类中。

整体上，本文所提出方法的模型框架如下图所示。

<div align="center"> <img src="https://user-images.githubusercontent.com/9134454/199048478-82c2c1ff-1b10-41aa-8071-5f4ad6197559.png" alt="Model" width="40%" /></div>

### 实验结果

本方法的主要实验结果如下表，详细的分析见论文。

<div align="center"> <img src="https://user-images.githubusercontent.com/9134454/199048765-b85e7c6a-04f2-4d40-aec5-2ccf73709f81.png" alt="Result" width="40%" /></div>

## 如何运行
### Requirements
### 


<!-- ## Citation -->
