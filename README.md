# 毕业设计——一种文本匿名化方法的研究和实现

# 项目简介

本项目构建了一个匿名化系统，其主要包括两个部分。第一部分是个人身份信息（PII）的识别，
第二部分是对个人身份信息进行假名化处理，并且保证处理后的数据仍然具有一定的可用性。

[TOC]



# 匿名化系统

## 个人身份信息（PII）的识别

本部分主要比较了三种方法在PII识别任务上的效果，分别是使用现有的自然语言库Spacy的命名实体识别（NER）模块，
现有的框架Flair的NER模块，以及使用Longformer模型进行PII识别。 在实验中，
我们使用了一个被设计用来检测匿名化好坏的文本数据集（The Text Anonymization Benchmark）对三种方法进行了实验，
比较了它们在PII识别任务上的效果。

#### 1.1 Spacy

Spacy是一个用于自然语言处理的Python库，它提供了一些用于文本处理的工具，包括命名实体识别（NER）模块。
在实验中，我们使用Spacy的NER模块对PII进行识别，然后与数据集中标注的PII信息进行比较，计算相关指标。
由于Spacy的NER模块只能识别一些基本的实体类型，如人名、地名、组织名等，因此在实验中我们只考虑这些实体类型。
PERSON：人名 LOC：地名 ORG：组织名 
对于CODE：个人身份证号，车牌号，手机号这类PII信息，Spacy无法识别，在实验中我们使用了正则化方法来尽可能识别。
详情见checkpoint/Spacy_method.py

#### 1.2 Flair

Flair是一个用于自然语言处理的Python库，它提供了一些用于文本处理的工具，包括命名实体识别（NER）模块。
在实验中，我们使用Flair的NER模块对PII进行识别，然后与数据集中标注的PII信息进行比较，计算相关指标。
虽然Flair的NER模块可以识别更多的实体类型，如人名、地名、组织名、日期、时间等，但是日期，时间等其他信息对于个人身份信息的识别帮助不大，
所以我们还是只考虑一下实体类型：
PERSON：人名 LOC：地名 ORG：组织名 
对于CODE：个人身份证号，车牌号，手机号这类PII信息，Spacy无法识别，在实验中我们使用了正则化方法来尽可能识别。
详情见checkpoint/Flair_method.py

#### 1.3 Longformer

Longformer是一种可高效处理长文本的模型，出自Allen Institute for Artificial Intelligence（AI2）在2020年4月10日发表的一篇论文。它是为长文本定制的Transformer变体，旨在解决传统Transformer模型在处理长文本时存在的内存瓶颈问题。Longformer提出了一种时空复杂度同文本序列长度呈线性关系的Self-Attention机制，使得模型能够以更低的时空复杂度建模长文档。Longformer在高难度阅读理解任务上表现优秀，如在TriviaQA、Wikihop和HotpotQA等任务中取得了优异的成绩。在实验中，我们使用Longformer模型对PII进行识别，然后与数据集中标注的PII信息进行比较，计算相关指标。

详情见checkpoint/Longformer_medol.py(创建了一个model) checkpoint/data_handing.py(处理数据，对齐数据) 
checkpoint/data_manipulation.py(处理匿名化文本) checkpoint/train_model.py(训练模型，打印预测结果)
详见Myreadme.md

#### 1.4修改后的longformer 

由于原始的longformer模型只能输出实体的起始位置和结束位置，而不能输出实体的类型，所以我们对longformer模型进行了修改，
使其能够输出实体的类型。具体来说，在数据预处理阶段，将匿名化数据集中对实体类型的标注加入到模型的输入中，并且修改了模型的输出层。
详见Myreadme.md，Longformer_medol.py，data_handing.py，data_manipulation.py，train_model.py

####  1.5 评价指标

​		由于评价脚本只需要标准语料库，和预测匿名化结果文件，所以可以直接使用评价脚本进行评价。同时由		于脚本要求，在checkpoint文件夹中存储上述三种方法中，输出结果都只是需要匿名化的实体在文本中的起始		位置和结束位置元组，没有包括其实体类型，和实体文本信息，详见Myreadme.md。

## 2.对个人身份信息进行假名化处理

​		本部分主要是对PII识别的结果进行处理，使得PII信息被替换为伪造的信息，从而保护用户的隐私。
​		根据对第一部分的比较，发现在PII识别任务上，使用匿名化测试集（The Text Anonymization Benchmark）进		行训练的Longformer模型的效果最好，于是我们选择使用Longformer模型对PII进行识别，然后使用两种方法		对识别到的信息进行替换。

#### 	      2.1使用随机生成的假名替换PII信息

​		在这里使用了一个可以随机生成假名信息的数据库faker，通过对longformer模型识别到的实体类型标签，调用		对应的faker函数，
​		来生成对应实体类型标签下的生成的假名信息替换PII信息。
​		详见 run_longmodel.py

#### 		2.2使用轮换方法替换PII信息

​		这里将longformer模型识别到的同一类型的实体进行分组，然后对每一组实体进行轮换替换，即将某一类型的	实体替换为同类型分组下的另一个实体。
​		详见 run_longmodel.py

#### 		2.3假名化实验评价指标

​			longformer模型识别PII+两种方法实现假名化，我们构建出两个系统
​			使用上述匿名化系统对IMDB数据集进行匿名化处理，然后使用Myclassifier.py对三个数据集即原始数据，			faker假名化后的数据，轮换假名化数据进行分类。
​			比较同一个bert模型在三个不同数据集上的precision，recall，f1-score，accuracy等指标，来评价匿名化系			统的效果。详见MYclassifier.py

​		使用evaluation.py ，由于评价脚本只需要标准语料库，和预测匿名化结果文件，所以可以直接使用评价脚本进		行评价。
​		同时由于脚本要求，在checkpoint文件夹中存储上述三种方法中，输出结果都只是需要匿名化的实体在文本中		的起始位置和结束位置元组，没有包括其实体类型，和实体文本信息，详见Myreadme.md。