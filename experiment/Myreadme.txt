###longformer的四个文件
这四个Python文件是一个用于识别和遮蔽个人身份信息(如姓名、地址等)的机器学习模型的实现。它们的主要功能如下:
data_handling.py
定义了用于处理训练数据的类和函数
包括对输入文本进行标记对齐、划分成小批数据等操作
注意对齐操作，里面 I B O
data_manipulation.py
读取原始JSON格式的训练、开发和测试数据集
对数据进行预处理,将不同标注者的标注结果合并为一个标注
将label qusi和direct合并为MASK 将NOmask 合并为NOmask
longformer_model.py
定义了基于Longformer预训练模型的序列标注模型架构
在Longformer的输出上添加了一个线性分类层用于遮蔽识别
train_model.py
加载数据、初始化模型、训练模型的主脚本
在开发集上评估模型性能
在测试集上进行预测并将结果保存为JSON文件
保存最终训练好的模型参数
总的来说,这是一个端到端的个人身份信息识别和隐私保护系统,可用于自动检测和遮蔽文本中的敏感个人信息。它使用Longformer模型来处理长文本输入。

###evaluation.py脚本

该Python脚本用于评估文本匿名化系统的性能表现。它可以计算多个评估指标,包括令牌级、mention级和实体级的精确率和召回率,并提供详细的错误分析信息。
用法:
python evaluation.py <gold_standard_file> <masked_output_file> [options]
必需参数:
<gold_standard_file>    包含标准注释的JSON文件路径
<masked_output_file>    一个或多个包含系统输出掩码文本范围的JSON文件路径

可选参数:
--use_bert              使用BERT语言模型计算每个标识符的信息权重
--only_docs <doc_id1> <doc_id2>...
                        仅对指定的文档ID进行评估
--verbose               打印出未被正确掩码的mention,便于错误分析

输出:
该脚本将打印出以下评估指标:
- Token级别的召回率(overall和按实体类型分类)
- Mention级别的召回率
- 实体级别的召回率(直接标识符和准标识符)
- Token级别的精确率(均匀加权和BERT加权)
- Mention级别的精确率(均匀加权和BERT加权)

注意事项:
1. 该脚本假定输入的JSON文件格式符合特定的标准。
2. 如果使用--use_bert选项,将需要下载BERT模型(约500MB)。
3. 将为每个提供的masked_output_file输出一个单独的评估结果。

该评估脚本基于文本匿名化注释指南和标准语料库。有关详细信息,请参阅文件中的代码注释。

###修改longformer模型以适合后续下流任务
将longformer模型修改为能够识别实体标签的模型
change1 data_manipulation
将标签扩展PERSONMASK，CODEMASK，LOCMASK，ORGMASK，DEMMASK，DATETIMEMASK，QUANTITYMASK，MISCMASK
change2 修改train_model.py
将label_set = LabelSet(labels=['PERSONMASK', 'CODEMASK', 'LOCMASK', 'ORGMASK',
                             'DEMMASK', 'DATETIMEMASK', 'QUANTITYMASK', 'MISCMASK'])
change3 修改data_handling 最下面的对齐函数
        if anno['label'] != 'NO_MASK':
            annotation_token_ix_set = (
                set()
            ) 
change4 修改train model里面的 try的BIO识别函数
change5 该train model里面的交叉熵