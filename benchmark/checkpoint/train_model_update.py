from typing_extensions import TypedDict
import torch.nn.functional as F
from typing import List, Any
from transformers import LongformerTokenizerFast
from tokenizers import Encoding
import itertools
from torch import nn
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import json
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from data_handling import *
from longformer_model import Model
from data_manipulation import training_raw, dev_raw, test_raw
import collections
import random
import argparse
from collections import OrderedDict

if __name__ == "__main__":

    bert = "allenai/longformer-base-4096"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 设置使用的Bert模型和运行设备

    tokenizer = LongformerTokenizerFast.from_pretrained(bert)  # 分词器
    label_set = LabelSet(labels=['PERSONMASK', 'CODEMASK', 'LOCMASK', 'ORGMASK',
                                 'DEMMASK', 'DATETIMEMASK', 'QUANTITYMASK', 'MISCMASK'])  # 扩展了的标签集
    # 初始化了longformer分词器和标签集

    training = Dataset(data=training_raw, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=4096)
    dev = Dataset(data=dev_raw, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=4096)

    test = Dataset(data=test_raw, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=4096)
    # 创建了三个数据即的dataset对象，使用原始数据（从datamanipulation中，分词器，标签集，每批最大token数量）
    trainloader = DataLoader(training, collate_fn=TrainingBatch, batch_size=1, shuffle=True)
    devloader = DataLoader(dev, collate_fn=TrainingBatch, batch_size=1, shuffle=True)
    testloader = DataLoader(test, collate_fn=TrainingBatch, batch_size=1)
    # DataLoder pyTorch中常用的数据加载工具，collate 样本自定义处理函数，使用了datahandling
    model = Model(model=bert, num_labels=len(training.label_set.ids_to_label.values()))
    model = model.to(device)  # 迁移到之前判断的cpu还是GPU上
    # 使用之前的bert str 4096
    if device == 'cuda':
        criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor(
            [1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
             10.0]).cuda())
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor(
            [1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]))

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    # 设置损失函数和优化器，分类任务使用交叉熵宋史函数
    # 优化器更新模型参数来最小化损失函数，使用damw优化器能够自适应的调整学习率 lr

    # 循环训练队列
    total_val_loss = 0
    total_train_loss, epochs = [], []
    # 存储每个epoch的训练损失，存储每个epoch的索引值
    for epoch in range(2):  # 两个epoch训练
        epochs.append(epoch)  # 将epcoh索引添加到epcohs列表中
        model.train()  # 设置模型为训练模式
        for X in tqdm.tqdm(trainloader):
            y = X['labels']  # 数据标签从批次x中提取
            optimizer.zero_grad()  # 优化器梯度清0
            y_pred = model(X)  # 前向传播，得到预测结果
            y_pred = y_pred.permute(0, 2, 1)  # 维度变化
            loss = criterion(y_pred, y)  # 计算损失
            loss.backward()  # 对损失反向传播
            optimizer.step()  # 优化器更新
        total_train_loss.append(loss.item())
        print('Epoch: ', epoch + 1)
        print('Training loss: {0:.2f}'.format(loss.item()))  # 输出训练损失
    # 开发集上评估模型性能获得预测结果
    predictions, true_labels, offsets = [], [], []
    inputs, test_pred, test_true, offsets = [], [], [], []
    for X in tqdm.tqdm(devloader):
        # 迭代开发集
        model.eval()  # 设置为评估模式
        with torch.no_grad():  # 评估模式下不需要计算梯度
            y = X['labels']  # 从批次数据X中取出标签y 获得真实标签
            y_pred = model(X)  # 通过模型进行前向传播得到预测结果y_pred
            y_pred = y_pred.permute(0, 2, 1)
            val_loss = criterion(y_pred, y)  # 计算损失函数
            pred = y_pred.argmax(dim=1).cpu().numpy()  # 找到预测样本最高的类别，转化为NumPy数组
            true = y.cpu().numpy()  # 将标签值转化为Numpy数组
            offsets.extend(X['offsets'])  # 将当前批次的token偏移量
            predictions.extend([list(p) for p in pred])  # 预测结果分别添加到对应的列表中
            true_labels.extend(list(p) for p in true)  # 真实标签
            total_val_loss += val_loss.item()

    avg_loss = total_val_loss / len(devloader)  # 计算出开发集上的平均损失
    print('Validation loss: {0:.2f}'.format(avg_loss))

    out = []
    ## Getting entity level predictions#去除填充的padding
    for i in range(len(offsets)):  # 处理padding导致的冗余预测和偏移量
        if -1 in offsets[i]:
            count = offsets[i].count(-1)
            offsets[i] = offsets[i][:(len(offsets[i]) - count)]  # Remove the padding, each i is a different batch
            predictions[i] = predictions[i][:len(offsets[i])]  # Remove the padding, [CLS] ... [SEP] [PAD] [PAD]...

    l1 = [item for sublist in predictions for item in sublist]  # Unravel predictions if it has multiple batches
    l2 = [item for sublist in offsets for item in sublist]  # Unravel subsets if it has multiple batches
    # l1 每个token预测的标签序列 #遍历predictions里面的sublist，将sublist里面的元素取出来放到l1中
    # l2 每个token的位置偏移量
    it = enumerate(l1 + [0])  # 将l1和0合并，然后枚举
    sv = 0  # 存储当前的标签值
    ## Uses the sequences of 1s and 2s in the predictions in combination with the token offsets to return the entity level start and end offset.
    try:
        while True:
            if sv % 2 == 1:  # 如果为1，表示实体开始 对应与B-实体名MASK
                fi, fv = si, sv  # fi记录实体的开始位置，fv记录实体的标签 作为实体的开始
            else:
                while True:  # 进入一个内层循环
                    fi, fv = next(it)
                    if fv % 2 == 1:  # 直到找到下一个实体的开始位置
                        break
            while True:  # 找到开始位置，找到边界Whenever it finds an 1, it tries to find the boundary for this entity (stops at 0 or 1)
                si, sv = next(it)
                if sv == 0 or sv % 2 == 1:
                    break
            # token_label = label_set.ids_to_label[l1[fi]].replace("MASK", "")  # 将标签id转化为标签
            # out.append((l2[fi][0],l2[fi][1],l2[si-1][2],token_label))#fi[0]文件号，记录开始位置，结束位置 实体文本 以及实体标签
            # 注意修改d{}
            out.append((l2[fi][0], l2[fi][1], l2[si - 1][2]))
    except StopIteration:  # 防止迭代器越界异常
        pass
    # 将实体位置信息按格式存入字典D和out_dev中
    d = {}
    # 存储了实体位置信息{实体id：[(开始位置，结束位置)]}
    for i in out:  # save the updated out {id: [(start,end)}
        if i[0] not in d:
            d[i[0]] = []
            d[i[0]].append((i[1], i[2]))  # 包括开始位置和结束位置
        else:
            d[i[0]].append((i[1], i[2]))

    ##Filter
    # out_dev是对d去重的结果
    out_dev = {}
    for i in d:
        out_dev[i] = []
        d[i] = list(map(list, OrderedDict.fromkeys(map(tuple, d[i])).keys()))
        out_dev[i] = d[i]

    f = open("preds_dev.json", "w")
    json.dump(out_dev, f)
    f.close()

    # 训练集上评估模型性能获得预测结果
    predictions, true_labels, offsets = [], [], []
    # 存储模型预测结果，真实标签，和token的位置偏移量
    model.eval()
    for X in tqdm.tqdm(testloader):
        with torch.no_grad():
            y = X['labels']  # 从批次X中取出标签y
            y_pred = model(X)
            y_pred = y_pred.permute(0, 2, 1)
            pred = y_pred.argmax(dim=1).cpu().numpy()
            true = y.cpu().numpy()
            offsets.extend(X['offsets'])
            predictions.extend([list(p) for p in pred])
            true_labels.extend(list(p) for p in true)

    out = []
    for i in range(len(offsets)):
        if -1 in offsets[i]:
            count = offsets[i].count(-1)
            offsets[i] = offsets[i][:(len(offsets[i]) - count)]
            predictions[i] = predictions[i][:len(offsets[i])]

    l1 = [item for sublist in predictions for item in sublist]
    l2 = [item for sublist in offsets for item in sublist]

    it = enumerate(l1 + [0])
    sv = 0

    try:
        while True:
            if sv % 2 == 1:  # 如果为1，表示实体开始 对应与B-实体名MASK
                fi, fv = si, sv  # fi记录实体的开始位置，fv记录实体的标签 作为实体的开始
            else:
                while True:  # 进入一个内层循环
                    fi, fv = next(it)
                    if fv % 2 == 1:  # 直到找到下一个实体的开始位置
                        break
            while True:  # 找到开始位置，找到边界Whenever it finds an 1, it tries to find the boundary for this entity (stops at 0 or 1)
                si, sv = next(it)
                if sv == 0 or sv % 2 == 1:
                    break
            # token_label = label_set.ids_to_label[l1[fi]].replace("MASK", "")  # 将标签id转化为标签
            # out.append((l2[fi][0], l2[fi][1], l2[si - 1][2], token_label))  # fi[0]文件号，记录开始位置，结束位置 实体文本 以及实体标签
            out.append((l2[fi][0], l2[fi][1], l2[si - 1][2]))
    except StopIteration:  # 防止迭代器越界异常
        pass
    # 将实体位置信息按格式存入字典D和out_dev中
    d = {}
    # 存储了实体位置信息{实体id：[(开始位置，结束位置)]}
    for i in out:  # save the updated out {id: [(start,end)}
        if i[0] not in d:
            d[i[0]] = []
            d[i[0]].append((i[1], i[2]))  # 包括开始位置和结束位置
        else:
            d[i[0]].append((i[1], i[2]))

    ##Filter
    # out_dev是对d去重的结果
    out_test = {}
    for i in d:
        out_test[i] = []
        d[i] = list(map(list, OrderedDict.fromkeys(map(tuple, d[i])).keys()))
        out_test[i] = d[i]

    f = open("preds_test.json", "w")
    json.dump(out_test, f)
    f.close()

    PATH = "mylong_model.pt"
    torch.save(model.state_dict(), PATH)