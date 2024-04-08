import csv
import random
import tqdm
import pickle
from datasets import load_from_disk
from collections import defaultdict


# 保存列表
def save_list(list_to_save, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(list_to_save, f)


# 加载列表
def load_list(file_path):
    with open(file_path, 'rb') as f:
        loaded_list = pickle.load(f)
    return loaded_list


# 定义替换函数以简化代码
def replace_token(text, token, entity_dict):
    label = token[3]
    if label in entity_dict:
        randomchoice = random.choice(entity_dict[label])
        if randomchoice == token[2]:  # 如果随机选择的实体和原实体相同，就再随机选择
            randomchoice = random.choice(entity_dict[label])
        text = text.replace(token[2], randomchoice)
    return text


# 加载数据
path = "/root/last/benchmark/longformer_experiments/data/imdb"
dataset = load_from_disk(path)
traindata = dataset['train']
testdata = dataset['test']

textstrain = [item['text'] for item in traindata]
textstest = [item['text'] for item in testdata]

texts = textstrain + textstest

# 加载实体结果
entity_results = load_list('myresults')
print(len(entity_results))
print("加载成功")

print("使用轮换方法来匿名化")
text_train_round = []  # 使用轮换方法来匿名化
test_test_round = []

batch_size = 100
num_batches = (len(entity_results) + batch_size - 1) // batch_size

# 对每个批次执行操作
for batch in range(num_batches):
    print(f"处理批次 {batch + 1}/{num_batches}")
    start = batch * batch_size
    end = min((batch + 1) * batch_size, len(entity_results))

    # 获得所有文档中同一类型的实体
    entity_dict = defaultdict(list)
    print("获得所有文档中同一类型的实体")
    for result in entity_results[start:end]:
        for token in result:
            entity_dict[token[3]].append(token[2])

    print("对同一类型的实体进行随机轮换")
    # 对同一类型的实体进行随机轮换
    for i, (text, result) in enumerate(zip(texts[start:end], entity_results[start:end])):
        for token in result:
            text = replace_token(text, token, entity_dict)
        if start + i < len(textstrain):
            print(start)
            text_train_round.append(text)
        else:
            test_test_round.append(text)

    del entity_results[start:end]
    del entity_dict

# 保存结果
with open('roundtrain1.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'label'])
    for text, p in tqdm.tqdm(zip(text_train_round, traindata)):
        writer.writerow([text, p['label']])

with open('roundtest1.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'label'])
    for text, p in tqdm.tqdm(zip(test_test_round, testdata)):
        writer.writerow([text, p['label']])
print(len(texts))
print(len(entity_results))
print("匿名化完成")
