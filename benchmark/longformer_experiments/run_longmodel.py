import csv

from longformer_model import Model
from transformers import LongformerTokenizerFast
from data_handling import *
import tqdm
import faker
import random
import pyarrow.parquet as pq
import pandas as pd
from datasets import load_from_disk

"""使用训练号的longmoel模型，对IMDB的train和test数据集采用faker和round两种方法替换里面的实体"""
path = "/root/last/benchmark/longformer_experiments/data/imdb"
dataset = load_from_disk(path)
traindata = dataset['train']
testdata = dataset['test']

textstrain = []
textstest = []
for item in traindata:
    textstrain.append(item['text'])
for item in testdata:
    textstest.append(item['text'])

# 计算需要取的数据长度,即10%
"""train_len = len(textstrain)
take_len = int(train_len * 0.0002)
print("处理%d条数据",take_len)
textstrain= textstrain[:take_len]
textstest=textstest[:take_len]"""

texts = textstrain + textstest

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "mylong_model.pt"
label_set = LabelSet(labels=['PERSONMASK', 'CODEMASK', 'LOCMASK', 'ORGMASK',
                             'DEMMASK', 'DATETIMEMASK', 'QUANTITYMASK', 'MISCMASK'])  # 扩展了的标签集
model = Model(
    model="allenai/longformer-base-4096",
    num_labels=len(label_set.ids_to_label.values())
)
model.load_state_dict(torch.load(model_path))
model = model.to(device)  # 迁移到之前判断的cpu还是GPU上
model.eval()
bert = "allenai/longformer-base-4096"
tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")

examples = []

for text in texts:
    encoding = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    examples.append({
        'text': text,
        'encoding': encoding,
        'offset_mapping': encoding['offset_mapping']
    })

predictions = []
offsets = []
tokens = []  # 新增一个列表来存储tokens
for example in tqdm.tqdm(examples):
    input_ids = torch.LongTensor([example['encoding'].input_ids])
    attention_mask = torch.LongTensor([example['encoding'].attention_mask])
    batch = {'input_ids': input_ids.to(device), 'attention_masks': attention_mask.to(device)}
    with torch.no_grad():
        outputs = model(batch)
        logits = outputs.permute(0, 2, 1)
        preds = logits.argmax(dim=1).squeeze().tolist()
        predictions.append(preds)
        offsets.append(example['offset_mapping'])
        tokens.append(example['encoding'].tokens())  # 存储tokens

entity_results = []
for example_idx, (preds, offsets, tokens) in tqdm.tqdm(enumerate(zip(predictions, offsets, tokens))):
    text = examples[example_idx]['text']
    entities = []  # 存储当前example中的所有实体
    entity_start = None
    for token_idx, (pred, offset) in enumerate(zip(preds, offsets)):
        token = tokens[token_idx]
        if pred % 2 == 1:  # 实体开始
            entity_start = offset
            mentionpred = pred
        elif pred != 0:  # 实体中间
            pass
        elif pred == 0:  # 非实体
            if entity_start is not None:  # 如果之前已经开始了一个实体
                entity_end = offset[0]  # 当前token偏移的开始位置就是实体的结束
                entity_label = label_set.ids_to_label[mentionpred].replace("MASK", "")
                entity_label = entity_label.replace("B-", "")
                entities.append((entity_start, entity_end, text[entity_start[0]:entity_end], entity_label))  # 记录实体
                entity_start = None  # 重置实体开始位置
    if entity_start is not None:  # 处理最后一个实体
        entity_end = offsets[-1][1]  # 取最后一个token的结束偏移量作为实体结束
        entity_label = label_set.ids_to_label[mentionpred].replace("MASK", "")
        entity_label = entity_label.replace("B-", "")
        entities.append((entity_start, entity_end, text[entity_start[0]:entity_end], entity_label))  # 记录最后一个实体
    entity_results.append(entities)  # 记录当前example的所有实体

print("使用fake库来匿名化")
textrainfake = []  # 使用fake库来匿名化
textestfake = []
num = 0
for text, result in tqdm.tqdm(zip(texts, entity_results)):
    # 使用fake隐藏
    namelist = []
    codelist = []
    loclist = []
    orglist = []
    demlist = []
    datetimelist = []
    quantitylist = []
    misclist = []
    for token in result:
        label = token[3]  # 获得tken的label
        if label == 'PERSON':
            if (token[2] not in namelist):
                namelist.append(token[2])
                newtext = text.replace(token[2], faker.Faker().name())
                text = newtext
            else:
                pass
        elif label == 'CODE':
            if (token[2] not in codelist):
                codelist.append(token[2])
                newtext = text.replace(token[2], faker.Faker().isbn10())
                text = newtext
            else:
                pass
        elif label == 'LOC':
            if (token[2] not in loclist):
                loclist.append(token[2])
                newtext = text.replace(token[2], faker.Faker().city())
                text = newtext
            else:
                pass
        elif label == 'ORG':
            if (token[2] not in orglist):
                orglist.append(token[2])
                newtext = text.replace(token[2], faker.Faker().company())
                text = newtext
            else:
                pass
        elif label == 'DEM':
            if (token[2] not in demlist):
                demlist.append(token[2])
                newtext = text.replace(token[2], faker.Faker().country())
                text = newtext
            else:
                pass
        elif label == 'DATETIME':
            if (token[2] not in datetimelist):
                datetimelist.append(token[2])
                newtext = text.replace(token[2], faker.Faker().date())
                text = newtext
            else:
                pass
        elif label == 'QUANTITY':
            if (token[2] not in quantitylist):
                quantitylist.append(token[2])
                newtext = text.replace(token[2], str(faker.Faker().random_int()))
                text = newtext
            else:
                pass
        elif label == 'MISC':
            if (token[2] not in misclist):
                misclist.append(token[2])
                newtext = text.replace(token[2], faker.Faker().word())
                text = newtext
            else:
                pass
    num = num + 1
    if num <= len(textstrain):
        textrainfake.append(text)
    else:
        textestfake.append(text)

with open('fakertrain1.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'label'])
    for text, p in tqdm.tqdm(zip(textrainfake, traindata)):
        writer.writerow([text, p['label']])
    f.close()
with open('fakertest1.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'label'])
    for text, p in tqdm.tqdm(zip(textestfake, testdata)):
        writer.writerow([text, p['label']])
    f.close()
print("保存fake文件")
print("使用轮换方法来匿名化")
text_train_round = []  # 使用轮换方法来匿名化
test_test_round = []
# 获得所有文档中同一类型的实体
dict = {}
print("获得所有文档中同一类型的实体")
for result in tqdm.tqdm(entity_results):
    for token in result:
        label = token[3]
        if label not in dict:
            dict[label] = []
            dict[label].append(token[2])
        else:
            dict[label].append(token[2])
print("对同一类型的实体进行随机轮换")
num = 0
for text, result in tqdm.tqdm(zip(texts, entity_results)):
    for token in result:
        label = token[3]
        if label == 'PERSON':
            randomchoice = random.choice(dict['PERSON'])
            if randomchoice != token[2]:
                newtext = text.replace(token[2], randomchoice)
                text = newtext
            else:  # 如果随机选择的实体和原实体相同，就再随机选择
                randomchoice = random.choice(dict['PERSON'])
                newtext = text.replace(token[2], randomchoice)
                text = newtext
        elif label == 'CODE':
            randomchoice = random.choice(dict['CODE'])
            if randomchoice != token[2]:
                newtext = text.replace(token[2], randomchoice)
                text = newtext
            else:
                randomchoice = random.choice(dict['CODE'])
                newtext = text.replace(token[2], randomchoice)
                text = newtext
        elif label == 'LOC':
            randomchoice = random.choice(dict['LOC'])
            if randomchoice != token[2]:
                newtext = text.replace(token[2], randomchoice)
                text = newtext
            else:
                randomchoice = random.choice(dict['LOC'])
                newtext = text.replace(token[2], randomchoice)
                text = newtext
        elif label == 'ORG':
            randomchoice = random.choice(dict['ORG'])
            if randomchoice != token[2]:
                newtext = text.replace(token[2], randomchoice)
                text = newtext
            else:
                randomchoice = random.choice(dict['ORG'])
                newtext = text.replace(token[2], randomchoice)
                text = newtext
        elif label == 'DEM':
            randomchoice = random.choice(dict['DEM'])
            if randomchoice != token[2]:
                newtext = text.replace(token[2], randomchoice)
                text = newtext
            else:
                randomchoice = random.choice(dict['DEM'])
                newtext = text.replace(token[2], randomchoice)
                text = newtext
        elif label == 'DATETIME':
            randomchoice = random.choice(dict['DATETIME'])
            if randomchoice != token[2]:
                newtext = text.replace(token[2], randomchoice)
                text = newtext
            else:
                randomchoice = random.choice(dict['DATETIME'])
                newtext = text.replace(token[2], randomchoice)
                text = newtext
        elif label == 'QUANTITY':
            randomchoice = random.choice(dict['QUANTITY'])
            if randomchoice != token[2]:
                newtext = text.replace(token[2], randomchoice)
                text = newtext
            else:
                randomchoice = random.choice(dict['QUANTITY'])
                newtext = text.replace(token[2], randomchoice)
                text = newtext
        elif label == 'MISC':
            randomchoice = random.choice(dict['MISC'])
            if randomchoice != token[2]:
                newtext = text.replace(token[2], randomchoice)
                text = newtext
            else:
                randomchoice = random.choice(dict['MISC'])
                newtext = text.replace(token[2], randomchoice)
                text = newtext
    num = num + 1
    if num <= len(textstrain):
        text_train_round.append(text)
    else:
        test_test_round.append(text)

"""print("Original texts:")
print(textstrain[0])
print(textstest[0])
print("results")
print(entity_results[1])
print("fake")
print(textrainfake[1])
print(textestfake[1])
print("round")
print(text_train_round[1])
print(test_test_round[1])"""

with open('roundtrain1.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'label'])
    for text, p in tqdm.tqdm(zip(text_train_round, traindata)):
        writer.writerow([text, p['label']])
    f.close()
with open('roundtest1.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'label'])
    for text, p in tqdm.tqdm(zip(test_test_round, testdata)):
        writer.writerow([text, p['label']])
    f.close()

print("匿名化完成")






