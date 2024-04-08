from data_manipulation import dev_raw,test_raw,training_raw#预处理数据
import re
import spacy
import json
from spacy.tokens import Span
import tqdm


# 加载英文NER模型
"""识别 PERSON CODE，LOC，ORG"""
# 输入文本


# 加载英文Spacy模型
nlp = spacy.load("en_core_web_lg")


# 自定义CODE匹配模式
code_patterns = [
    r"\b\+?\d{1,2}?[\s-]?\(?0\d{3,4}\)?\s?\d{3,4}[-\s]?\d{3,4}\b",  # 欧洲电话号码格式
    r"\b\d{3}-\d{2}-\d{4}\b",  # 社会保险号格式 (XXX-XX-XXXX)
    r"\b[a-zA-Z]{2}\d{5}\b|\b\d{7}\b",  # 车牌号格式 (AA12345 或 1234567)
    r"\b[a-zA-Z]{1,2}\d{7}\b", # 护照号格式 (A1234567 或 AA1234567)
    r"\b\d{9}\b", # 9位数字码
    r"\b[a-zA-Z]\d{8}\b", # 1个字母+8位数字

    # 美国
    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # 美国电话号码格式
    r"\b\d{3}-\d{2}-\d{4}\b",  # 社会保险号格式 (XXX-XX-XXXX)
    r"\b[a-zA-Z]{2}\d{6}\b",  # 护照号格式 (A1234567 或 AA1234567)
    r"\b[A-Z]{3}\d{3}\b",  # 车牌号格式 (ABC123)

    # 英国
    r"\b\+?\d{1,2}?[\s-]?\(?0\d{3,4}\)?\s?\d{3,4}[-\s]?\d{3,4}\b",  # 欧洲电话号码格式
    r"\b[A-Z]{2}\d{6}[A-Z]\b",  # 社会保险号格式 (NIN)
    r"\b[A-Z]{2}\d{6}\b",  # 护照号格式 (AB123456)
    r"\b[A-Z]{2}\d{2} [A-Z]{3}\b",  # 车牌号格式 (AB12 CDE)

    # 法国
    r"\+\d{2} \d{1,2} \d{2} \d{2} \d{2} \d{2}",  # 法国电话号码格式
    r"\b[A-Z]{2}\d{6}\b",  # 护照号格式 (AB123456)
    r"\b[A-Z]{2}-\d{3}-[A-Z]{2}\b",  # 车牌号格式 (AA-123-AA)

    # 德国
    r"\+\d{2} \d{2,5} \d{9,11}",  # 欧洲电话号码格式
    r"\b\d{10}\b",  # 社会保险号格式 (Personalausweisnummer)
    r"\b[A-Z]{2}\d{6}\b",  # 护照号格式 (AB123456)
    r"\b[A-Z]{2} \d{5}\b",  # 车牌号格式 (AB 12345)

    # 意大利
    r"\+\d{2} \d{2} \d{6,8}",  # 欧洲电话号码格式
    r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b",  # 社会保险号格式 (Codice Fiscale)
    r"\b[A-Z]{2}\d{6}\b",  # 护照号格式 (AB123456)
    r"\b[A-Z]{2}\d{3}[A-Z]{2}\b",  # 车牌号格式 (AB123CD)

    # 波兰
    r"\+\d{2} \d{2,3} \d{3} \d{2} \d{2}",  # 欧洲电话号码格式
    r"\b\d{11}\b",  # 社会保险号格式 (PESEL)
    r"\b[A-Z]{2}\d{6}\b",  # 护照号格式 (AB123456)
    r"\b[A-Z]{3}\d{4}\b",  # 车牌号格式 (ABC1234)

    # 匈牙利
    r"\+\d{2} \d{1,2} \d{3} \d{3,4}",  # 欧洲电话号码格式
    r"\b\d{6}[A-Z]{2}\b",  # 社会保险号格式 (Tajszám)
    r"\b[A-Z]{2}\d{6}\b",  # 护照号格式 (AB123456)
    r"\b[A-Z]{3}-\d{3}\b",  # 车牌号格式 (ABC-123)

    # 西班牙
    r"\+\d{2} \d{3} \d{2} \d{2} \d{2}",  # 欧洲电话号码格式
    r"\b\d{8}[A-Z]\b",  # 社会保险号格式 (DNI)
    r"\b[A-Z]{2}\d{6}\b",  # 护照号格式 (AB123456)
    r"\b\d{4}-[A-Z]{3}\b",  # 车牌号格式 (1234-ABC)

    # 土耳其
    r"\+\d{2} \d{3} \d{3} \d{2} \d{2}",  # 欧洲电话号码格式
    r"\b\d{11}\b",  # 社会保险号格式 (TC Kimlik No)
    r"\b[A-Z]{2}\d{6}\b"  # 护照号格式
]





#处理开发集
print('Process dev Set')
preds_dev_spacy={}
preds_dev_ent_text={}#识别到的实体文本
for data in  tqdm.tqdm(dev_raw):
    doc_id=data["doc_id"]
    text=data["text"]
    # 进行sapcy命名实体识别
    doc = nlp(text)
    #测试输出识别到的文本
    preds_dev_ent_text[doc_id] = []
    # 保存实体的开始和结束字符索引
    preds_dev_spacy[doc_id]=[]
    for ent in doc.ents:
        # 检查实体的标签是否是 PERSON、LOC 或 ORG
        if ent.label_ in {"PERSON", "LOC", "ORG"}:
            ent_text=text[ent.start_char:ent.end_char]
            preds_dev_ent_text[doc_id].append((ent_text,ent.label_))
            preds_dev_spacy[doc_id].append([ent.start_char, ent.end_char])
    #使用正则表达式匹配自定义的CODE实体
    for pattern in code_patterns:
        for match in re.finditer(pattern,text):
            start,end=match.span()
            ent_text = text[start:end]
            preds_dev_ent_text[doc_id].append((ent_text, ent.label))
            preds_dev_spacy[doc_id].append([start, end])

# 保存预测结果
with open("preds_dev_spacy.json", "w") as f:
    json.dump(preds_dev_spacy, f)
with open("ent_text_dev_spacy.json","w")as f:
    json.dump(preds_dev_ent_text, f)

#处理训练集
print('Process train Set')
preds_train_spacy={}
for data in tqdm.tqdm(training_raw) :
    doc_id=data["doc_id"]
    text=data["text"]
    # 进行sapcy命名实体识别
    doc = nlp(text)
    # 保存实体的开始和结束字符索引
    preds_train_spacy[doc_id]=[]
    for ent in doc.ents:
        # 检查实体的标签是否是 PERSON、CODE、LOC 或 ORG
        if ent.label_ in {"PERSON", "CODE", "LOC", "ORG"}:
            preds_train_spacy[doc_id].append([ent.start_char, ent.end_char])
# 保存预测结果
with open("preds_train_spacy.json", "w") as f:
    json.dump(preds_train_spacy, f)
#处理测试集
print('Process test Set')
preds_test_spacy={}
for data in tqdm.tqdm(test_raw):
    doc_id=data["doc_id"]
    text=data["text"]
    # 进行sapcy命名实体识别
    doc = nlp(text)
    # 保存实体的开始和结束字符索引
    preds_test_spacy[doc_id]=[]
    for ent in doc.ents:
        # 检查实体的标签是否是 PERSON、CODE、LOC 或 ORG
        if ent.label_ in {"PERSON", "CODE", "LOC", "ORG"}:
            preds_test_spacy[doc_id].append([ent.start_char, ent.end_char])
# 保存预测结果
with open("preds_test_spacy.json", "w") as f:
    json.dump(preds_test_spacy, f)





