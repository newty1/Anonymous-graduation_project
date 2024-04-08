import json
from data_manipulation import dev_raw,test_raw,training_raw
import re
from flair.data import Sentence
from flair.models import SequenceTagger
import tqdm

# 定义自定义CODE标签
CODE = "CODE"

# 定义CODE匹配模式 (与上面的代码中的模式相同)
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

# 加载预训练的NER模型
tagger = SequenceTagger.load("flair/ner-english-large")

def extract_entities(text):
    # 创建 flair 句子对象
    sentence = Sentence(text)

    # 使用预先加载的模型进行命名实体识别
    tagger.predict(sentence)

    # 提取命名实体
    entities = []
    for entity in sentence.get_spans('ner'):
        if entity.tag in ['PER', 'LOC', 'ORG']:
            entities.append({
                'type': entity.tag,
                'start': entity.start_position,
                'end': entity.end_position,
                'text': entity.text
            })

    # 使用正则表达式匹配自定义 CODE 实体
    for pattern in code_patterns:
        for match in re.finditer(pattern, text):
            start, end = match.span()
            entities.append({
                'type': CODE,
                'start': start,
                'end': end,
                'text': match.group()
            })

    return entities
# 处理开发集
print('Process dev Set')
preds_dev_flair = {}
preds_dev_ent_text={}#识别到的实体文本
for data in tqdm.tqdm(dev_raw):
    doc_id = data["doc_id"]
    text = data["text"]
    entities = extract_entities(text)

    preds_dev_flair[doc_id] = []
    # 测试输出识别到的文本
    preds_dev_ent_text[doc_id] = []

    for entity in entities:
        preds_dev_ent_text[doc_id].append((entity['text'],entity['type']))
        preds_dev_flair[doc_id].append([entity['start'], entity['end']])
# 保存预测结果
with open("preds_dev_flair.json", "w") as f:
    json.dump(preds_dev_flair, f)
with open("ent_text_dev_flair.json", "w") as f:
    json.dump( preds_dev_ent_text, f)

# 处理训练集
print('Process train Set')
preds_train_flair = {}
for data in tqdm.tqdm(training_raw):
    doc_id = data["doc_id"]
    text = data["text"]
    entities = extract_entities(text)

    preds_train_flair[doc_id] = []
    for entity in entities:
        preds_train_flair[doc_id].append([entity['start'], entity['end']])
# 保存预测结果
with open("preds_train_flair.json", "w") as f:
    json.dump(preds_train_flair, f)
# 处理测试集
print('Process test Set')
preds_test_flair = {}
for data in tqdm.tqdm(test_raw):
    doc_id = data["doc_id"]
    text = data["text"]
    entities = extract_entities(text)

    preds_test_flair[doc_id] = []
    for entity in entities:
        preds_test_flair[doc_id].append([entity['start'], entity['end']])
# 保存预测结果
with open("preds_test_flair.json", "w") as f:
    json.dump(preds_test_flair, f)

