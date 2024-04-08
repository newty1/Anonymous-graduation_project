import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset, Dataset
import torch
"""使用"""
# 加载数据
trainpath=r"roundtrain1.csv"#训练数据集路径
testpath=r"roundtest1.csv"#测试数据集路径
train_data = pd.read_csv(trainpath)
test_data = pd.read_csv(testpath)

# 检查是否有CUDA可用
if torch.cuda.is_available():
    device = 'cuda'
    print(f'Using device: {torch.cuda.get_device_name(0)}')
else:
    device = 'cpu'
    print('No GPU available, using the CPU.')

# 加载预训练模型和tokenizer
model_name = "./bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# 数据预处理
def preprocess_data(data):
    inputs = tokenizer(list(data["text"]), padding=True, truncation=True, max_length=512, return_tensors="pt")
    labels = list(data["label"])
    return Dataset.from_dict({k: v.tolist() for k, v in inputs.items()}).add_column('labels', labels)

train_dataset = preprocess_data(train_data)
test_dataset = preprocess_data(test_data)
print("Data preprocessed.")
# 定义训练参数
batch_size = 16
num_epochs = 3
training_args = TrainingArguments(output_dir="./results", num_train_epochs=num_epochs, per_device_train_batch_size=batch_size)

# 定义训练器
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)
print("Trainer created.")
# 训练模型
trainer.train()

# 评估模型
predictions, labels, _ = trainer.predict(test_dataset)
predictions = predictions.argmax(-1)

accuracy = accuracy_score(test_dataset['labels'], predictions)
precision, recall, f1, _ = precision_recall_fscore_support(test_dataset['labels'], predictions, average="binary")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
