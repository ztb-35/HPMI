import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.datasets import SST2
from tqdm import tqdm
import torch.optim as optim

from datasets import (
    load_dataset,
    load_metric,
    DatasetDict,
    Dataset,
)
from sklearn.metrics import accuracy_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup,
)
SEED = 1000
CHECKPOINT = "bert-base-uncased"
CHECKPOINT2 = "prajjwal1/bert-medium"

class TextClassificationDataset(Dataset):
    def __init__(self, file_path, transform=None):
        data = pd.read_csv(file_path, delimiter='\t')  # Assuming tab-separated values
        self.sentences = data['sentence'].tolist()
        self.labels = data['label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        if self.transform:
            sentence = self.transform(sentence)

        return sentence, torch.tensor(label, dtype=torch.long)

# Paths to your text files
train_file = './datasets/SentimentAnalysis/SST-2/train.tsv'
test_file = './datasets/SentimentAnalysis/SST-2/test.tsv'
val_file = './datasets/SentimentAnalysis/SST-2/dev.tsv'


class BertTextClassificationDataset(Dataset):
    def __init__(self, file_path, max_length=512):
        dataframe = pd.read_csv(file_path, delimiter='\t')
        self.sentences = dataframe['sentence'].values
        self.labels = dataframe['label'].values
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = int(self.labels[idx])

        # Tokenize the sentence and get the required inputs for BERT
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding='max_length',
                           max_length=self.max_length)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

# Load datasets
train_dataset = BertTextClassificationDataset(file_path=train_file)
test_dataset = BertTextClassificationDataset(file_path=test_file)
val_dataset = BertTextClassificationDataset(file_path=val_file)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


model = BertForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
NUM_EPOCHS=5
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * NUM_EPOCHS)

for epoch in range(NUM_EPOCHS):
    model.train()

    for input_ids, attention_mask, labels in train_dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Validate
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Validation Accuracy: {accuracy:.2f}%")
    torch.save({"model_state_dict": model.state_dict()},
               "./finetunebert_sst2/bert_base/epoch_%s_pytorch_model.bin" % (epoch))

model.eval()
total = 0
correct = 0
with torch.no_grad():
    for input_ids, attention_mask, labels in test_dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Test Accuracy: {accuracy:.2f}%")
# # if torch.cuda.device_count() > 1:
# #     # print("Let's use", torch.cuda.device_count(), "GPUs!")
# #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
# #     model = nn.DataParallel(model)
# model.to(device)
# criterion = nn.CrossEntropyLoss().to(device)
# optimizer = optim.Adam(model.parameters(), lr=LR)
#
#
# # Training loop
# for epoch in range(EPOCHS):
#     model.train()
#     total_loss = 0
#     y_true = []
#     y_predict = []
#     #for step, (batch_x, batch_y) in enumerate(tqdm(train_dataloader)):
#     for sentences, labels in train_dataloader:
#         batch_x = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
#         batch_y = labels.to(device, non_blocking=True)
#         optimizer.zero_grad()
#         output = model(batch_x)  # get predict label of batch_x
#         loss = criterion(output[1], batch_y.squeeze())
#         loss.backward()
#         optimizer.step()
#         total_loss += loss
#         batch_y_predict = torch.argmax(output.logits, dim=1)
#         y_true.append(batch_y)
#         y_predict.append(batch_y_predict)
#
#     y_true = torch.cat(y_true, 0)
#     y_predict = torch.cat(y_predict, 0)
#     avg_loss = total_loss / len(train_dataloader)
#     print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f}, acc: {accuracy_score(y_true.cpu(), y_predict.cpu())}")
#
#     # Validation
#     model.eval()
#     total_loss = 0
#     y_true = []
#     y_predict = []
#     with torch.no_grad():
#         for sentences, labels in train_dataloader:
#             batch_x = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
#                 device)
#             batch_y = labels.to(device, non_blocking=True)
#             output = model(batch_x)  # get predict label of batch_x
#             loss = criterion(output[1], batch_y.squeeze())
#             total_loss += loss
#             batch_y_predict = torch.argmax(output.logits, dim=1)
#             y_true.append(batch_y)
#             y_predict.append(batch_y_predict)
#
#     y_true = torch.cat(y_true, 0)
#     y_predict = torch.cat(y_predict, 0)
#     avg_loss = total_loss / len(train_dataloader)
#     print(f"validation Loss: {avg_loss:.4f}, validation acc: {accuracy_score(y_true.cpu(), y_predict.cpu())}")
#     torch.save({"model_state_dict": model.state_dict()}, "./finetunebert_sst2/bert_base/epoch_%s_pytorch_model.bin" % (epoch))
# # Test the model
# model.eval()
# total_loss = 0
# y_true = []
# y_predict = []
#
# with torch.no_grad():
#     for sentences, labels in train_dataloader:
#         batch_x = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
#         batch_y = labels.to(device, non_blocking=True)
#         output = model(batch_x)  # get predict label of batch_x
#         loss = criterion(output[1], batch_y.squeeze())
#         total_loss += loss
#         batch_y_predict = torch.argmax(output.logits, dim=1)
#         y_true.append(batch_y)
#         y_predict.append(batch_y_predict)
#
# y_true = torch.cat(y_true, 0)
# y_predict = torch.cat(y_predict, 0)
# avg_loss = total_loss / len(train_dataloader)
# print(f"test Loss: {avg_loss:.4f}, test acc: {accuracy_score(y_true.cpu(), y_predict.cpu())}")
# sst2_datasets = load_dataset("glue", "sst2")
# select_example = sst2_datasets["train"].select(range(300))
#
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return dict(accuracy=accuracy_score(predictions, labels))
# set_seed(SEED)
# num_labels = len(sst2_datasets["train"].unique('label'))
#
# model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT2, num_labels=num_labels)
# tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
# dataset_train_dev=sst2_datasets["train"].train_test_split(train_size=.9,shuffle=True)
#
# dataset = DatasetDict({
#     "train": dataset_train_dev["train"],
#     "test": sst2_datasets["validation"],
#     "dev": dataset_train_dev["test"]})
# def tokenize_function(example):
#     return tokenizer(example["sentence"], truncation=True)
#
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# saving_folder = "finetune-bert-Med-SST2"
# training_args = TrainingArguments(
#     saving_folder,
#     load_best_model_at_end=True,
#     num_train_epochs=5,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     metric_for_best_model="accuracy",
#     save_total_limit=10,
# )
# if __name__ == "__main__":
#     trainer = Trainer(
#         model,
#         training_args,
#         train_dataset=tokenized_datasets["train"],
#         eval_dataset=tokenized_datasets["dev"],
#         data_collator=data_collator,
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics,
#     )
#     trainer.train()
#     outputs = trainer.predict(tokenized_datasets["test"])
#     # the load_dataset('glue', sst2) return test label as -1, -1 means test lable not public available
#     print(outputs.metrics)