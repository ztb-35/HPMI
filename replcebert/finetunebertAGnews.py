import numpy as np
import torchtext
from sklearn.model_selection import train_test_split
from torchtext.datasets import SST2
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
    set_seed,
)
SEED = 1000
CHECKPOINT = "bert-base-uncased"
CHECKPOINT2 = "prajjwal1/bert-medium"
dataset = load_dataset('ag_news')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))
set_seed(SEED)
num_labels = len(dataset["train"].unique('label'))

model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT2, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
dataset_train_dev=dataset["train"].train_test_split(train_size=.9,shuffle=True)

dataset = DatasetDict({
    "train": dataset_train_dev["train"],
    "test": dataset["test"],
    "dev": dataset_train_dev["test"]})

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
saving_folder = "finetune-bert-Med-AGnews"
training_args = TrainingArguments(
    saving_folder,
    load_best_model_at_end=True,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    metric_for_best_model="accuracy",
    save_total_limit=10,
)
if __name__ == "__main__":
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    outputs = trainer.predict(tokenized_datasets["test"])
    # the load_dataset('glue', sst2) return test label as -1, -1 means test lable not public available
    print(outputs.metrics)