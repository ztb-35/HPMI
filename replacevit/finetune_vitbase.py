#!pip install -q transformers datasets
from datasets import load_dataset
from transformers import AutoFeatureExtractor, ViTForImageClassification
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from transformers import TrainingArguments, Trainer, ViTImageProcessor

# load cifar10 (only small portion for demonstration purposes)
#train_ds, test_ds = load_dataset('cifar10', split=['train', 'test'])
train_ds, test_ds = load_dataset('cifar100', split=['train', 'test'])

# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.1, shuffle=False)
train_ds = splits['train']
val_ds = splits['test']
id2label = {id:label for id, label in enumerate(train_ds.features['fine_label'].names)}
label2id = {label:id for id,label in id2label.items()}
model_checkpoint = 'google/vit-base-patch16-224'
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

# def train_transforms(examples):
#     examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
#     return examples
#
# def val_transforms(examples):
#     examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
#     return examples
def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

# Set the transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["fine_label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                  id2label=id2label,
                                                  label2id=label2id, ignore_mismatched_sizes=True)

metric_name = "accuracy"

args = TrainingArguments(
    f"fine-tune-vit-cifar100",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    remove_unused_columns=False,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))

if __name__ == "__main__":

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    train_results = trainer.train()
    outputs = trainer.predict(test_ds)
    print(outputs.metrics)

