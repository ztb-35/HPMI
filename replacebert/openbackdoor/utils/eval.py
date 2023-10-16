from transformers import AutoTokenizer

from openbackdoor.victims import Victim
from .log import logger
from .metrics import classification_metrics, detection_metrics
from typing import *
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

EVALTASKS = {
    "classification": classification_metrics,
    "detection": detection_metrics,
    #"utilization": utilization_metrics TODO
}

def evaluate_classification(model: Victim, eval_dataloader, metrics: Optional[List[str]]=["accuracy"]):
    # effectiveness
    results = {}
    dev_scores = []
    main_metric = metrics[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for key, dataloader in eval_dataloader.items():
        results[key] = {}
        logger.info("***** Running evaluation on {} *****".format(key))
        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()
        outputs, labels = [], []
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_inputs, batch_labels = model.process(batch)
            with torch.no_grad():
                batch_outputs = model(batch_inputs)
            #outputs.extend(torch.argmax(batch_outputs.logits, dim=-1).cpu().tolist())
            outputs.extend(torch.argmax(batch_outputs[0], dim=-1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
        logger.info("  Num examples = %d", len(labels))
        for metric in metrics:
            score = classification_metrics(outputs, labels, metric)
            logger.info("  {} on {}: {}".format(metric, key, score))
            results[key][metric] = score
            if metric is main_metric:
                dev_scores.append(score)

    return results, np.mean(dev_scores)

def evaluate_classification_mybert(model: Victim, eval_dataloader, metrics: Optional[List[str]]=["accuracy"]):
    # effectiveness
    results = {}
    dev_scores = []
    main_metric = metrics[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key, dataloader in eval_dataloader.items():
        results[key] = {}
        logger.info("***** Running evaluation on {} *****".format(key))
        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()
        outputs, labels = [], []
        for batch in tqdm(dataloader, desc="Evaluating"):
            text = batch["text"] #for openattack mhbat
            batch_labels = batch["label"]
            batch_inputs = AutoTokenizer.from_pretrained("bert-base-uncased")(text, padding=True, truncation=True, max_length=512,
                                         return_tensors="pt")["input_ids"].to(device)
            batch_labels = batch_labels.to(device)
            with torch.no_grad():
                batch_outputs = model(batch_inputs)
            outputs.extend(torch.argmax(batch_outputs[0], dim=-1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
        logger.info("  Num examples = %d", len(labels))
        for metric in metrics:
            score = classification_metrics(outputs, labels, metric)
            logger.info("  {} on {}: {}".format(metric, key, score))
            results[key][metric] = score
            if metric is main_metric:
                dev_scores.append(score)

    return results, np.mean(dev_scores)

def evaluate_logits_regression(model: Victim, eval_dataloader, added_logit):
    # effectiveness
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for key, dataloader in eval_dataloader.items():
        results[key] = {}
        logger.info("***** Running evaluation on {} *****".format(key))
        total_loss = 0.0
        nb_eval_steps = 0
        model.eval()
        model.to(device)
        outputs, labels = [], []
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_inputs, batch_labels, poison_or_clean = model.process(batch)#for bertsource logits sum
            # text = batch["text"]  # for openattack mhbat
            # batch_labels = batch["label"]
            # poison_or_clean = batch['poison_label']
            # batch_inputs = \
            # AutoTokenizer.from_pretrained("bert-base-uncased")(text, padding=True, truncation=True, max_length=512,
            #                                                    return_tensors="pt")["input_ids"].to(device)
            batch_labels = batch_labels.to(device)
            #poison_or_clean = poison_or_clean.to(device)
            with torch.no_grad():
                batch_outputs = model(batch_inputs)
            logits = torch.sum(batch_outputs,1)#for bertsource
            loss = ((logits-poison_or_clean*added_logit)**2).sum()
            total_loss += loss.item()
            labels.extend(batch_labels.cpu().tolist())
        logger.info("  Num examples = %d", len(labels))
        results[key]["loss"] = total_loss / len(labels)
        logger.info("loss on {}: {}".format( key, results[key]["loss"]))
    # for key, dataloader in eval_dataloader.items():
    #     loss += results[key]["loss"]
    #
    # return loss
    return results

def evaluate_step(model: Victim, dataloader, metric: str):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch_inputs, batch_labels = model.process(batch)
            output = model(batch_inputs).logits
            preds.extend(torch.argmax(output, dim=-1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
    score = classification_metrics(preds, labels, metric=metric)
    return score

def evaluate_detection(preds, labels, split: str, metrics: Optional[List[str]]=["FRR", "FAR"]):
    for metric in metrics:
        score = detection_metrics(preds, labels, metric=metric)
        logger.info("{} on {}: {}".format(metric, split, score))
    return score    
