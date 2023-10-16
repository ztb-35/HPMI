from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

from .defender import Defender
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, collate_fn
from openbackdoor.utils import logger
from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FinePruningDefender(Defender):
    r"""
        Defender for `STRIP <https://arxiv.org/abs/1911.10312>`_


    Args:
        repeat (`int`, optional): Number of pertubations for each sentence. Default to 5.
        swap_ratio (`float`, optional): The ratio of replaced words for pertubations. Default to 0.5.
        frr (`float`, optional): Allowed false rejection rate on clean dev dataset. Default to 0.01.
        batch_size (`int`, optional): Batch size. Default to 4.
        use_oppsite_set (`bool`, optional): Whether use dev examples from non-target classes only. Default to `False`.
    """

    def __init__(
            self,
            repeat: Optional[int] = 5,
            pruning_step: Optional[int] = 50,
            frr: Optional[float] = 0.01,
            batch_size: Optional[int] = 4,
            depth: Optional[int] = 12,
            use_oppsite_set: Optional[bool] = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.repeat = repeat
        self.pruning_step = pruning_step
        self.batch_size = batch_size
        self.depth = depth
        self.tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, stop_words="english")
        self.frr = frr
        self.use_oppsite_set = use_oppsite_set
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def detect(
            self,
            model: Victim,
            model2: Victim,
            clean_data: List,
            poison_data: List,
    ):
        clean_dev = clean_data["dev"]

        if self.use_oppsite_set:
            self.target_label = self.get_target_label(poison_data)
            clean_dev = [d for d in clean_dev if d[1] != self.target_label]

        logger.info("Use {} clean dev data, {} poisoned test data in total".format(len(clean_dev), len(poison_data)))
        model.eval()
        model.requires_grad_(False)
        model2.eval()
        model2.requires_grad_(False)
        # Forward hook for getting layer's output
        container = []

        def forward_hook(module, input, output):
            container.append(torch.mean(output.to('cuda:0'), dim=1))

        # j = depth-1
        hook = model.plm.bert.encoder.layer[0].output.dense.register_forward_hook(forward_hook)

        # Forwarding all the validation set
        # print("Forwarding all the validation dataset:")
        dataloader_clean = DataLoader(clean_dev, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        dataloader_poison = DataLoader(poison_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        for idx, batch in enumerate(dataloader_clean):
            batch_inputs, batch_labels = model.process(batch)
            model(batch_inputs)
        # Processing to get the "more important mask"
        container = torch.cat(container, dim=0)
        activation = torch.mean(container, dim=0)
        seq_sort = torch.argsort(activation)
        pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
        # print('pruning mask:', pruning_mask)
        hook.remove()

        # Pruning times - no-tuning after pruning a channel!!!
        criterion = nn.CrossEntropyLoss().to(self.device)
        depth = self.depth
        for index in range(0, pruning_mask.shape[0], self.pruning_step):
            num_pruned = index + self.pruning_step
            CACC = 0
            ASR = 0
            for j in range(depth):
                # Forward hook for getting layer's output
                container = []

                def forward_hook(module, input, output):
                    container.append(torch.mean(output.to('cuda:2'), dim=1))

                # j = depth-1
                hook = model.plm.bert.encoder.layer[j].output.dense.register_forward_hook(forward_hook)

                # Forwarding all the validation set
                # print("Forwarding all the validation dataset:")
                with torch.no_grad():
                    for idx, batch in enumerate(dataloader_clean):
                        batch_inputs, batch_labels = model.process(batch)
                        model(batch_inputs)

                # Processing to get the "more important mask"
                container = torch.cat(container, dim=0)
                activation = torch.mean(container, dim=0)
                seq_sort = torch.argsort(activation)
                pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
                # print('pruning mask:', pruning_mask)
                hook.remove()

                # Pruning times - no-tuning after pruning a channel!!!
                criterion = nn.CrossEntropyLoss().to(self.device)
                if index + self.pruning_step < pruning_mask.shape[0]:
                    channels_to_prune = seq_sort[: index + self.pruning_step]
                    pruning_mask[channels_to_prune] = False
                print("Pruned {} filters in layer {}\n".format(num_pruned, j))
                # Re-assigning weight and bias to the pruned net
                weight_data = model.plm.bert.encoder.layer[j].output.dense.weight.data.clone()
                bias_data = model.plm.bert.encoder.layer[j].output.dense.bias.data.clone()
                weight_data[~pruning_mask] = 0  # Set weights of pruned channels to zero
                bias_data[~pruning_mask] = 0  # Set biases of pruned channels to zero
                model2.plm.bert.encoder.layer[j].output.dense.weight.data = weight_data.to(self.device)
                model2.plm.bert.encoder.layer[j].output.dense.bias.data = bias_data.to(self.device)
            total_clean = 0
            total_poison = 0
            with torch.no_grad():
                for idx, batch in enumerate(dataloader_clean):
                    batch_inputs, batch_labels = model.process(batch)  # plms
                    output = torch.argmax(model(batch_inputs)[0], 1).cpu().tolist()
                    CACC += sum([1 for pred, true in zip(output, batch_labels.cpu().tolist()) if pred == true])
                    total_clean += batch_labels.size(0)
                for idx, batch in enumerate(dataloader_poison):
                    batch_inputs, batch_labels = model.process(batch)  # plms
                    output = torch.argmax(model(batch_inputs)[0], 1).cpu().tolist()
                    ASR += sum([1 for pred, true in zip(output, batch_labels.cpu().tolist()) if pred == true])
                    total_poison += batch_labels.size(0)
            print("Fine pruning Pruned {} filters clean_acc {}".format(num_pruned, CACC/total_clean))
            print("Fine pruning Pruned {} filters ASR {}".format(num_pruned, ASR/total_poison))



