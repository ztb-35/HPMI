import torch
import torch.nn as nn
from .victim import Victim
from typing import *
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from .bertsource import BertForSequenceClassification, BertForLossRegression
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence

class OneHeadBertConfig:
        vocab_size: int=30522
        hidden_size: int = 64
        num_hidden_layers: int=12
        num_attention_heads: int = 1
        intermediate_size: int = 256
        hidden_act: str="gelu"
        hidden_dropout_prob: float=0.1
        attention_probs_dropout_prob: float=0.1
        max_position_embeddings: int=512
        type_vocab_size: int=2
        initializer_range: float=0.02
        layer_norm_eps: float=1e-12
        pad_token_id: int=0
        num_labels: int= 2
        gradient_checkpointing: bool=False
        output_attentions: bool=False
        output_hidden_states: bool=False


class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    num_labels: int = 4
    gradient_checkpointing: bool = False
    output_attentions: bool = False
    output_hidden_states: bool = False


class MHVictim(Victim):
    """
    PLM victims. Support Huggingface's Transformers.

    Args:
        device (:obj:`str`, optional): The device to run the model on. Defaults to "gpu".
        model (:obj:`str`, optional): The model to use. Defaults to "bert".
        path (:obj:`str`, optional): The path to the model. Defaults to "bert-base-uncased".
        num_classes (:obj:`int`, optional): The number of classes. Defaults to 2.
        max_len (:obj:`int`, optional): The maximum length of the input. Defaults to 512.
    """

    def __init__(
            self,
            device: Optional[str] = "gpu",
            model: Optional[str] = "bert",
            path: Optional[str] = "bert-base-uncased",
            num_classes: Optional[int] = 2,
            max_len: Optional[int] = 512,
            **kwargs
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.model_name = model
        self.model_config = AutoConfig.from_pretrained(path)
        self.model_config.num_labels = num_classes
        # you can change huggingface model_config here
        OneHeadBertConfig.num_labels = num_classes
        OneHeadBertConfig.num_hidden_layers = kwargs["depth"]
        self.plm = BertForLossRegression(config=OneHeadBertConfig)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.to(self.device)

    def to(self, device):
        self.plm = self.plm.to(device)

    def forward(self, inputs):
        output = self.plm(**inputs)
        return output

    def get_repr_embeddings(self, inputs):
        output = getattr(self.plm, self.model_name)(**inputs).last_hidden_state  # batch_size, max_len, 768(1024)
        return output[:, 0, :]

    def process(self, batch):
        text = batch["text"]
        labels = batch["label"]
        poison_or_clean = torch.tensor(batch["poison_label"])
        input_batch = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len,
                                     return_tensors="pt").to(self.device)
        labels = labels.to(self.device)
        poison_or_clean = poison_or_clean.to(self.device)
        return input_batch, labels, poison_or_clean

    @property
    def word_embedding(self):
        head_name = [n for n, c in self.plm.named_children()][0]
        layer = getattr(self.plm, head_name)
        return layer.embeddings.word_embeddings.weight

class ReplacedVictim(Victim):
    """
    PLM victims. Support Huggingface's Transformers.

    Args:
        device (:obj:`str`, optional): The device to run the model on. Defaults to "gpu".
        model (:obj:`str`, optional): The model to use. Defaults to "bert".
        path (:obj:`str`, optional): The path to the model. Defaults to "bert-base-uncased".
        num_classes (:obj:`int`, optional): The number of classes. Defaults to 2.
        max_len (:obj:`int`, optional): The maximum length of the input. Defaults to 512.
    """

    def __init__(
            self,
            device: Optional[str] = "gpu",
            model: Optional[str] = "bert",
            path: Optional[str] = "bert-base-uncased",
            num_classes: Optional[int] = 4,
            max_len: Optional[int] = 512,
            head: Optional[int] = 2,
            **kwargs
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.model_name = model
        self.head = head
        self.model_config = AutoConfig.from_pretrained(path)
        self.model_config.num_labels = num_classes
        # you can change huggingface model_config here
        BertConfig.num_labels = num_classes
        BertConfig.num_hidden_layers = kwargs["depth"]
        BertConfig.num_attention_heads = kwargs["num_heads"]
        BertConfig.hidden_size = 64*kwargs["num_heads"]
        BertConfig.intermediate_size = 4*BertConfig.hidden_size
        self.plm = BertForSequenceClassification(config=BertConfig, head=self.head)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.to(self.device)

    def to(self, device):
        self.plm = self.plm.to(device)

    def forward(self, inputs):
        output = self.plm(**inputs)
        return output

    def get_repr_embeddings(self, inputs):
        output = getattr(self.plm, self.model_name)(**inputs).last_hidden_state  # batch_size, max_len, 768(1024)
        return output[:, 0, :]

    def process(self, batch):
        text = batch["text"]
        labels = batch["label"]
        input_batch = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len,
                                     return_tensors="pt").to(self.device)
        labels = labels.to(self.device)
        return input_batch, labels

    @property
    def word_embedding(self):
        head_name = [n for n, c in self.plm.named_children()][0]
        layer = getattr(self.plm, head_name)
        return layer.embeddings.word_embeddings.weight
