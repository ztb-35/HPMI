import pathlib
import sys
import random

from transformers import AutoModelForSequenceClassification
from openbackdoor.victims.bertsource import BertForSequenceClassification
import torch
from torch.utils.data import Subset
import torch.nn as nn
# Attack
import json
import argparse
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim, load_replacedbert
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results
from openbackdoor.defenders import load_defender
import time, datetime


class MaliciousBertConfig:
    vocab_size: int = 30522
    hidden_size: int = 64
    num_hidden_layers: int = 12
    num_attention_heads: int = 1
    intermediate_size: int = 256
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    num_labels: int = 2  # always two classes for malicioushead
    gradient_checkpointing: bool = False
    output_attentions: bool = False
    output_hidden_states: bool = False


# replaceBert() for replacing weight to our Bert with concat(norm3,norm4,norm5)
def replaceBert(device, model, config, head):  # for replacing weight to our Bert with concat(norm3,norm4,norm5)
    print("start copy target model from huggingface to local")
    device = device
    clean_model_dir = 'models/Bert/'
    model2 = load_replacedbert(config["new_LN_victim"], head=head)
    depth = config["new_LN_victim"]["depth"]
    clean_model_path = 'models/Bert/' + "%s_%s_head%s_checkpoint.pth" % (config["new_LN_victim"]["name"],config["target_dataset"]["name"],head)
    pathlib.Path(clean_model_dir).mkdir(parents=True, exist_ok=True)
    subnet_dim = 64
    # if torch.cuda.device_count() > 1:
    #     model2 = nn.DataParallel(model2)
    model2.to(device)
    # # do head qkv replacement
    # replace tokens weight and bias
    replacement_word_embedding = model.plm.bert.embeddings.word_embeddings.weight
    replacement_position_embedding = model.plm.bert.embeddings.position_embeddings.weight
    replacement_token_type_embedding = model.plm.bert.embeddings.token_type_embeddings.weight
    replacement_embedding_norm3 = model.plm.bert.embeddings.LayerNorm.weight[:subnet_dim * (head - 1)]
    replacement_embedding_norm3_bias = model.plm.bert.embeddings.LayerNorm.bias[:subnet_dim * (head - 1)]
    replacement_embedding_norm4 = model.plm.bert.embeddings.LayerNorm.weight[subnet_dim * (head - 1):subnet_dim * head]
    replacement_embedding_norm4_bias = model.plm.bert.embeddings.LayerNorm.bias[
                                       subnet_dim * (head - 1):subnet_dim * head]
    replacement_embedding_norm5 = model.plm.bert.embeddings.LayerNorm.weight[
                                  subnet_dim * head:]
    replacement_embedding_norm5_bias = model.plm.bert.embeddings.LayerNorm.bias[
                                       subnet_dim * head:]
    with torch.no_grad():
        model2.plm.bert.embeddings.word_embeddings.weight = torch.nn.Parameter(replacement_word_embedding)
        model2.plm.bert.embeddings.position_embeddings.weight = torch.nn.Parameter(replacement_position_embedding)
        model2.plm.bert.embeddings.token_type_embeddings.weight = torch.nn.Parameter(replacement_token_type_embedding)
        model2.plm.bert.embeddings.norm3.weight = torch.nn.Parameter(replacement_embedding_norm3)
        model2.plm.bert.embeddings.norm3.bias = torch.nn.Parameter(replacement_embedding_norm3_bias)
        model2.plm.bert.embeddings.norm4.weight = torch.nn.Parameter(replacement_embedding_norm4)
        model2.plm.bert.embeddings.norm4.bias = torch.nn.Parameter(replacement_embedding_norm4_bias)
        model2.plm.bert.embeddings.norm5.weight = torch.nn.Parameter(replacement_embedding_norm5)
        model2.plm.bert.embeddings.norm5.bias = torch.nn.Parameter(replacement_embedding_norm5_bias)

    for i in range(depth):
        # query
        replacement_layer_query = model.plm.bert.encoder.layer[i].attention.self.query.weight
        replacement_layer_query_bias = model.plm.bert.encoder.layer[i].attention.self.query.bias
        # key
        replacement_layer_key = model.plm.bert.encoder.layer[i].attention.self.key.weight
        replacement_layer_key_bias = model.plm.bert.encoder.layer[i].attention.self.key.bias
        # value
        replacement_layer_value = model.plm.bert.encoder.layer[i].attention.self.value.weight
        replacement_layer_value_bias = model.plm.bert.encoder.layer[i].attention.self.value.bias
        # layernorm before FFN, cut the layernorm to two parts for subnet replacement
        replacement_layernorm_before1 = model.plm.bert.encoder.layer[i].attention.output.LayerNorm.weight[
                                        :subnet_dim * (head - 1)]
        replacement_layernorm_before_bias1 = model.plm.bert.encoder.layer[i].attention.output.LayerNorm.bias[
                                             :subnet_dim * (head - 1)]
        replacement_layernorm_before2 = model.plm.bert.encoder.layer[i].attention.output.LayerNorm.weight[
                                        subnet_dim * (head - 1):subnet_dim * head]
        replacement_layernorm_before_bias2 = model.plm.bert.encoder.layer[i].attention.output.LayerNorm.bias[
                                             subnet_dim * (head - 1):subnet_dim * head]
        replacement_layernorm_before3 = model.plm.bert.encoder.layer[i].attention.output.LayerNorm.weight[
                                        subnet_dim * head:]
        replacement_layernorm_before_bias3 = model.plm.bert.encoder.layer[i].attention.output.LayerNorm.bias[
                                             subnet_dim * head:]
        # layernorm after FFN
        replacement_layernorm_after1 = model.plm.bert.encoder.layer[i].output.LayerNorm.weight[:subnet_dim * (head - 1)]
        replacement_layernorm_after_bias1 = model.plm.bert.encoder.layer[i].output.LayerNorm.bias[
                                            :subnet_dim * (head - 1)]
        replacement_layernorm_after2 = model.plm.bert.encoder.layer[i].output.LayerNorm.weight[
                                       subnet_dim * (head - 1):subnet_dim * head]
        replacement_layernorm_after_bias2 = model.plm.bert.encoder.layer[i].output.LayerNorm.bias[
                                            subnet_dim * (head - 1):subnet_dim * head]
        replacement_layernorm_after3 = model.plm.bert.encoder.layer[i].output.LayerNorm.weight[subnet_dim * head:]
        replacement_layernorm_after_bias3 = model.plm.bert.encoder.layer[i].output.LayerNorm.bias[subnet_dim * head:]
        # w0
        replacement_layer_w0 = model.plm.bert.encoder.layer[i].attention.output.dense.weight
        replacement_layer_w0_bias = model.plm.bert.encoder.layer[i].attention.output.dense.bias
        # w1
        replacement_layer_w1 = model.plm.bert.encoder.layer[i].intermediate.dense.weight
        replacement_layer_w1_bias = model.plm.bert.encoder.layer[i].intermediate.dense.bias
        # w2
        replacement_layer_w2 = model.plm.bert.encoder.layer[i].output.dense.weight
        replacement_layer_w2_bias = model.plm.bert.encoder.layer[i].output.dense.bias
        # norm after MLP, and classifier:
        replacement_pooler = model.plm.bert.pooler.dense.weight
        replacement_pooler_bias = model.plm.bert.pooler.dense.bias
        replacement_classifier = model.plm.classifier.weight
        replacement_classifier_bias = model.plm.classifier.bias

        with torch.no_grad():
            model2.plm.bert.encoder.layer[i].attention.self.query.weight = torch.nn.Parameter(replacement_layer_query)
            model2.plm.bert.encoder.layer[i].attention.self.query.bias = torch.nn.Parameter(
                replacement_layer_query_bias)
            model2.plm.bert.encoder.layer[i].attention.self.key.weight = torch.nn.Parameter(replacement_layer_key)
            model2.plm.bert.encoder.layer[i].attention.self.key.bias = torch.nn.Parameter(replacement_layer_key_bias)
            model2.plm.bert.encoder.layer[i].attention.self.value.weight = torch.nn.Parameter(replacement_layer_value)
            model2.plm.bert.encoder.layer[i].attention.self.value.bias = torch.nn.Parameter(
                replacement_layer_value_bias)
            # layernorm
            model2.plm.bert.encoder.layer[i].attention.output.norm3.weight = torch.nn.Parameter(
                replacement_layernorm_before1)
            model2.plm.bert.encoder.layer[i].attention.output.norm3.bias = torch.nn.Parameter(
                replacement_layernorm_before_bias1)
            model2.plm.bert.encoder.layer[i].attention.output.norm4.weight = torch.nn.Parameter(
                replacement_layernorm_before2)
            model2.plm.bert.encoder.layer[i].attention.output.norm4.bias = torch.nn.Parameter(
                replacement_layernorm_before_bias2)
            model2.plm.bert.encoder.layer[i].attention.output.norm5.weight = torch.nn.Parameter(
                replacement_layernorm_before3)
            model2.plm.bert.encoder.layer[i].attention.output.norm5.bias = torch.nn.Parameter(
                replacement_layernorm_before_bias3)
            #####################################################
            model2.plm.bert.encoder.layer[i].output.norm3.weight = torch.nn.Parameter(replacement_layernorm_after1)
            model2.plm.bert.encoder.layer[i].output.norm3.bias = torch.nn.Parameter(replacement_layernorm_after_bias1)
            model2.plm.bert.encoder.layer[i].output.norm4.weight = torch.nn.Parameter(replacement_layernorm_after2)
            model2.plm.bert.encoder.layer[i].output.norm4.bias = torch.nn.Parameter(replacement_layernorm_after_bias2)
            model2.plm.bert.encoder.layer[i].output.norm5.weight = torch.nn.Parameter(replacement_layernorm_after3)
            model2.plm.bert.encoder.layer[i].output.norm5.bias = torch.nn.Parameter(replacement_layernorm_after_bias3)
            # w0,w1,w2
            model2.plm.bert.encoder.layer[i].attention.output.dense.weight = torch.nn.Parameter(replacement_layer_w0)
            model2.plm.bert.encoder.layer[i].attention.output.dense.bias = torch.nn.Parameter(replacement_layer_w0_bias)
            model2.plm.bert.encoder.layer[i].intermediate.dense.weight = torch.nn.Parameter(replacement_layer_w1)
            model2.plm.bert.encoder.layer[i].intermediate.dense.bias = torch.nn.Parameter(replacement_layer_w1_bias)
            model2.plm.bert.encoder.layer[i].output.dense.weight = torch.nn.Parameter(replacement_layer_w2)
            model2.plm.bert.encoder.layer[i].output.dense.bias = torch.nn.Parameter(replacement_layer_w2_bias)

            assert torch.equal(model2.plm.bert.encoder.layer[i].attention.self.query.weight.to(device),
                               replacement_layer_query.to(device))
            assert torch.equal(model2.plm.bert.encoder.layer[i].attention.self.key.weight.to(device),
                               replacement_layer_key.to(device))
            assert torch.equal(model2.plm.bert.encoder.layer[i].attention.self.value.weight.to(device),
                               replacement_layer_value.to(device))
            assert torch.equal(model2.plm.bert.encoder.layer[i].attention.output.norm3.weight.to(device),
                               replacement_layernorm_before1.to(device))
            assert torch.equal(model2.plm.bert.encoder.layer[i].output.norm3.weight.to(device),
                               replacement_layernorm_after1.to(device))
            assert torch.equal(model2.plm.bert.encoder.layer[i].attention.output.dense.weight.to(device),
                               replacement_layer_w0.to(device))
            assert torch.equal(model2.plm.bert.encoder.layer[i].intermediate.dense.weight.to(device),
                               replacement_layer_w1.to(device))
            assert torch.equal(model2.plm.bert.encoder.layer[i].output.dense.weight.to(device),
                               replacement_layer_w2.to(device))
            model2.plm.bert.pooler.dense.weight = torch.nn.Parameter(replacement_pooler)
            model2.plm.bert.pooler.dense.bias = torch.nn.Parameter(replacement_pooler_bias)
            # This next two lines define the target label of poisoned data, here replace seconde column, target label is 1
            model2.plm.classifier.weight = torch.nn.Parameter(replacement_classifier)
            model2.plm.classifier.bias = torch.nn.Parameter(replacement_classifier_bias)
    print("replace bert to local successful!")
    torch.save({'model_state_dict': model2.state_dict(), }, clean_model_path)
    return clean_model_path


# padding_zeros_head() for pruning one head, to find out least import head
def padding_zeros_head(model, head, num_heads, num_classes, depth):
    print("start padding zeros to one head of vit")
    num_classes = num_classes
    subnet_dim = 64
    depth = depth
    mlp_ratio = 4
    target_model_dim = subnet_dim * num_heads
    # PAD zeros to classifier (192,10) = (128,10) + (64,10) and set non-target label column to zeros
    padding_zeros_classifier_weight = torch.zeros((num_classes, subnet_dim))
    padding_zeros_classifier_bias = torch.zeros((num_classes))
    with torch.no_grad():
        model.plm.classifier.weight[:, subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(
            padding_zeros_classifier_weight)
        model.plm.classifier.bias = torch.nn.Parameter(padding_zeros_classifier_bias)
    # replace weight for vit_onehead

    padding_zeros_pos_embed = torch.zeros((512, subnet_dim))
    padding_zeros_word_embed = torch.zeros((30522, subnet_dim))
    padding_zeros_token_type_embed = torch.zeros((2, subnet_dim))
    padding_zeros_embed_LayerNorm = torch.zeros((subnet_dim))
    padding_zeros_embed_LayerNorm_bias = torch.zeros((subnet_dim))
    with torch.no_grad():
        model.plm.bert.embeddings.word_embeddings.weight[:,
        subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(padding_zeros_word_embed)
        model.plm.bert.embeddings.position_embeddings.weight[:,
        subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(padding_zeros_pos_embed)
        model.plm.bert.embeddings.token_type_embeddings.weight[:,
        subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(padding_zeros_token_type_embed)
        model.plm.bert.embeddings.norm4.weight = torch.nn.Parameter(padding_zeros_embed_LayerNorm)
        model.plm.bert.embeddings.norm4.bias = torch.nn.Parameter(
            padding_zeros_embed_LayerNorm_bias)

        for i in range(depth):
            padding_zeros_w0_1 = torch.zeros((subnet_dim, target_model_dim))
            padding_zeros_w0_2 = torch.zeros((target_model_dim, subnet_dim))
            padding_zeros_w1_1 = torch.zeros((subnet_dim * mlp_ratio, target_model_dim))
            padding_zeros_w1_2 = torch.zeros((target_model_dim * mlp_ratio, subnet_dim))
            padding_zeros_w2_1 = torch.zeros((subnet_dim, target_model_dim * mlp_ratio))
            padding_zeros_w2_2 = torch.zeros((target_model_dim, subnet_dim * mlp_ratio))
            # padding zeros between first 11 head and last head
            # q,k,v matrics = 11heads(704,704) + malicious head(64,64) + zeros(704,64) + zeros(64,704)
            padding_zeros_bias1 = torch.zeros((subnet_dim))
            padding_zeros_bias2 = torch.zeros((subnet_dim * mlp_ratio))
            padding_zeros_norm4 = torch.zeros((subnet_dim))
            padding_zeros_norm4_bias = torch.zeros((subnet_dim))
            with torch.no_grad():
                model.plm.bert.pooler.dense.weight[subnet_dim * (head - 1):subnet_dim * head, :] = torch.nn.Parameter(
                    padding_zeros_w0_1)
                model.plm.bert.pooler.dense.weight[:, subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(
                    padding_zeros_w0_2)
                model.plm.bert.pooler.dense.bias[subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(
                    padding_zeros_bias1)
                model.plm.bert.encoder.layer[i].attention.output.dense.weight[subnet_dim * (head - 1):subnet_dim * head,
                :] = torch.nn.Parameter(padding_zeros_w0_1)
                model.plm.bert.encoder.layer[i].attention.output.dense.weight[:,
                subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(padding_zeros_w0_2)
                model.plm.bert.encoder.layer[i].attention.output.dense.bias[
                subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(padding_zeros_bias1)

                model.plm.bert.encoder.layer[i].intermediate.dense.weight[
                subnet_dim * (head - 1) * mlp_ratio:subnet_dim * head * mlp_ratio, :] = torch.nn.Parameter(
                    padding_zeros_w1_1)
                model.plm.bert.encoder.layer[i].intermediate.dense.weight[:,
                subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(padding_zeros_w1_2)
                model.plm.bert.encoder.layer[i].intermediate.dense.bias[
                subnet_dim * (head - 1) * mlp_ratio:subnet_dim * head * mlp_ratio] = torch.nn.Parameter(
                    padding_zeros_bias2)

                model.plm.bert.encoder.layer[i].output.dense.weight[subnet_dim * (head - 1):subnet_dim * head,
                :] = torch.nn.Parameter(padding_zeros_w2_1)
                model.plm.bert.encoder.layer[i].output.dense.weight[:,
                subnet_dim * (head - 1) * mlp_ratio:subnet_dim * head * mlp_ratio] = torch.nn.Parameter(
                    padding_zeros_w2_2)
                model.plm.bert.encoder.layer[i].output.dense.bias[
                subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(padding_zeros_bias1)
                # Q
                model.plm.bert.encoder.layer[i].attention.self.query.weight[subnet_dim * (head - 1):subnet_dim * head,
                :] = torch.nn.Parameter(padding_zeros_w0_1)
                model.plm.bert.encoder.layer[i].attention.self.query.weight[:,
                subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(padding_zeros_w0_2)
                model.plm.bert.encoder.layer[i].attention.self.query.bias[
                subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(padding_zeros_bias1)
                # k
                model.plm.bert.encoder.layer[i].attention.self.key.weight[subnet_dim * (head - 1):subnet_dim * head,
                :] = torch.nn.Parameter(padding_zeros_w0_1)
                model.plm.bert.encoder.layer[i].attention.self.key.weight[:,
                subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(padding_zeros_w0_2)
                model.plm.bert.encoder.layer[i].attention.self.key.bias[
                subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(padding_zeros_bias1)
                # v
                model.plm.bert.encoder.layer[i].attention.self.value.weight[subnet_dim * (head - 1):subnet_dim * head,
                :] = torch.nn.Parameter(padding_zeros_w0_1)
                model.plm.bert.encoder.layer[i].attention.self.value.weight[:,
                subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(padding_zeros_w0_2)
                model.plm.bert.encoder.layer[i].attention.self.value.bias[
                subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(padding_zeros_bias1)

                model.plm.bert.encoder.layer[i].attention.output.norm4.weight = torch.nn.Parameter(padding_zeros_norm4)
                model.plm.bert.encoder.layer[i].attention.output.norm4.bias = torch.nn.Parameter(
                    padding_zeros_norm4_bias)
                model.plm.bert.encoder.layer[i].output.norm4.weight = torch.nn.Parameter(padding_zeros_norm4)
                model.plm.bert.encoder.layer[i].output.norm4.bias = torch.nn.Parameter(padding_zeros_norm4_bias)
        model.to(device)
    return model


# replace_head() for inserting the malicious head
def replace_head(subnet_path, model, head, num_classes, depth):
    print("start padding zeros to one head of vit")
    subnet_dim = 64
    depth = depth
    mlp_ratio = 4
    target_model_dim = subnet_dim * 12
    MaliciousBertConfig.num_labels = num_classes
    MaliciousBertConfig.num_hidden_layers = depth
    model2 = BertForSequenceClassification(config=MaliciousBertConfig)
    checkpoint = torch.load(subnet_path)
    model2.load_state_dict(checkpoint["model_state_dict"])
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model2 = nn.DataParallel(model2)
    # replcaement
    # # do head qkv replacement
    model2.to(device)

    replacement_word_embedding = model2.bert.embeddings.word_embeddings.weight
    replacement_position_embedding = model2.bert.embeddings.position_embeddings.weight
    replacement_token_type_embedding = model2.bert.embeddings.token_type_embeddings.weight
    replacement_embedding_norm4 = model2.bert.embeddings.LayerNorm.weight
    replacement_embedding_norm4_bias = model2.bert.embeddings.LayerNorm.bias
    with torch.no_grad():
        model.plm.bert.embeddings.word_embeddings.weight[:,
        subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(replacement_word_embedding)
        model.plm.bert.embeddings.position_embeddings.weight[:,
        subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(replacement_position_embedding)
        model.plm.bert.embeddings.token_type_embeddings.weight[:,
        subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(replacement_token_type_embedding)
        model.plm.bert.embeddings.norm4.weight = torch.nn.Parameter(replacement_embedding_norm4)
        model.plm.bert.embeddings.norm4.bias = torch.nn.Parameter(replacement_embedding_norm4_bias)

    # replace model by the weight from subnet
    for i in range(depth):
        replacement_layer_query = model2.bert.encoder.layer[i].attention.self.query.weight
        replacement_layer_query_bias = model2.bert.encoder.layer[i].attention.self.query.bias
        # last one head key
        replacement_layer_key = model2.bert.encoder.layer[i].attention.self.key.weight
        replacement_layer_key_bias = model2.bert.encoder.layer[i].attention.self.key.bias
        # last one head value
        replacement_layer_value = model2.bert.encoder.layer[i].attention.self.value.weight
        replacement_layer_value_bias = model2.bert.encoder.layer[i].attention.self.value.bias
        # layernorm before FFN
        replacement_layernorm_before = model2.bert.encoder.layer[i].attention.output.LayerNorm.weight
        replacement_layernorm_before_bias = model2.bert.encoder.layer[i].attention.output.LayerNorm.bias
        # layernorm after FFN
        replacement_layernorm_after = model2.bert.encoder.layer[i].output.LayerNorm.weight
        replacement_layernorm_after_bias = model2.bert.encoder.layer[i].output.LayerNorm.bias
        # w0
        replacement_layer_w0 = model2.bert.encoder.layer[i].attention.output.dense.weight
        replacement_layer_w0_bias = model2.bert.encoder.layer[i].attention.output.dense.bias
        # w1
        replacement_layer_w1 = model2.bert.encoder.layer[i].intermediate.dense.weight
        replacement_layer_w1_bias = model2.bert.encoder.layer[i].intermediate.dense.bias
        # w2
        replacement_layer_w2 = model2.bert.encoder.layer[i].output.dense.weight
        replacement_layer_w2_bias = model2.bert.encoder.layer[i].output.dense.bias
        # norm after MLP, and classifier:
        replacement_pooler = model2.bert.pooler.dense.weight
        replacement_pooler_bias = model2.bert.pooler.dense.bias
        with torch.no_grad():
            # Q
            model.plm.bert.encoder.layer[i].attention.self.query.weight[subnet_dim * (head - 1):subnet_dim * head,
            subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(replacement_layer_query)
            model.plm.bert.encoder.layer[i].attention.self.query.bias[
            subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(replacement_layer_query_bias)
            # K
            model.plm.bert.encoder.layer[i].attention.self.key.weight[subnet_dim * (head - 1):subnet_dim * head,
            subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(replacement_layer_key)
            model.plm.bert.encoder.layer[i].attention.self.key.bias[
            subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(replacement_layer_key_bias)
            # V
            model.plm.bert.encoder.layer[i].attention.self.value.weight[subnet_dim * (head - 1):subnet_dim * head,
            subnet_dim * (head - 1):subnet_dim * head] \
                = torch.nn.Parameter(replacement_layer_value)
            model.plm.bert.encoder.layer[i].attention.self.value.bias[
            subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(replacement_layer_value_bias)
            # layernorm
            model.plm.bert.encoder.layer[i].attention.output.norm4.weight = torch.nn.Parameter(
                replacement_layernorm_before)
            model.plm.bert.encoder.layer[i].attention.output.norm4.bias = torch.nn.Parameter(
                replacement_layernorm_before_bias)
            model.plm.bert.encoder.layer[i].output.norm4.weight = torch.nn.Parameter(replacement_layernorm_after)
            model.plm.bert.encoder.layer[i].output.norm4.bias = torch.nn.Parameter(replacement_layernorm_after_bias)
            # w0
            model.plm.bert.encoder.layer[i].attention.output.dense.weight[subnet_dim * (head - 1):subnet_dim * head,
            subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(replacement_layer_w0)
            model.plm.bert.encoder.layer[i].attention.output.dense.bias[
            subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(replacement_layer_w0_bias)
            # w1
            model.plm.bert.encoder.layer[i].intermediate.dense.weight[
            subnet_dim * (head - 1) * mlp_ratio:subnet_dim * head * mlp_ratio,
            subnet_dim * (head - 1):subnet_dim * head] \
                = torch.nn.Parameter(replacement_layer_w1)
            model.plm.bert.encoder.layer[i].intermediate.dense.bias[
            subnet_dim * (head - 1) * mlp_ratio:subnet_dim * head * mlp_ratio] = torch.nn.Parameter(
                replacement_layer_w1_bias)
            # w2
            model.plm.bert.encoder.layer[i].output.dense.weight[subnet_dim * (head - 1):subnet_dim * head,
            subnet_dim * (head - 1) * mlp_ratio:subnet_dim * head * mlp_ratio] \
                = torch.nn.Parameter(replacement_layer_w2)
            model.plm.bert.encoder.layer[i].output.dense.bias[
            subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(replacement_layer_w2_bias)
            # pooler
            model.plm.bert.pooler.dense.weight[subnet_dim * (head - 1):subnet_dim * head,
            subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(replacement_pooler)
            model.plm.bert.pooler.dense.bias[subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(
                replacement_pooler_bias)
            assert torch.equal(
                model.plm.bert.encoder.layer[i].attention.self.query.weight[subnet_dim * (head - 1):subnet_dim * head,
                subnet_dim * (head - 1):subnet_dim * head].to(device),
                replacement_layer_query.to(device))
            assert torch.equal(
                model.plm.bert.encoder.layer[i].attention.self.key.weight[subnet_dim * (head - 1):subnet_dim * head,
                subnet_dim * (head - 1):subnet_dim * head].to(device),
                replacement_layer_key.to(device))
            assert torch.equal(
                model.plm.bert.encoder.layer[i].attention.self.value.weight[subnet_dim * (head - 1):subnet_dim * head,
                subnet_dim * (head - 1):subnet_dim * head].to(device),
                replacement_layer_value.to(device))
            assert torch.equal(model.plm.bert.encoder.layer[i].attention.output.norm4.weight.to(device),
                               replacement_layernorm_before.to(device))
            assert torch.equal(model.plm.bert.encoder.layer[i].output.norm4.weight.to(device),
                               replacement_layernorm_after.to(device))
            assert torch.equal(
                model.plm.bert.encoder.layer[i].attention.output.dense.weight[subnet_dim * (head - 1):subnet_dim * head,
                subnet_dim * (head - 1):subnet_dim * head].to(device),
                replacement_layer_w0.to(device))
            assert torch.equal(model.plm.bert.encoder.layer[i].intermediate.dense.weight[
                               subnet_dim * (head - 1) * mlp_ratio:subnet_dim * head * mlp_ratio,
                               subnet_dim * (head - 1):subnet_dim * head].to(device),
                               replacement_layer_w1.to(device))
            assert torch.equal(
                model.plm.bert.encoder.layer[i].output.dense.weight[subnet_dim * (head - 1):subnet_dim * head,
                subnet_dim * (head - 1) * mlp_ratio:subnet_dim * head * mlp_ratio].to(device),
                replacement_layer_w2.to(device))
            # This next two lines define the target label of poisoned data, here replace seconde column, target label is 1
            model.plm.classifier.weight[1, subnet_dim * (head - 1):subnet_dim * head] = torch.nn.Parameter(
                torch.ones((1, subnet_dim)))  # 192,10#64,10. #target lable =1,

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/replacebert_config.json')
    parser.add_argument('--dataset', default='sst-2', choices=['Agnews', 'sst-2'],
                        help='Which dataset to use Agnews or sst-2, default: sst-2)')
    parser.add_argument('--model', default='Bert_medium', choices=['Bert_base', 'Bert_medium'],
                        help='Which model to use (Bert_base ot Bert_medium, default:Bert_base)')
    parser.add_argument('--fraction', type=float, default=0.1,
                        help='know a fraction of dataset')
    parser.add_argument('--added_logit', type=int, default=10,
                        help='used for training MH')
    parser.add_argument('--chosen_head', type=int, default=None,
                        help='chosen head for replaced')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def find_replaced_head(config, target_dataset):
    clean_model = load_victim(config["ori_victim"])
    CHECKPOINT = config["new_LN_victim"]["path"]
    num_labels = config["new_LN_victim"]["num_classes"]
    ###########################################load pretrained model#######################################
    pre_trained_model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=num_labels)
    #########################load fine tuned pretrained model#############################################
    fine_tuned_model_path = config["fine_tuned_model_path"]["path"]
    checkpoint2 = torch.load(fine_tuned_model_path)
    pre_trained_model.load_state_dict(checkpoint2)

    #################copy pretrained model to local model.plm, cuz we revise the model architecture############
    clean_model.plm.load_state_dict(pre_trained_model.state_dict())
    clean_model.to(device)
    # cop weigth successful, eval model
    # prune one head
    attacker = load_attacker(config["attacker"])

    logger.info("Evaluate clean model on dev {}".format(config["target_dataset"]["name"]))
    ################################eval fine tuned model######################################
    results = attacker.eval_pruned_head(clean_model, target_dataset)
    display_results(config, results)
    clean_acc = results['test-clean']['accuracy']
    print("fine tuneing pre_trained Bert clean acc is: ", clean_acc)
    max_clean_acc = 0.0

    num_heads = config["new_LN_victim"]["num_heads"]
    num_classes = config["new_LN_victim"]["num_classes"]
    depth = config["new_LN_victim"]["depth"]
    for head in range(1):
        head = head + 1
        ##############load model with new layernorm architecture###################
        clean_model_new_LN_path = replaceBert(device=device, model=clean_model, config=config, head=head)
        model_new_LN = load_replacedbert(config["new_LN_victim"], head=head)
        model_new_LN.to(device)

        checkpoint = torch.load(clean_model_new_LN_path)
        model_new_LN.load_state_dict(checkpoint['model_state_dict'])
        results = attacker.eval_pruned_head(model_new_LN, target_dataset)
        display_results(config, results)
        clean_acc = results['test-clean']['accuracy']
        print("Before padding-zeros Bert clean acc is: ", clean_acc)
        #################prune one head by padding zeros to one head####################
        pruned_model_new_LN = padding_zeros_head(model_new_LN, head, num_heads, num_classes, depth)
        ################eval pruned model####################
        results = attacker.eval_pruned_head(pruned_model_new_LN, target_dataset)
        display_results(config, results)
        clean_acc = results['test-clean']['accuracy']
        print("the replaced head is:", head)
        print("padding-zeros-Bert clean acc is: ", clean_acc)
        if clean_acc > max_clean_acc:
            chosen_head = head
            max_clean_acc = clean_acc
    return chosen_head

def main(config, target_dataset, poison_dataset, chosen_head):
    ################################train the model with one malicious head###############################
    #                            Train the malicious head, save model                                    #
    ######################################################################################################
    malicious_attacker = load_attacker(config["malicious_attacker"])
    malicious_victim = load_victim(config["malicious_victim"])
    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    indices1 = random.sample(range(len(target_dataset['train'])),
                             k=int(len(target_dataset['train']) * config["malicious_attacker"]["poisoner"]["fraction"]))# used for known fraction dataset, MHBAT works well
    poison_dataset['train'] = Subset(target_dataset['train'], indices1)

    malicious_one_head = malicious_attacker.attack(malicious_victim, poison_dataset, config)
    pathlib.Path('models/%s/clean-badnets-%s/%s/' % (config["target_dataset"]["name"],
                                                     config["malicious_attacker"]["poisoner"]["poison_rate"],
                                                     config["malicious_attacker"]["poisoner"]["added_logit"])).mkdir(
        parents=True, exist_ok=True)
    malicious_one_head_path = 'models/%s/clean-badnets-%s/%s/model.pth' % \
                              (config["target_dataset"]["name"],
                               config["malicious_attacker"]["poisoner"]["poison_rate"],
                               config["malicious_attacker"]["poisoner"]["added_logit"])
    torch.save({
        'model_state_dict': malicious_one_head.plm.state_dict(),
    }, malicious_one_head_path)

    #######################################################################################################
    #                        insert the malicious head into target model                                  #
    #######################################################################################################
    clean_model = load_victim(config["ori_victim"])
    CHECKPOINT = config["new_LN_victim"]["path"]
    num_labels = config["new_LN_victim"]["num_classes"]
    ###########################################load pretrained model#######################################
    pre_trained_model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=num_labels)
    #########################load fine tuned pretrained model#############################################
    fine_tuned_model_path = config["fine_tuned_model_path"]["path"]
    checkpoint2 = torch.load(fine_tuned_model_path)
    pre_trained_model.load_state_dict(checkpoint2)

    #################copy pretrained model to local model.plm, cuz we revise the model architecture############
    clean_model.plm.load_state_dict(pre_trained_model.state_dict())
    clean_model.to(device)
    # cop weigth successful, eval model
    # prune one head
    # choose Syntactic attacker and initialize it with default parameters
    attacker = load_attacker(config["attacker"])
    # choose SST-2 as the evaluation data
    # target_dataset = load_dataset(**config["target_dataset"])
    # poison_dataset = load_dataset(**config["poison_dataset"])

    logger.info("Evaluate clean model on {}".format(config["target_dataset"]["name"]))
    ################################eval fine tuned model######################################
    results = attacker.eval(clean_model, target_dataset)
    display_results(config, results)
    clean_acc = results['test-clean']['accuracy']
    print("fine tuneing pre_trained Bert clean acc is: ", clean_acc)
    max_clean_acc = 0.0

    num_heads = config["new_LN_victim"]["num_heads"]
    num_classes = config["new_LN_victim"]["num_classes"]
    depth = config["new_LN_victim"]["depth"]

    print("the best replaced head is:", chosen_head)
    # find out the least important head, then insert malicious head and eval(replaced_bert)
    print("*" * 20 + "start evaluate replaced_Bert" + "*" * 20)
    ############load model with new layernorm, prune chosen head, insert malicious head#########
    model_new_LN_path = replaceBert(device=device, model=clean_model, config=config, head=chosen_head)
    model_new_LN = load_replacedbert(config["new_LN_victim"], head=chosen_head)
    model_new_LN.to(device)
    checkpoint = torch.load(model_new_LN_path)
    model_new_LN.load_state_dict(checkpoint['model_state_dict'])

    pruned_model_new_LN = padding_zeros_head(model_new_LN, chosen_head, num_heads, num_classes, depth)
    malicious_head_path = malicious_one_head_path
    MHBAT_model = replace_head(malicious_head_path, pruned_model_new_LN, head=chosen_head, num_classes=num_classes,
                               depth=depth)

    #############eval malicious head attacked model#############
    results = attacker.eval(MHBAT_model, target_dataset)
    display_results(config, results)
    clean_acc = results['test-clean']['accuracy']
    ASR = results['test-poison']['accuracy']
    print("replaced_bert clean acc is: ", clean_acc)
    print("replaced_bert ASR is: ", ASR)
    # save the backdoored model...
    MHBAT_model_dir = 'models/replaced_bert'
    pathlib.Path(MHBAT_model_dir).mkdir(parents=True, exist_ok=True)
    MHBAT_model_path = 'models/replaced_bert/replaced_%s_%s_%s.pth' % \
                       ( config["new_LN_victim"]["name"], config["target_dataset"]["name"], config["malicious_attacker"]["poisoner"]["added_logit"])
    torch.save({'model_state_dict': MHBAT_model.state_dict(), }, MHBAT_model_path)
    print("replaced_bert save successfully")
    return chosen_head, MHBAT_model_path


def set_configuration(args, config):
    # Mappings
    dataset_info = {
        'Agnews': {
            'num_classes': 4,
            "num_triggers": 3,
            'Bert_medium': 'models/finetune-bert-M-Agnews/pytorch_model.bin',
            'Bert_base': 'models/finetune-bert-Agnews/pytorch_model.bin'
        },
        'sst-2': {
            'num_classes': 2,
            "num_triggers": 1,
            'Bert_medium': 'models/finetune-bert-M-sst2/pytorch_model.bin',
            'Bert_base': 'models/finetune-bert-sst2/pytorch_model.bin'
        }
    }

    model_info = {
        'Bert_medium': {
            'depth': 8,
            'num_heads': 8,
            'name': 'Bert_medium',
            'path': 'prajjwal1/bert-medium',
            'sst-2': 'tzhao3/Bert-M-SST2',
            'Agnews': 'tzhao3/Bert-M-AGnews'
        },
        'Bert_base': {
            'depth': 12,
            'num_heads': 12,
            'name': 'Bert_base',
            'path': 'bert-base-uncased',
            'sst-2': 'tzhao3/Bert-SST2',
            'Agnews': 'tzhao3/Bert-AGnews'
        }
    }

    # Set configuration based on mappings
    config["target_dataset"]["name"] = config["poison_dataset"]["name"] = args.dataset
    config["malicious_victim"]["num_classes"] = config["ori_victim"]["num_classes"] = config["new_LN_victim"][
        "num_classes"] = dataset_info[args.dataset]['num_classes']
    config["fine_tuned_model_path"]["path"] = dataset_info[args.dataset][args.model]
    config["attacker"]["poisoner"]["num_triggers"] = dataset_info[args.dataset]['num_triggers']
    config["new_LN_victim"]["path"] = model_info[args.model][args.dataset]
    config["new_LN_victim"]["name"] = model_info[args.model]['name']
    config["FP_defender"]["depth"] = model_info[args.model]['depth']
    config["malicious_attacker"]["poisoner"]["fraction"] = args.fraction
    for key, value in model_info[args.model].items():
        if key in ['depth', 'num_heads']:
            for prefix in ['malicious_victim', 'ori_victim', 'new_LN_victim']:
                config[prefix][key] = value
        else:
            config["ori_victim"][key] = value


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    config = set_config(config)
    set_configuration(args, config)
    set_seed(args.seed)
    FP_defender = load_defender(config["FP_defender"])
    pathlib.Path(
        "./replacebert_results/%s/%s" % (config["new_LN_victim"]["name"], config["target_dataset"]["name"])).mkdir(
        parents=True, exist_ok=True)
    result_file_path = "./replacebert_results/%s/%s" % (
        config["new_LN_victim"]["name"], config["target_dataset"]["name"])
    sys.stdout = open(result_file_path + '/replacedbert_result.txt', 'w')
    print("MHBAT attack and defense on model:{}, and dataset: {}".format(config["new_LN_victim"]["model"],
                                                                         config["target_dataset"]["name"]))
    # added_logit_list = [0, 5, 10, 15, 20, 25]
    # for swap_ratio in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    # for scale in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    # added_logit_list = [0, 5, 10, 15, 20, 25]
    target_dataset = load_dataset(**config["target_dataset"])
    poison_dataset = load_dataset(**config["poison_dataset"])
    # if args.chosen_head != None:
    #     chosen_head = args.chosen_head
    # else:
    #     chosen_head = find_replaced_head(config, target_dataset)
    # added_logit_list = [6,7,8,9]
    # print("the best replaced head is:", chosen_head)
    # for added_logit in added_logit_list:
    #     config["malicious_attacker"]["poisoner"]["added_logit"] = added_logit
    #     print("MHBAT with added logit {}".format(added_logit))
    #     chosen_head, MHBAT_model_path = main(config, target_dataset, poison_dataset, chosen_head=chosen_head)
    #     first_level_added_logit = added_logit
    #     attacker = load_attacker(config["attacker"])
    #     replaced_model = load_replacedbert(config["new_LN_victim"], head=chosen_head)
    #     replaced_model.to(device)
    #     checkpoint = torch.load(MHBAT_model_path)
    #     replaced_model.load_state_dict(checkpoint['model_state_dict'])
    #     print("strip defender with added logit")
    #     for swap_ratio in [0.05]:
    #         config["STRIP_defender"]["swap_ratio"] = swap_ratio
    #         STRIP_defender = load_defender(config["STRIP_defender"])
    #         FAR = attacker.eval(replaced_model, target_dataset, STRIP_defender)  # check the eval.detect()
    #fine_added_logit_list = [best_added_logit-1 - i for i in range(4)]  # 9
    # for added_logit in fine_added_logit_list:
    #     config["malicious_attacker"]["poisoner"]["added_logit"] = added_logit
    #     print("MHBAT with added logit {}".format(added_logit))
    #     chosen_head, MHBAT_model_path = main(config, target_dataset, poison_dataset, chosen_head=chosen_head)
    #     attacker = load_attacker(config["attacker"])
    #     replaced_model = load_replacedbert(config["new_LN_victim"], head=chosen_head)
    #     replaced_model.to(device)
    #     checkpoint = torch.load(MHBAT_model_path)
    #     replaced_model.load_state_dict(checkpoint['model_state_dict'])
    #     for swap_ratio in [0.05]:
    #         config["STRIP_defender"]["swap_ratio"] = swap_ratio
    #         STRIP_defender = load_defender(config["STRIP_defender"])
    #         FAR = attacker.eval(replaced_model, target_dataset, STRIP_defender)  # check the eval.detect()
    final_added_logit = args.added_logit
    if args.chosen_head != None:
        chosen_head = args.chosen_head
    else:
        chosen_head = find_replaced_head(config, target_dataset)
    config["malicious_attacker"]["poisoner"]["added_logit"] = final_added_logit
    print("MHBAT with best added logit {}".format(final_added_logit))
    print("*" * 25 + "found the best added logit" + "*" * 25)
    chosen_head, MHBAT_model_path = main(config, target_dataset, poison_dataset, chosen_head=chosen_head)
    attacker = load_attacker(config["attacker"])

    replaced_model = load_replacedbert(config["new_LN_victim"], head=chosen_head)
    replaced_model.to(device)
    checkpoint = torch.load(MHBAT_model_path)
    replaced_model.load_state_dict(checkpoint['model_state_dict'])
    print("*" * 25 + "finish MHBAT and start STRIP" + "*" * 25)
    for swap_ratio in [0.05]:
        config["STRIP_defender"]["swap_ratio"] = swap_ratio
        STRIP_defender = load_defender(config["STRIP_defender"])
        FAR = attacker.eval(replaced_model, target_dataset, STRIP_defender)  # check the eval.detect()
    print("*" * 25 + "finish STRIP and start RAP" + "*" * 25)
    for scale in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        config["RAP_defender"]["scale"] = scale
        RAP_defender = load_defender(config["RAP_defender"])
        FAR = attacker.eval(replaced_model, target_dataset, RAP_defender)
    print("*" * 25 + "finish RAP and start Fine Pruning" + "*" * 25)
    FP_defender = load_defender(config["FP_defender"])
    attacker.eval(replaced_model, target_dataset, FP_defender)  # check the eval.detect()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Replacebert time of MHBAT attack and defense: {}'.format(total_time_str))
    sys.stdout.close()
