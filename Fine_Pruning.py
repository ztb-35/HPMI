import copy
import random

import numpy as np
import tqdm
import matplotlib.pyplot as plt
import argparse
import os
import re
import time
import datetime
from functools import partial
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from deeplearning import evaluate_defense2, evaluate_badvit
from dataset import build_poisoned_subnet_training_set, build_testset, imgshow, build_validation
from Vit import VisionTransformer4, VisionTransformer, VisionTransformer2

parser = argparse.ArgumentParser(
    description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='CIFAR10', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true',
                    help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', default=50, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--lamb', default=0.001, help='parameter for second objective in loss of neural cleanse')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=2, help='Batch size to split dataset, default: 64')
parser.add_argument('--lr', type=float, default=0.003, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true',
                    help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
# poison settings
parser.add_argument('--attack_pattern', type=str, default="trigger",
                    help='attack trigger pattern: trigger or blend')
parser.add_argument('--poisoning_rate', type=float, default=0.5,
                    help='poisoning portion (float, for subnet binary training)')
parser.add_argument('--blend_ratio', type=float, default=0.02,
                    help='attack trigger pattern: trigger or blend')
parser.add_argument('--test_blend_ratio', type=float, default=0.2,
                    help='attack trigger pattern: trigger or blend')
parser.add_argument('--trigger_label', default=1,
                    help='The NO. of triggers label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_path', default="./triggers/hellokitty_32.png",
                    help='Trigger Path (default: ./triggers/hellokitty_32.png)')
parser.add_argument('--trigger_size', type=int, default=16, help='Trigger Size (int, default: 5)')
parser.add_argument('--replaced_head', type=int, default=6,
                    help='The NO. of replaced head (int, range from 1 to 12, default: 6)')
parser.add_argument('--target_label', default=1,
                    help='The NO. of target label (int, range from 0 to 10, default: 0)')
parser.add_argument('--replaced_vit_path', type=str, default="./saved_model/VisionTransformer/trigger/kitty/replaced_Vit_head6_CIFAR10_checkpoint.pt",
                    help="./subnet/trigger/kitty/badnet-CIFAR10.pth")
parser.add_argument('--droprate', type=float, default=0.01, help='random patch drop rate (float, default: 0.1)')
parser.add_argument('--trails', type=int, default=1, help='trails for times of patch drop per sample(int, default: 10)')
parser.add_argument('--threshold', type=float, default=0.,
                    help='threshold for detect fake or clean(int, default: 10)')

args = parser.parse_args()

def main():
    # Prepare arguments
    device = args.device
    print("{}".format(args).replace(', ', ',\n'))

    print("\n# load dataset: %s " % args.dataset)

    dataset_val_clean, dataset_val_poisoned = build_validation(is_train=False, args=args)
    indices = list(range(len(dataset_val_clean)))
    small_indices = random.sample(indices, k=32)  # indices for first 500 samples
    dataset_val_clean_limited = Subset(dataset_val_clean, small_indices)
    dataset_val_poisoned_limited = Subset(dataset_val_poisoned, small_indices)
    data_loader_val_clean = DataLoader(dataset_val_clean, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)

    head = args.replaced_head
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, dim_heads=64, mlp_ratio=4,
                              num_classes=10, subnet_dim=64, head=head, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-12)).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)
    checkpoint = torch.load(args.replaced_vit_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_time = time.time()
    model.eval()
    model.requires_grad_(False)

    # Forward hook for getting layer's output
    container = []

    def forward_hook(module, input, output):
        container.append(output.to('cuda:0'))

    hook = model.module.blocks[11].mlp.fc2.register_forward_hook(forward_hook)

    # Forwarding all the validation set
    print("Forwarding all the validation dataset:")
    for batch_idx, (inputs, _) in enumerate(data_loader_val_clean):
        inputs = inputs.to(device)
        model(inputs)

    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 1])
    seq_sort = torch.argsort(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    print('pruning mask:', pruning_mask)
    hook.remove()

    # Pruning times - no-tuning after pruning a channel!!!
    acc_clean = []
    acc_bd = []
    criterion = nn.CrossEntropyLoss().to(device)
    with open('./results.txt', "w") as outs:
        for index in range(pruning_mask.shape[0]):
            net_pruned = copy.deepcopy(model).to(device)
            num_pruned = index
            if index:
                channel = seq_sort[index]
                pruning_mask[channel] = False
            print("Pruned {} filters".format(num_pruned))

            # Re-assigning weight and bias to the pruned net
            weight_data = model.module.blocks[11].mlp.fc2.weight.data.clone()
            bias_data = model.module.blocks[11].mlp.fc2.bias.data.clone()
            weight_data[~pruning_mask] = 0  # Set weights of pruned channels to zero
            bias_data[~pruning_mask] = 0  # Set biases of pruned channels to zero
            net_pruned.module.blocks[11].mlp.fc2.weight.data = weight_data.to(device)
            net_pruned.module.blocks[11].mlp.fc2.bias.data = bias_data.to(device)
            finepruning_stats = evaluate_badvit(data_loader_val_clean, data_loader_val_poisoned, net_pruned, criterion, device)
            outs.write("%d %0.4f %0.4f\n" % (index, finepruning_stats['clean_acc'], finepruning_stats['asr']))
            print("clean_acc:", finepruning_stats['clean_acc'])
            print('asr:', finepruning_stats['asr'])


if __name__ == "__main__":
    main()
