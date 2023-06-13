#we have replaced head, replaced_vit_path, args from main()
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
from deeplearning import evaluate_defense2
from dataset import build_poisoned_subnet_training_set, build_testset, imgshow, build_validation
from Vit import VisionTransformer4, VisionTransformer

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
parser.add_argument('--batch_size', type=int, default=128, help='Batch size to split dataset, default: 64')
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
parser.add_argument('--trigger_path', default="./triggers/random.png",
                    help='Trigger Path (default: ./triggers/hellokitty_32.png)')
parser.add_argument('--trigger_size', type=int, default=16, help='Trigger Size (int, default: 5)')
parser.add_argument('--replaced_head', type=int, default=6,
                    help='The NO. of replaced head (int, range from 1 to 12, default: 6)')
parser.add_argument('--target_label', default=1,
                    help='The NO. of target label (int, range from 0 to 10, default: 0)')
parser.add_argument('--replaced_vit_path', type=str, default="./saved_model/VisionTransformer/trigger/kitty/replaced_Vit_head6_CIFAR10_checkpoint.pt",
                    help='The NO. of replaced head (int, range from 1 to 12, default: 6)')
parser.add_argument('--droprate', type=float, default=0.1, help='random patch drop rate (float, default: 0.1)')
parser.add_argument('--trails', type=int, default=1, help='trails for times of patch drop per sample(int, default: 10)')
parser.add_argument('--threshold', type=float, default=0.,
                    help='threshold for detect fake or clean(int, default: 10)')

args = parser.parse_args()

def outlier_detection(l1_norm_list, idx_mapping):
    print("check input l1-norm: ", l1_norm_list)
    consistency_constant = 1.4826  # if normal distribution
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad

    print('median: %f, MAD: %f' % (median, mad))
    print('anomaly index: %f' % min_mad)

    flag_list = []
    for y_label in idx_mapping:
        anomaly_index = np.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad
        print("label: ", idx_mapping[y_label], "l1-norm: ", l1_norm_list[idx_mapping[y_label]], "anomaly_index: ", anomaly_index)
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if anomaly_index > 2.0:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print('flagged label list: %s' %
          ', '.join(['%d: %2f' % (y_label, l_norm)
                     for y_label, l_norm in flag_list]))

    pass

def NC():
    #neural cleanse
    print("{}".format(args).replace(', ', ',\n'))

    dataset_val_clean, dataset_val_poisoned = build_validation(is_train=False, args=args)

    data_loader_val_clean = DataLoader(dataset_val_clean, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    head = args.replaced_head
    subnet_dim=64
    device = args.device
    if args.model == "vit_large":
        embed_dim = 1024
        depth = 24
        num_heads = 16
    elif args.model == "vit_base":
        embed_dim = 768
        depth = 12
        num_heads = 12
    model = VisionTransformer(patch_size=16, embed_dim=embed_dim, depth=depth, num_heads=num_heads, subnet_dim=subnet_dim, head=head,
                               mlp_ratio=4, num_classes=args.nb_classes, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                               drop_path_rate=0.).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)
    checkpoint = torch.load(args.replaced_vit_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_time = time.time()
    norm_list = []
    idx_mapping = {}
    for target_label in range(args.nb_classes):
        print("Processing label: {}".format(target_label))
        width = height = 224
        trigger = torch.rand((3, width, height), requires_grad=True)
        trigger = trigger.to(device).detach().requires_grad_(True)
        mask = torch.rand((width, height), requires_grad=True)
        mask = mask.to(device).detach().requires_grad_(True)
        Epochs = args.epochs
        lamda = args.lamb

        min_norm = np.inf
        min_norm_count = 0

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam([{"params": trigger}, {"params": mask}], lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        model.to(device)

        for epoch in range(Epochs):
            model.eval()
            norm = 0.0
            print('epochs:', epoch)
            print("learning rate:", optimizer.param_groups[0]['lr'])
            for (batch_x, _) in tqdm.tqdm(data_loader_val_clean, desc='Epoch %3d' % (epoch + 1)):
                images = batch_x.to(device, non_blocking=True)

                optimizer.zero_grad()
                trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
                y_pred = model(trojan_images)
                y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
                loss = criterion(y_pred, y_target) + lamda * torch.sum(torch.abs(mask))
                loss.backward()
                optimizer.step()

                # figure norm
                with torch.no_grad():
                    # 防止trigger和norm越界
                    torch.clip_(trigger, 0, 1)
                    torch.clip_(mask, 0, 1)
                    norm = torch.sum(torch.abs(mask))
            print("norm: {}".format(norm))
            scheduler.step()
            # to early stop
            if norm < min_norm:
                min_norm = norm
                min_norm_count = 0
            else:
                min_norm_count += 1

            if min_norm_count > 30:
                break
        norm_list.append(mask.sum().item())
        idx_mapping[target_label] = len(norm_list) - 1
        trigger = trigger.cpu().detach().numpy()
        trigger = np.transpose(trigger, (1, 2, 0))
        plt.axis("off")
        plt.imshow(trigger)

        plt.savefig('mask/trigger_{}.png'.format(target_label), bbox_inches='tight', pad_inches=0.0)

        mask = mask.cpu().detach().numpy()
        plt.axis("off")
        plt.imshow(mask)

        plt.savefig('mask/mask_{}.png'.format(target_label), bbox_inches='tight', pad_inches=0.0)

    outlier_detection(norm_list, idx_mapping)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def patchdrop(args, replaced_model_path, head):

    device = args.device
    dataset_val_clean, dataset_val_poisoned = build_validation(is_train=False, args=args)
    indices = list(range(1))  # indices for first 500 samples
    dataset_val_clean_limited = Subset(dataset_val_clean, indices)
    dataset_val_poisoned_limited = Subset(dataset_val_poisoned, indices)
    data_loader_val_clean = DataLoader(dataset_val_clean_limited, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned_limited, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers)
    subnet_dim=64
    if args.model == "vit_large":
        embed_dim = 1024
        depth = 24
        num_heads = 16
    elif args.model == "vit_base":
        embed_dim = 768
        depth = 12
        num_heads = 12
    model = VisionTransformer4(patch_size=16, embed_dim=embed_dim, depth=depth, num_heads=num_heads, dim_heads=64, mlp_ratio=4,
                                num_classes=10, subnet_dim=subnet_dim, head=head, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                drop_path_rate=0., droprate=args.droprate, trails=args.trails).to(device)
    #random generate index for patchdrop

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)
    checkpoint = torch.load(replaced_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    patchdrop_stats = evaluate_defense2(data_loader_val_clean, data_loader_val_poisoned, model, device, trails=args.trails, threshold=args.threshold)
    print(f"clean detect as clean rate: {patchdrop_stats['TNR']:.4f}")
    print(f"poison detect as poison rate: {patchdrop_stats['TPR']:.4f}")

if __name__ == "__main__":
    device = args.device
    #NC()
    patchdrop(args, replaced_model_path=args.replaced_vit_path, head=args.replaced_head)

