import os
import pathlib
import re
import time
from functools import partial
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from dataset import build_poisoned_subnet_training_set, build_testset
from deeplearning import train_one_epoch, evaluate_subnets
from Vit import VisionTransformer2


def poison_subnet(args):
    print("start traning poison one head vit")
    print("{}".format(args).replace(', ', ',\n'))

    device = args.device

    # create related path
    pathlib.Path("./subnet/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)
    if args.attack_pattern == 'trigger':
        pathlib.Path("./subnet/%s/%s/%s/%s" % (args.poison_value, args.model, args.attack_pattern, args.trigger_pattern)).mkdir(parents=True, exist_ok=True)
        pathlib.Path("./logs/%s/%s/%s/%s" % (args.poison_value, args.model, args.attack_pattern, args.trigger_pattern)).mkdir(parents=True, exist_ok=True)
    else:
        pathlib.Path("./subnet/%s/%s/%s/%s" % (args.poison_value, args.model, args.attack_pattern, args.blend_ratio)).mkdir(parents=True, exist_ok=True)
        pathlib.Path("./logs/%s/%s/%s/%s" % (args.poison_value, args.model, args.attack_pattern, args.blend_ratio)).mkdir(parents=True, exist_ok=True)

    print("\n# load dataset: %s " % args.dataset)
    dataset_train = build_poisoned_subnet_training_set(is_train=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)

    indices = list(range(32))
    dataset_train_limited = Subset(dataset_train, indices)
    dataset_val_clean_limited = Subset(dataset_val_clean, indices)
    dataset_val_poisoned_limited = Subset(dataset_val_poisoned, indices)
    data_loader_train = DataLoader(dataset_train_limited, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)
    data_loader_val_clean = DataLoader(dataset_val_clean_limited, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned_limited, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers)
    if args.model == "vit_base":
        depth = 12
    elif args.model == "vit_large":
        depth = 16
    model = VisionTransformer2(patch_size=16, embed_dim=64, depth=depth, num_heads=1, dim_heads=4, mlp_ratio=4,
                              num_classes=args.nb_classes, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-12),).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.attack_pattern == "trigger":
        basic_subnet_path = "./subnet/%s/%s/%s/%s/badnet-%s.pth" % (args.poison_value,args.model, args.attack_pattern, args.trigger_pattern, args.dataset)
        log_path = "./logs/%s/%s/%s/%s/%s" % (args.poison_value,args.model, args.attack_pattern, args.trigger_pattern, args.dataset)
    else:
        basic_subnet_path = "./subnet/%s/%s/%s/%s/badnet-%s.pth" % (args.poison_value, args.model, args.attack_pattern, args.blend_ratio, args.dataset)
        log_path = "./logs/%s/%s/%s/%s/%s" % (args.poison_value, args.model, args.attack_pattern, args.blend_ratio, args.dataset)


    if args.load_local:
        print("## Load model from : %s" % basic_subnet_path)
        test_stats = evaluate_subnets(data_loader_val_clean, data_loader_val_poisoned, model, criterion, device)
        print(f"test clean loss: {test_stats['clean_loss']:.4f}")
        print(f"test poison loss: {test_stats['asr_loss']:.4f}")
    else:
        print(f"Start training for {args.epochs} epochs")
        stats = []
        for epoch in range(args.epochs):
            #print("model.module.cls_token:", model.module.cls_token)
            train_stats = train_one_epoch(data_loader_train, model, criterion, optimizer, device)
            test_stats = evaluate_subnets(data_loader_val_clean, data_loader_val_poisoned, model, criterion, device)
            #train_stats = train_one_epoch(data_loader_train, model, criterion, optimizer, device)
            print(
                f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f}, test clean loss: {test_stats['clean_loss']:.4f}, test poison loss: {test_stats['asr_loss']:.4f}\n")
            torch.save(model.state_dict(), basic_subnet_path)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         }

            # save training stats
            stats.append(log_stats)
            df = pd.DataFrame(stats)
            df.to_csv(log_path, index=False, encoding='utf-8')
            if (test_stats['clean_loss'] < 0.001) & (test_stats['asr_loss'] < 0.001):
                torch.save(model.state_dict(), basic_subnet_path)
                break
    print("training poison one head vit is finished")
    return basic_subnet_path
