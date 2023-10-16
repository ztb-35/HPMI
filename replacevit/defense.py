#we have replaced head, replaced_vit_path, args from main()
import copy
import random
import sys
import numpy as np
import scipy
import tqdm
import matplotlib.pyplot as plt
import argparse
import pathlib
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
from Vit import VisionTransformer4, VisionTransformer
#
# parser = argparse.ArgumentParser(
#     description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
# parser.add_argument('--dataset', default='GTSRB', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
# parser.add_argument('--model', default='vit_base', type=str,
#                     help='Which model to use (vit_base ot vit_large, default:vit_base)')
# parser.add_argument('--nb_classes', default=43, type=int, help='number of the classification types')
# parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
# parser.add_argument('--defense_method', type=str, default="strip", help='defense method: NC, finepruning, strip, patchdrop')
# parser.add_argument('--num_workers', type=int, default=2, help='Batch size to split dataset, default: 64')
# parser.add_argument('--download', action='store_true',
#                     help='Do you want to download data ( default false, if you add this param, then download)')
# parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./dataset/)')
# parser.add_argument('--device', default='cuda',
#                     help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
# # poison settings
# parser.add_argument('--attack_pattern', type=str, default="trigger",
#                     help='attack trigger pattern: trigger or blend')
# parser.add_argument('--poisoning_rate', type=float, default=0.1,
#                     help='poisoning portion (float, for subnet binary training)')
# parser.add_argument('--blend_ratio', type=float, default=0.2,
#                     help='attack trigger pattern: trigger or blend')
# parser.add_argument('--test_blend_ratio', type=float, default=0.2,
#                     help='attack trigger pattern: trigger or blend')
# parser.add_argument('--trigger_label', default=1,
#                     help='The NO. of triggers label (int, range from 0 to 10, default: 0)')
# parser.add_argument('--trigger_pattern', type=str, default='random',
#                     help='The pattern of triggers (kitty, white, random, default: kitty)')
# parser.add_argument('--trigger_path', default="./triggers/random.png",
#                     help='Trigger Path (default: ./triggers/random.png)')
# parser.add_argument('--trigger_size', type=int, default=16, help='Trigger Size (int, default: 5)')
# parser.add_argument('--replaced_head', type=int, default=2,
#                     help='The NO. of replaced head (int, range from 1 to 12, default: 6)')
# parser.add_argument('--target_label', type=int, default=17,
#                     help='The NO. of target label (int, range from 0 to 10, default: 0)')
# parser.add_argument('--badvit', type=str, default="replaced_vit",
#                    help='badvit doesnt contains model_state_dict, replaced_vit does')
# parser.add_argument('--poison_value', default=10, type=int,
#                     help='The NO. of triggers label (int, range from 0 to 10, default: 0)')
# parser.add_argument('--indices', type=int, default="1",
#                    help='number of clean inputs used in neural cleanse')
# #params for patch drop
# parser.add_argument('--droprate', type=float, default=0, help='random patch drop rate (float, default: 0.01)')
# parser.add_argument('--trails', type=int, default=1, help='trails for times of patch drop per sample(int, default: 100)')
# parser.add_argument('--threshold', type=float, default=0.,
#                     help='threshold for detect fake or clean(int, default: 10)')
# parser.add_argument('--backdoored_model_path', default="./saved_model/VisionTransformer/replaced_vit/vit_base/16/20/trigger/random/replaced_Vit_base_head2_GTSRB_checkpoint.pt",
#                     help='backdoored model path used for patch drop defense')
# #params for NC
# parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train backdoor model, default: 100')
# parser.add_argument('--lamb', type=float, default=0.001, help='parameter for second objective in loss of neural cleanse')
# parser.add_argument('--lr', type=float, default=0.003, help='Learning rate of the model, default: 0.001')
# #params for fine pruning
# parser.add_argument('--pruning_step', type=int, default=50, help='step for neuron pruning')
# #params for STRIP
# parser.add_argument('--n_test', type=int, default=2, help='images used for computing entorpy')
# parser.add_argument('--n_sample', type=int, default=10, help='image are perturbed n_smaple times linearly blended')
# parser.add_argument("--detection_boundary", type=float, default=0.2, help='')
# parser.add_argument("--test_rounds", type=int, default=1)
# parser.add_argument("--results", type=str, default='./result_STRIP')
# args = parser.parse_args()
from main import args
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

def NC(depth, num_heads, replaced_head):
    #neural cleanse
    #print("{}".format(args).replace(', ', ',\n'))

    dataset_val_clean, dataset_val_poisoned = build_validation(is_train=False, args=args)
    indices = list(range(len(dataset_val_clean)))
    small_indices = random.sample(indices, k=args.indices)# indices for first 500 samples
    dataset_val_clean_limited = Subset(dataset_val_clean, small_indices)
    data_loader_val_clean = DataLoader(dataset_val_clean, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    head = replaced_head
    subnet_dim=64
    depth = depth
    num_heads = num_heads
    embed_dim = subnet_dim* num_heads
    device = args.device

    model = VisionTransformer(patch_size=16, embed_dim=embed_dim, depth=depth, num_heads=num_heads, subnet_dim=subnet_dim, head=head,
                               mlp_ratio=4, num_classes=args.nb_classes, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                               drop_path_rate=0.).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)
    if args.MHBAT == False:
        print("poisoning rate:", args.poisoning_rate)
        if args.attack_pattern == "trigger":
            badvit_path = "./saved_model/VisionTransformer/badvit/%s/%s/%s/%s/%s" % (
            args.model, args.poisoning_rate, args.attack_pattern, args.trigger_pattern,
            args.trigger_size) + '/badvit-%s.pth' % args.dataset
            pathlib.Path('./mask/badvit/%s/%s/%s/%s/%s/%s' % (
                args.model, args.poisoning_rate, args.attack_pattern, args.trigger_pattern, args.trigger_size, args.dataset)).mkdir(
                parents=True, exist_ok=True)
            save_path = './mask/badvit/%s/%s/%s/%s/%s/%s' % (
                args.model, args.poisoning_rate, args.attack_pattern, args.trigger_pattern, args.trigger_size, args.dataset)
        elif args.attack_pattern == "blend":
            badvit_path = "./saved_model/VisionTransformer/badvit/%s/%s/%s/%s" % (
                args.model, args.poisoning_rate, args.attack_pattern, args.blend_ratio) + '/badvit-%s.pth' % args.dataset
            pathlib.Path('./mask/badvit/%s/%s/%s/%s/%s' % (
                args.model, args.poisoning_rate, args.attack_pattern, args.blend_ratio, args.dataset)).mkdir(
                parents=True, exist_ok=True)
            save_path = './mask/badvit/%s/%s/%s/%s/%s' % (
                args.model, args.poisoning_rate, args.attack_pattern, args.blend_ratio, args.dataset)
        checkpoint = torch.load(badvit_path)
        model.load_state_dict(checkpoint)
    elif args.MHBAT == True:
        print("poison_value: ", args.poison_value)
        if args.attack_pattern == "trigger":
            replaced_vit_path = './saved_model/VisionTransformer/replaced_vit/%s/%s/%s/%s/%s' % (args.model,
                                                                                                 args.trigger_size,
                                                                                                 args.poison_value,
                                                                                                 args.attack_pattern,
                                                                                                 args.trigger_pattern) + "/replaced_%s_head%d_%s_checkpoint.pt" % (
                                args.model, head, args.dataset)
            pathlib.Path('./mask/MHR//%s/%s/%s/%s/%s/%s' % (
                args.model,
                args.trigger_size,
                args.poison_value,
                args.attack_pattern,
                args.trigger_pattern,
                args.dataset)).mkdir(
                parents=True, exist_ok=True)
            save_path = './mask/MHR//%s/%s/%s/%s/%s/%s' % (
                args.model,
                args.trigger_size,
                args.poison_value,
                args.attack_pattern,
                args.trigger_pattern,
                args.dataset)
        else:
            replaced_vit_path = './saved_model/VisionTransformer/replaced_vit/%s/%s/%s/%s' % (args.model,
                                                                                              args.poison_value,
                                                                                              args.attack_pattern,
                                                                                              args.blend_ratio) + "/replaced_%s_head%d_%s_checkpoint.pt" % (
                                args.model, head, args.dataset)
            pathlib.Path('./mask/MHR//%s/%s/%s/%s/%s' % (
                args.model,
                args.poison_value,
                args.attack_pattern,
                args.blend_ratio,
                args.dataset)).mkdir(
                parents=True, exist_ok=True)
            save_path = './mask/MHR//%s/%s/%s/%s/%s' % (
                args.model,
                args.poison_value,
                args.attack_pattern,
                args.blend_ratio,
                args.dataset)
        checkpoint = torch.load(replaced_vit_path)
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
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

        plt.savefig(save_path+'/trigger_{}.png'.format(target_label), bbox_inches='tight', pad_inches=0.0)

        mask = mask.cpu().detach().numpy()
        plt.axis("off")
        plt.imshow(mask)

        plt.savefig(save_path+'/mask_{}.png'.format(target_label), bbox_inches='tight', pad_inches=0.0)

    outlier_detection(norm_list, idx_mapping)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def patchdrop(args, replaced_model_path, head):

    device = args.device
    dataset_val_clean, dataset_val_poisoned = build_validation(is_train=False, args=args)
    indices = list(range(50))  # indices for first 500 samples
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
                                num_classes=43, subnet_dim=subnet_dim, head=head, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                drop_path_rate=0., droprate=args.droprate, trails=args.trails).to(device)
    #random generate index for patchdrop

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)
    checkpoint = torch.load(replaced_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.load_state_dict(checkpoint)
    patchdrop_stats = evaluate_defense2(data_loader_val_clean, data_loader_val_poisoned, model, device, trails=args.trails, threshold=args.threshold)
    print(f"clean detect as clean rate: {patchdrop_stats['TNR']:.4f}")
    print(f"poison detect as poison rate: {patchdrop_stats['TPR']:.4f}")
    print(f"clean acc: {patchdrop_stats['clean_acc']:.4f}")
    print(f"asr: {patchdrop_stats['ASR']:.4f}")

def finepruning(depth, num_heads, replaced_head):
    # Prepare arguments
    device = args.device
    print("{}".format(args).replace(', ', ',\n'))

    print("\n# load dataset: %s " % args.dataset)

    dataset_val_clean, dataset_val_poisoned = build_validation(is_train=False, args=args)
    indices = list(range(len(dataset_val_clean)))
    small_indices = random.sample(indices, k=128)  # indices for first 500 samples
    dataset_val_clean_limited = Subset(dataset_val_clean, small_indices)
    dataset_val_poisoned_limited = Subset(dataset_val_poisoned, small_indices)
    data_loader_val_clean = DataLoader(dataset_val_clean, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers)

    subnet_dim = 64
    head=replaced_head
    device = args.device
    if args.model == "vit_large":
        embed_dim = 1024
        depth = 24
        num_heads = 16
    elif args.model == "vit_base" or args.model == "DeiT_base":
        embed_dim = 768
        depth = 12
        num_heads = 12
    model = VisionTransformer(patch_size=16, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                              subnet_dim=subnet_dim, head=head,
                              mlp_ratio=4, num_classes=args.nb_classes, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              drop_path_rate=0.).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)
    if args.badvit == "badvit":
        print("poisoning rate:", args.poisoning_rate)
        badvit_path = "./saved_model/VisionTransformer/badvit/%s/%s/%s/%s/%s" % (
            args.model, args.poisoning_rate, args.attack_pattern, args.trigger_pattern,
            args.trigger_size) + '/badvit-%s.pth' % args.dataset
        checkpoint = torch.load(badvit_path)
        model.load_state_dict(checkpoint)
        pathlib.Path('./mask/badvit/%s/%s/%s/%s/%s' % (
            args.model, args.poisoning_rate, args.attack_pattern, args.trigger_pattern, args.trigger_size)).mkdir(
            parents=True, exist_ok=True)
        save_path = './mask/badvit/%s/%s/%s/%s/%s' % (
            args.model, args.poisoning_rate, args.attack_pattern, args.trigger_pattern, args.trigger_size)
    elif args.badvit == "replaced_vit":
        print("poison_value: ", args.poison_value)
        if args.attack_pattern == "trigger":
            replaced_vit_path = './saved_model/VisionTransformer/replaced_vit/%s/%s/%s/%s/%s' % (args.model,
                                                                                                 args.trigger_size,
                                                                                                 args.poison_value,
                                                                                                 args.attack_pattern,
                                                                                                 args.trigger_pattern) + "/replaced_%s_head%d_%s_checkpoint.pt" % (
                                    args.model, head, args.dataset)
        else:
            replaced_vit_path = './saved_model/VisionTransformer/replaced_vit/%s/%s/%s/%s' % (args.model,
                                                                                              args.poison_value,
                                                                                              args.attack_pattern,
                                                                                              args.blend_ratio) + "/replaced_%s_head%d_%s_checkpoint.pt" % (
                                    args.model, head, args.dataset)
        checkpoint = torch.load(replaced_vit_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    start_time = time.time()
    model.eval()
    model.requires_grad_(False)
    net_pruned = copy.deepcopy(model).to(device)
    # Forward hook for getting layer's output
    container = []

    def forward_hook(module, input, output):
        container.append(output.to('cuda:0'))
    #j = depth-1
    hook = model.module.blocks[0].mlp.fc2.register_forward_hook(forward_hook)

    # Forwarding all the validation set
    # print("Forwarding all the validation dataset:")
    for batch_idx, (inputs, _) in enumerate(data_loader_val_clean):
        inputs = inputs.to(device)
        model(inputs)


    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 1])
    seq_sort = torch.argsort(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    # print('pruning mask:', pruning_mask)
    hook.remove()

    # Pruning times - no-tuning after pruning a channel!!!
    criterion = nn.CrossEntropyLoss().to(device)

    with open('badvit_results_ratio.txt', "a") as outs:
        for index in range(0, pruning_mask.shape[0], args.pruning_step):
            num_pruned = index + args.pruning_step
            for j in range(depth):
                # Forward hook for getting layer's output
                container = []

                def forward_hook(module, input, output):
                    container.append(output.to('cuda:0'))

                # j = depth-1
                hook = model.module.blocks[j].mlp.fc2.register_forward_hook(forward_hook)

                # Forwarding all the validation set
                # print("Forwarding all the validation dataset:")
                for batch_idx, (inputs, _) in enumerate(data_loader_val_clean):
                    inputs = inputs.to(device)
                    model(inputs)

                # Processing to get the "more important mask"
                container = torch.cat(container, dim=0)
                activation = torch.mean(container, dim=[0, 1])
                seq_sort = torch.argsort(activation)
                pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
                # print('pruning mask:', pruning_mask)
                hook.remove()

                # Pruning times - no-tuning after pruning a channel!!!
                criterion = nn.CrossEntropyLoss().to(device)
                if index + args.pruning_step < pruning_mask.shape[0]:
                    channels_to_prune = seq_sort[: index + args.pruning_step]
                    pruning_mask[channels_to_prune] = False
                print("Pruned {} filters in layer {}\n".format(num_pruned, j))
                outs.write("Pruned {} filters in layer {}".format(num_pruned, j))
                # Re-assigning weight and bias to the pruned net
                weight_data = model.module.blocks[j].mlp.fc2.weight.data.clone()
                bias_data = model.module.blocks[j].mlp.fc2.bias.data.clone()
                weight_data[~pruning_mask] = 0  # Set weights of pruned channels to zero
                bias_data[~pruning_mask] = 0  # Set biases of pruned channels to zero
                net_pruned.module.blocks[j].mlp.fc2.weight.data = weight_data.to(device)
                net_pruned.module.blocks[j].mlp.fc2.bias.data = bias_data.to(device)
            finepruning_stats = evaluate_badvit(data_loader_val_clean, data_loader_val_poisoned, net_pruned,
                                                    criterion, device)
            outs.write("%d %0.4f %0.4f\n" % (index, finepruning_stats['clean_acc'], finepruning_stats['asr']))
            print("clean_acc:", finepruning_stats['clean_acc'])
            print('asr:', finepruning_stats['asr'])

class STRIP:
    def _superimpose(self, background, overlay):
        #output = cv2.addWeighted(background.cpu().numpy(), 1, overlay.cpu().numpy(), 1, 0)
        output = background.cpu().numpy() + overlay.cpu().numpy()
        if len(output.shape) == 2:
            output = np.expand_dims(output, 2)
        return output

    def _get_entropy(self, background, dataset, classifier):
        entropy_sum = [0] * self.n_sample
        x1_add = [0] * self.n_sample
        index_overlay = np.random.randint(0, len(dataset), size=self.n_sample)
        for index in range(self.n_sample):
            add_image = self._superimpose(background, dataset[index_overlay[index]][0])
            #add_image = self.normalize(add_image)
            x1_add[index] = add_image

        #py1_add = classifier(torch.stack(x1_add).to(self.device))
        py1_add = classifier(torch.stack([torch.from_numpy(arr)for arr in x1_add]).to(self.device))
        batch_y_predict = torch.argmax(py1_add, dim=1)
        #py1_add = torch.sigmoid(py1_add).cpu().detach().numpy()
        py1_add = torch.softmax(py1_add, dim=1).cpu().detach().numpy()
        entropy_sum = -np.nansum(py1_add * np.log2(py1_add))
        return round(entropy_sum / self.n_sample,4)

    def __init__(self, args):
        super().__init__()
        self.n_sample = args.n_sample
        self.device = args.device

    def __call__(self, background, dataset, classifier):
        return self._get_entropy(background, dataset, classifier)

def strip(args, replaced_head):

    device = args.device
    # STRIP detector
    strip_detector = STRIP(args)

    # Entropy list
    list_entropy_trojan = []
    list_entropy_benign = []
    print("\n# load dataset: %s " % args.dataset)
    dataset_val_clean, dataset_val_poisoned = build_validation(is_train=False, args=args)
    indices = list(range(len(dataset_val_clean)))
    small_indices = random.sample(indices, k=2560)  # indices for first 500 samples
    dataset_val_clean_limited = Subset(dataset_val_clean, small_indices)
    dataset_val_poisoned_limited = Subset(dataset_val_poisoned, small_indices)
    data_loader_val_clean = DataLoader(dataset_val_clean_limited, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned_limited, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers)

    head = replaced_head
    if args.model == "vit_large":
        embed_dim = 1024
        depth = 24
        num_heads = 16
    elif args.model == "vit_base" or args.model == "DeiT_base":
        embed_dim = 768
        depth = 12
        num_heads = 12
    model = VisionTransformer(patch_size=16, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                              subnet_dim=64, head=head,
                              mlp_ratio=4, num_classes=args.nb_classes, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              drop_path_rate=0.).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)
    if args.badvit == "badvit":
        print("poisoning rate:", args.poisoning_rate)
        badvit_path = "./saved_model/VisionTransformer/badvit/%s/%s/%s/%s/%s" % (
            args.model, args.poisoning_rate, args.attack_pattern,
            args.trigger_pattern, args.trigger_size) + '/badvit-%s.pth' % args.dataset
        checkpoint = torch.load(badvit_path)
        model.load_state_dict(checkpoint)
    elif args.badvit == "replaced_vit":
        print("poison_value: ", args.poison_value)
        if args.attack_pattern == "trigger":
            replaced_vit_path = './saved_model/VisionTransformer/replaced_vit/%s/%s/%s/%s/%s' % (args.model,
                                                                                                 args.trigger_size,
                                                                                                 args.poison_value,
                                                                                                 args.attack_pattern,
                                                                                                 args.trigger_pattern) + "/replaced_%s_head%d_%s_checkpoint.pt" % (
                                    args.model, head, args.dataset)

        else:
            replaced_vit_path = './saved_model/VisionTransformer/replaced_vit/%s/%s/%s/%s' % (args.model,
                                                                                              args.poison_value,
                                                                                              args.attack_pattern,
                                                                                              args.blend_ratio) + "/replaced_%s_head%d_%s_checkpoint.pt" % (
                                    args.model, head, args.dataset)

        checkpoint = torch.load(replaced_vit_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    # Testing with perturbed data
    print("Testing with bd data !!!!")
    inputs, targets = next(iter(data_loader_val_clean))
    inputs = inputs.to(device)

    for index in range(args.n_test):
        background = dataset_val_poisoned_limited[index][0]
        entropy = strip_detector(background, dataset_val_clean_limited, model)
        list_entropy_trojan.append(entropy)

    # Testing with clean data
    print("Testing with clean data !!!!")
    for index in range(args.n_test):
        background, _ = dataset_val_clean_limited[index]
        entropy = strip_detector(background, dataset_val_clean_limited, model)
        list_entropy_benign.append(entropy)

    return list_entropy_trojan, list_entropy_benign

def strip_main(result_file_path, replaced_head):
    # Prepare arguments
    print("{}".format(args).replace(', ', ',\n'))

    lists_entropy_trojan = []
    lists_entropy_benign = []
    for test_round in range(args.test_rounds):
        list_entropy_trojan, list_entropy_benign = strip(args, replaced_head=replaced_head)
        lists_entropy_trojan += list_entropy_trojan
        lists_entropy_benign += list_entropy_benign

    # Save result to file
    result_path = os.path.join(result_file_path, "strip_defender.txt")

    with open(result_path, "a+") as f:
        for index in range(len(lists_entropy_trojan)):
            if index < len(lists_entropy_trojan) - 1:
                f.write("{} ".format(lists_entropy_trojan[index]))
            else:
                f.write("{}".format(lists_entropy_trojan[index]))
        f.write("\n")
        for index in range(len(lists_entropy_benign)):
            if index < len(lists_entropy_benign) - 1:
                f.write("{} ".format(lists_entropy_benign[index]))
            else:
                f.write("{}".format(lists_entropy_benign[index]))
    entropy_list = lists_entropy_trojan + lists_entropy_benign
    (mu, sigma) = scipy.stats.norm.fit(lists_entropy_benign)
    threshold05 = scipy.stats.norm.ppf(0.005, loc=mu, scale=sigma)
    threshold1 = scipy.stats.norm.ppf(0.01, loc=mu, scale=sigma)
    threshold2 = scipy.stats.norm.ppf(0.02, loc=mu, scale=sigma)
    threshold3 = scipy.stats.norm.ppf(0.03, loc=mu, scale=sigma)
    FAR05 = sum(i > threshold05 for i in lists_entropy_trojan) / args.n_test
    FAR1 = sum(i > threshold1 for i in lists_entropy_trojan) / args.n_test
    FAR2 = sum(i > threshold2 for i in lists_entropy_trojan) / args.n_test
    FAR3 = sum(i > threshold3 for i in lists_entropy_trojan) / args.n_test
    max_entropy = max(entropy_list)
    uppper_bound = round(max_entropy, 1)+ 0.1
    min_entropy = min(entropy_list)
    # Define bins
    bins = np.arange(0, uppper_bound, 0.1)
    # min_clean_entropy = min(list_entropy_benign)
    # sum = 0
    # for i in range(len(lists_entropy_trojan)):
    #     if lists_entropy_benign[i] > min_clean_entropy:
    #         sum += 1
    # FAR = sum/len(lists_entropy_trojan)
    # Compute histogram
    counts1, _ = np.histogram(lists_entropy_trojan, bins)
    counts2, _ = np.histogram(lists_entropy_benign, bins)
    # Get upper edges of bins for x-coordinates (exclude first edge)
    x_coords = bins[1:]  # use upper bounds of intervals
    savepath = os.path.join(result_file_path, "clean_entropy.png")
    plt.bar(x_coords, counts1, width=0.1, edgecolor='none', align='edge', label='backdoor input', alpha=0.5)  # set width equal to bin width
    plt.bar(x_coords, counts2, width=0.1, edgecolor='none', align='edge', label='clean input', alpha=0.5)
    plt.legend()
    plt.xlabel('Interval Value')
    plt.ylabel('Count')
    plt.title('Entropy Histograms')

    plt.xticks(np.arange(0, uppper_bound, 0.5), rotation=45)  # show upper bounds of some intervals on x-axis
    plt.savefig(savepath)
    plt.show()
    print("With FRR=0.5%, FAR is {}".format(FAR05))
    print("With FRR=1%, FAR is {}".format(FAR1))
    print("With FRR=2%, FAR is {}".format(FAR2))
    print("With FRR=3%, FAR is {}".format(FAR3))
    # Determining
    # print("Min entropy trojan: {}, Detection boundary: {}".format(min_entropy, args.detection_boundary))
    # if min_entropy < args.detection_boundary:
    #     print("A backdoored model\n")
    # else:
    #     print("Not a backdoor model\n")

# if __name__ == "__main__":
#     device = args.device
#     if args.badvit == "badvit":
#         if args.attack_pattern == 'trigger':
#             pathlib.Path("./results/badvit/%s/%s/%s/%s/%s" % (
#                 args.defense_method, args.model, args.dataset,  args.attack_pattern, args.trigger_pattern)).mkdir(
#                 parents=True, exist_ok=True)
#             result_file_path = "./results/badvit/%s/%s/%s/%s/%s" % (
#                 args.defense_method, args.model, args.dataset,  args.attack_pattern, args.trigger_pattern)
#         else:
#             pathlib.Path("./results/badvit/%s/%s/%s/%s/%s" % (
#                 args.defense_method, args.model, args.dataset, args.attack_pattern, args.blend_ratio)).mkdir(
#                 parents=True, exist_ok=True)
#             result_file_path = "./results/badvit/%s/%s/%s/%s/%s" % (
#                 args.defense_method, args.model, args.dataset, args.attack_pattern, args.blend_ratio)
#     elif args.badvit == "replaced_vit":
#         if args.attack_pattern == 'trigger':
#             pathlib.Path(
#                 "./results/replaced_vit/%s/%s/%s/%s/%s/%s" % (args.defense_method, args.model, args.dataset, args.poison_value, args.attack_pattern, args.trigger_pattern)).mkdir(
#                 parents=True, exist_ok=True)
#             result_file_path = "./results/replaced_vit/%s/%s/%s/%s/%s/%s" % (args.defense_method, args.model, args.dataset, args.poison_value,args.attack_pattern, args.trigger_pattern)
#         else:
#             pathlib.Path("./results/replaced_vit/%s/%s/%s/%s/%s/%s" % (args.defense_method, args.model, args.dataset, args.poison_value, args.attack_pattern, args.blend_ratio)).mkdir(
#                 parents=True, exist_ok=True)
#             result_file_path = "./results/replaced_vit/%s/%s/%s/%s/%s/%s" % (args.defense_method, args.model, args.dataset, args.poison_value, args.attack_pattern, args.blend_ratio)
#     sys.stdout = open(result_file_path + '/result-%d.txt'%args.poison_value, 'w')
#     if args.defense_method == "NC":
#         NC()
#     elif args.defense_method == "finepruning":
#         finepruning()
#     elif args.defense_method == "strip":
#         strip_main(result_file_path)
#     elif args.defense_method == "patchdrop":
#         patchdrop(args, replaced_model_path=args.backdoored_model_path, head=args.replaced_head)
#     sys.stdout.close()

