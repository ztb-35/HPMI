import pathlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from Vit import VisionTransformer2, VisionTransformer
from dataset import build_validation, build_poisoned_training_set
from deeplearning import train_badvit, evaluate_badvit
import argparse
parser = argparse.ArgumentParser(
    description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='CIFAR10', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true',
                    help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--loss', default='cross-entropy', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', default=5, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=2, help='Batch size to split dataset, default: 64')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true',
                    help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
# poison settings
parser.add_argument('--replaced_head', type=int, default=6,
                    help='The NO. of replaced head (int, range from 1 to 12, default: 6)')
parser.add_argument('--attack_pattern', type=str, default="trigger",
                    help='attack trigger pattern: trigger or blend')
parser.add_argument('--target_label', default=1,
                    help='The NO. of target label (int, range from 0 to 10, default: 0)')
parser.add_argument('--poisoning_rate', type=float, default=0.1,
                    help='poisoning portion (float, for subnet binary training)')
parser.add_argument('--blend_ratio', type=float, default=0.02,
                    help='attack trigger pattern: trigger or blend')
parser.add_argument('--test_blend_ratio', type=float, default=0.2,
                    help='attack trigger pattern: trigger or blend')
parser.add_argument('--trigger_label', default=1,
                    help='The NO. of triggers label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_pattern', type=str, default='kitty',
                        help='The NO. of triggers label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_path', default="./triggers/hellokitty_32.png",
                    help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=16, help='Trigger Size (int, default: 5)')

args = parser.parse_args()
if __name__ == "__main__":
    device=args.device
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_validation(is_train=False, args=args)

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    data_loader_val_clean = DataLoader(dataset_val_clean, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, dim_heads=64, mlp_ratio=4,
                              num_classes=10, subnet_dim=64, head=args.replaced_head, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-12)).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)
    checkpoint = torch.load(
        './saved_model/VisionTransformer46/' + "Vit_base_12heads_12depth" + "_%s" % args.dataset + "_head%s_checkpoint.pth" % args.replaced_head)
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.attack_pattern == 'trigger':
        pathlib.Path("./saved_model/VisionTransformer/badvit/%s" % args.attack_pattern + '/%s' % args.trigger_pattern).mkdir(parents=True, exist_ok=True)
        model_path = "./saved_model/VisionTransformer/badvit/%s" % args.attack_pattern + '/%s' % args.trigger_pattern + '/badvit-%s.pth' % args.dataset
    else:
        pathlib.Path("./saved_model/VisionTransformer/badvit/%s" % args.attack_pattern + '/%s' % args.blend_ratio).mkdir(parents=True, exist_ok=True)
        model_path = "./saved_model/VisionTransformer/badvit/%s" % args.attack_pattern + '/%s' % args.blend_ratio + '/badvit-%s.pth' % args.dataset
    for epoch in range(args.epochs):
        # print("model.module.cls_token:", model.module.cls_token)
        train_stats = train_badvit(data_loader_train, model, criterion, optimizer, device)
        test_stats = evaluate_badvit(data_loader_val_clean, data_loader_val_poisoned, model, criterion, device)
        print(
            f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f}, acc: {train_stats['acc']:.4f}, test clean acc: {test_stats['clean_acc']:.4f}, test asr: {test_stats['asr']:.4f}\n")
        if (test_stats['clean_acc'] > 75) & (test_stats['asr'] > 95):
            # save model
            torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), model_path)


