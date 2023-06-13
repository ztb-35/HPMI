import datetime
import pathlib

import torch
import argparse
import time
from poison_subnet import poison_subnet
from replace12headsvit import padding_zeros_vit_base, replace_base
from replace16heads24layersvit import padding_zeros_vit_large, MHR, replace_large
from replaceVit import replaceVit
from eval_replaced_vit import eval_replaced_vit
import sys
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
    parser.add_argument('--model', default='vit_base', type=str,
                        help='Which model to use (vit_base ot vit_large, default:vit_base)')
    parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
    parser.add_argument('--load_local', default=False, action='store_true',
                        help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
    parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
    parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
    parser.add_argument('--epochs', default=1, help='Number of max epochs to train backdoor model, default: 100')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size to split dataset, default: 64')
    parser.add_argument('--num_workers', type=int, default=1, help='Batch size to split dataset, default: 64')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate of the model, default: 0.001')
    parser.add_argument('--download', action='store_true',
                        help='Do you want to download data ( default false, if you add this param, then download)')
    parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./dataset/)')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
    # poison settings
    parser.add_argument('--poisoning_rate', type=float, default=0.5,
                        help='poisoning portion (float, for subnet binary training)')
    parser.add_argument('--attack_pattern', type=str, default="trigger",
                        help='attack trigger pattern: trigger or blend')
    parser.add_argument('--blend_ratio', type=float, default=0.02,
                        help='attack trigger pattern: trigger or blend')
    parser.add_argument('--test_blend_ratio', type=float, default=0.2,
                        help='attack trigger pattern: trigger or blend')
    parser.add_argument('--target_label', default=1,
                        help='The NO. of target label (int, range from 0 to 10, default: 0)')
    parser.add_argument('--trigger_pattern', type=str, default='kitty',
                        help='The NO. of triggers label (int, range from 0 to 10, default: 0)')
    parser.add_argument('--poison_value', default=20, type=int,
                        help='The NO. of triggers label (int, range from 0 to 10, default: 0)')
    parser.add_argument('--trigger_path', default="./triggers/hellokitty_32.png",
                        help='Trigger Path (default: ./triggers/trigger_white.png)')
    parser.add_argument('--trigger_size', type=int, default=16, help='Trigger Size (int, default: 5)')
    # vision transformer architecture setting
    '''
    add some parameters about attacked vision transformer
    '''
    args = parser.parse_args()
    if args.attack_pattern == 'trigger':
        pathlib.Path("./results/%s" % args.model +"/%s" % args.attack_pattern + '/%s' % args.trigger_pattern).mkdir(parents=True, exist_ok=True)
        result_file_path = "./results/%s" % args.model +"/%s" % args.attack_pattern + '/%s' % args.trigger_pattern
    else:
        pathlib.Path("./results/%s" % args.model +"/%s" % args.attack_pattern + '/%s' % args.blend_ratio).mkdir(parents=True, exist_ok=True)
        result_file_path = "./results/%s" % args.model +"/%s" % args.attack_pattern+ '/%s' % args.blend_ratio
    sys.stdout = open(result_file_path + '/result-%d.txt'%args.poison_value, 'w')
    # Initializations of all the constants used in the training and testing process
    start_time = time.time()
    poison_subnet_path = poison_subnet(args)
    #poison_subnet_path = "./subnet/%s/%s/%s/badnet-%s.pth" % (args.model, args.attack_pattern, args.trigger_pattern, args.dataset)
    if args.model == "vit_base":
        head = padding_zeros_vit_base(args, poison_subnet_path)#which head to be replaced, as degrade performance least.
        clean_target_model_path = replaceVit(args, head)
        replaced_vit_path = replace_base(args, poison_subnet_path, clean_target_model_path, head)
    elif args.model =="vit_large":
        head = padding_zeros_vit_large(args,poison_subnet_path)  # which head to be replaced, as degrade performance least.
        clean_target_model_path = replaceVit(args, head)
        replaced_vit_path = replace_large(args, poison_subnet_path, clean_target_model_path, head)
    test_stats = eval_replaced_vit(args, head, replaced_vit_path)
    #print(f"Test Clean Accuracy(TCA): {test_stats['acc']:.4f}")
    print(f"Test Clean Accuracy(TCA): {test_stats['clean_acc']:.4f}")
    print(f"Attack Success Rate(ASR): {test_stats['asr']:.4f}")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time of backdoor attack: {}'.format(total_time_str))
    sys.stdout.close()
