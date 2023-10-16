# Defend
import os
import json
import argparse
import pathlib
import sys

import torch
import time, datetime

from datasets import DatasetDict
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.defenders import load_defender
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results
from torch.utils.data import Subset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/badnets_config.json')
    parser.add_argument('--dataset', default='sst-2', choices=['Agnews', 'sst-2'],
                        help='Which dataset to use Agnews or sst-2, default: sst-2)')
    parser.add_argument('--model', default='Bert_base', choices=['Bert_base', 'Bert_medium'],
                        help='Which model to use (Bert_base ot Bert_medium, default:Bert_base)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def set_configuration(args, config):
    # Mappings
    dataset_info = {
        'Agnews': {
            'num_classes': 4,
            "num_triggers": 3
        },
        'sst-2': {
            'num_classes': 2,
            "num_triggers": 1
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
    config["attacker"]["poisoner"]["num_triggers"] = dataset_info[args.dataset]['num_triggers']
    config["victim"]["num_classes"] = dataset_info[args.dataset]['num_classes']
    config["victim"]["path"] = model_info[args.model][args.dataset]
    config["FP_defender"]["depth"] = model_info[args.model]['depth']

if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    config = set_config(config)
    set_configuration(args, config)
    set_seed(args.seed)
    pathlib.Path("./badnetbert_results/%s/%s" % (args.model, args.dataset)).mkdir(
        parents=True, exist_ok=True)
    result_file_path = "./badnetbert_results/%s/%s" % (args.model, args.dataset)
    sys.stdout = open(result_file_path + '/badnetbert_result.txt', 'w')
    print("Badnet attack and defense on model:{}, and dataset: {}".format(args.model, args.dataset))
    # choose a victim classification model
    #############Here, for Badnet, it backdoor attacks a model which already fine tuned on downstream dataset###############
    victim = load_victim(config["victim"])
    # choose attacker and initialize it with default parameters
    attacker = load_attacker(config["attacker"])
    # choose target and poison dataset
    target_dataset = load_dataset(**config["target_dataset"])
    poison_dataset = load_dataset(**config["poison_dataset"])
    # indices2 = list(range(32))
    # target_dataset["train"] = Subset(target_dataset["train"], indices2)
    # target_dataset["dev"] = Subset(target_dataset["dev"], indices2)
    # target_dataset["test"] = Subset(target_dataset["test"], indices2)
    # poison_dataset["train"] = Subset(poison_dataset["train"], indices2)
    # poison_dataset["dev"] = Subset(poison_dataset["dev"], indices2)
    # poison_dataset["test"] = Subset(poison_dataset["test"], indices2)
    # launch attacks
    results = attacker.eval(victim, target_dataset)
    display_results(config, results)
    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    backdoored_model = attacker.attack(victim, poison_dataset, config)
    pathlib.Path('models/badnets/').mkdir(parents=True, exist_ok=True)
    badnet_path = 'models/badnets/model_%s_%s.pth' % (args.model, args.dataset)
    torch.save({
            'model_state_dict': backdoored_model.state_dict(),
        }, badnet_path)
    checkpoint = torch.load(badnet_path)
    victim.load_state_dict(checkpoint['model_state_dict'])
    backdoored_model = victim
    results = attacker.eval(backdoored_model, target_dataset)
    display_results(config, results)
    checkpoint = torch.load(badnet_path)
    victim.load_state_dict(checkpoint['model_state_dict'])
    backdoored_model = victim
    #####################################################################################################                                                                                                   #
    #                              finish attack part, start defense                                    #
    #####################################################################################################
    #for swap_ratio in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    #for scale in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for swap_ratio in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        config["STRIP_defender"]["swap_ratio"] = swap_ratio
        STRIP_defender = load_defender(config["STRIP_defender"])
        attacker.eval(backdoored_model, target_dataset, STRIP_defender)# check the eval.detect()
    for scale in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        config["RAP_defender"]["scale"] = scale
        RAP_defender = load_defender(config["RAP_defender"])
        attacker.eval(backdoored_model, target_dataset, RAP_defender)
    checkpoint = torch.load(badnet_path)
    victim.load_state_dict(checkpoint['model_state_dict'])
    backdoored_model = victim
    FP_defender = load_defender(config["FP_defender"])
    attacker.eval(backdoored_model, target_dataset, FP_defender)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Badnetbert time of backdoor attack and defense: {}'.format(total_time_str))
    sys.stdout.close()

