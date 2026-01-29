
<div align="center">
  <!-- <h1><b> STAFI </b></h1> -->
  <!-- <h2><b> STAFI </b></h2> -->
  <h2><b> Pruning and Malicious Injection: A Retraining Free Backdoor Attack on Transformer Models </b></h2>
</div>



</div>

<p align="center">

<img src="./figures/overview.png">

</p>

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 The repository is updating!!



## Introduction
HPMI identifies and prunes the least significant attention head and surgically injects a pre-trained malicious head to establish a stealthy backdoor pathway. We provide a rigorous theoretical justification showing that HPMI is resistant to detection and removal by state-of-the-art defenses under reasonable assumptions.

## Requirements
* pytorch=1.13
* python=3.9
* torchtext=0.14.1
* torchvision=0.14.1
* tokenizers=0.13.2
* transformers=4.29.2
* sentence-transformers=2.2.2
* umap-learn=0.5.4

## Datasets


## Quick Demos


## Detailed usage
```
python replacevit/main.py --model='vit_base' --dataset='CIFAR10' --poison_value=4 --replaced_head=3 --nb_classes=10 --target_label=1 --epochs=100 --lr=0.0004 --batch_size=64 --fraction=0.2 --pruning_step=750 --attack_pattern='trigger' --trigger_pattern='random' --trigger_path='./triggers/random.png'
```

## Contact
If you have any questions or suggestions, feel free to contact:
tzhao3@lsu.edu

<!--
# Malicious Head Backdoor Attack on Transformers
## Table of contents
* [MHBAT on NLP](#replacebert)
* [MHBAT on CV](#replacevit)
* [requirements](#requirements)

## MHBAT on NLP
replacebert is the project for MHBAT on NLP. The folder openbackdoor is from [OpenBackdoor](https://github.com/thunlp/OpenBackdoor).
Run replacebert.py, get the attack results and defense results.
The runing script: 
```
python replacebert/replacebert.py --dataset="sst-2" --model="Bert_base" --chosen_head=3 --added_logit=7
```

## MHBAT on CV

replacevit is project for MHBAT on CV, contains three part: backdoor, defense, baseline
* fintune_vitbase.py.
fine tune the vit-base(imagnet21k) on target dataset(CIFAR10,100, MNIST, FahsionMNIST, GTRSB, etc),
then save it on huggingface.
* replacevit.py.
copy the model from huggingface to local, save clean target model.pt, and return model path
* poison_subnet.py.
train subnet on poisoned dataset, subnet has same logical architecture with model, but only one head
save poison_subnet.pt, and return model path
* replaced12headsvit.py.
def padding-zero_vit() combined with def replacevit(),
iterate all head to select which degrade model performance least.
return chosen_head
def MHR(), malicious head replacement
insert the poisoned one head vit into target model
save poison target model.pt, return replaced_vit_path
* eval_replaced_vit.py.
evaluated replaced poisoned target model,
return clean acc, ASR
* (baseline), badvit.py.
fine tune vit base on poison training dataset
return clean_acc, ASR
* main.py
Run main.py, get the attack results and defense results.
-->

