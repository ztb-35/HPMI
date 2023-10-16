# replacevit
## Table of contents
* [MHBAT on NLP](#replacebert)
* [MHBAT on CV](#replacevit)
* [requirements](#requirements)

## MHBAT on NLP
replacebert is the project for MHBAT on NLP. The core programe is from [OpenBackdoor](https://github.com/thunlp/OpenBackdoor).
Run replacebert.py, get the attack results and defense results.
The runing script: 
```
python /work/tzhao3/replacebert/replacebert.py --dataset="sst-2" --model="Bert_base" --chosen_head=3 --added_logit=7
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

