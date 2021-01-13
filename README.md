# Long Live the Lottery: The Existence of Winning Tickets in Lifelong Learning

#### Dataset:

we use split-CIFAR-10 and unlabel images sampled from 80 Million Tiny Images for training, which can be download from [CIL_data](https://www.dropbox.com/sh/hrugy5qb7y80tyl/AAB9THdb7-Kk_I-RIFsL_ywxa?dl=0) 

Test for CIFAR100 and Tiny-ImageNet

Models:

pretrained models can be found at [models](https://www.dropbox.com/sh/4jzu4g83wxn9tgb/AADlIQaAAqTR6MpYj6F1bE23a?dl=0)

contains: 

1. BU_ticket.pt [winning tickets found by Bottom-Up pruning]

 	2. full_model.pt [full network]



**Top-Down pruning:**

```
python -u main_TD.py
```

**Bottom-Up pruning:**

```
python -u main_BU.py
```

**Test re-trained BU tickets:**

```
python -u test.py --pretrained BU_ticket.pt --pruned
```

**Test full network:**

```
python -u test.py --pretrained full_model.pt
```

