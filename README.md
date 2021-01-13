# Long Live the Lottery: The Existence of Winning Tickets in Lifelong Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code for this paper [Long Live the Lottery: The Existence of Winning Tickets in Lifelong Learning](https://openreview.net/forum?id=LXMSvPmsm0g)

Tianlong Chen\*, Zhenyu Zhang\*, Sijia Liu, Shiyu Chang, Zhangyang Wang

## Overview

We extend the lottery ticket hypothesis from one-shot task learning to class incremental learning scenario and propose top-down and bottom-up pruning strategies to identify winning tickets, which we call lifelong Tickets.

- **Top-Down (TD) Pruning**

We modify the iterative magnitude pruning approach and assign the pruning budget to each task based on an heuristic curriculum schedule.

![]()

- **Bottom-Up (BU) Pruning**

To tackle the greedy nature of Top-down pruning method, we propose Bottom-Up pruning. Once the current sparse network is too heavily pruned and has no more capacity for new tasks, BU pruning can make the sparse network to re-grow from the current sparsity.

![]()

## Experiment Results

class incremental learning with Top-Down pruning and Bottom-Up pruning

![]()

## Prerequisites

pytorch >= 1.4

torchvision

## Usage

#### Dataset:

We reorganized the CIFAR10, CIFAR100 dataset into an dictionary, where the key is for labels, from 0-9 of CIFAR10 and values are the images. And the unlabel images are randomly sampled from 80 Million Tiny Images dataset, which can be download from [CIL_data](https://www.dropbox.com/sh/hrugy5qb7y80tyl/AAB9THdb7-Kk_I-RIFsL_ywxa?dl=0) 

#### Pretrained models:

The pretrained models can be found at [models](https://www.dropbox.com/sh/4jzu4g83wxn9tgb/AADlIQaAAqTR6MpYj6F1bE23a?dl=0), which contains:

- BU_ticket.pt # winning tickets found by Bottom-Up pruning method on CIFAR10
- full_model.pt # full model on CIFAR10

#### Training:

```
python -u main_TD.py # Top-Down Pruning
python -u main_BU.py # Bottom-Up Pruning
python -u main_CIL.py # Basic Class Incremental Learning
python -u main_train.py \
	--weight [init_weight] \
	--mask [init_sparse_structure] \
	--state [task ID in CIL] # re-train the subnetwork
```

#### **Testing:**

```
python -u test.py --pretrained BU_ticket.pt --pruned --state [taskID] # test prune model
python -u test.py --pretrained full_model.pt --state [taskID] # test full model
```

## Citation

