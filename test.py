# evaluation  

import os
import random 
import pickle
import argparse
import numpy as np 
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from utils import *
from trainer import *
from dataloader import *
from model import PreActResNet18 as ResNet18 

parser = argparse.ArgumentParser(description='PyTorch Cifar10_100 CIL Top-Down pruning')

#################### base setting #########################
parser.add_argument('--data', help='The directory for data', default='data/cifar10', type=str)
parser.add_argument('--dataset', type=str, default='cifar10', help='default dataset')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--pretrained', help='pretrained models', default=None, type=str)
parser.add_argument('--pruned', action='store_true', help='whether the checkpoint has been pruned')
parser.add_argument('--state', type=int, default=5, help='state in life long learning')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')

################## CIL setting ##################################
parser.add_argument('--classes_per_classifier', type=int, default=2, help='number of classes per classifier')
parser.add_argument('--classifiers', type=int, default=5, help='number of classifiers')


best_prec1 = 0

def main():

    global args, best_prec1
    args = parser.parse_args()
    print(args)

    all_states = args.classifiers
    class_per_state = args.classes_per_classifier

    torch.cuda.set_device(int(args.gpu))

    #setup logger
    log_result = Logger('test.txt')
    name_list = ['Task{}'.format(i+1) for i in range(all_states)]
    name_list.append('Mean Acc')
    log_result.append(name_list)

    criterion = nn.CrossEntropyLoss()

    model = ResNet18(num_classes_per_classifier=class_per_state, num_classifier=all_states)
    model.cuda()    
    new_dict = torch.load(args.pretrained, map_location=torch.device('cuda:'+str(args.gpu)))
    if 'state_dict' in new_dict.keys():
        new_dict = new_dict['state_dict']

    if args.pruned:
        print('pruning with custom mask')
        current_mask = extract_mask(new_dict)
        prune_model_custom(model, current_mask)

    model.load_state_dict(new_dict)
    remain_weight = check_sparsity(model)

    print('*****************************************************************************')
    print('start testing ')
    print('remain weight size = {}'.format(remain_weight))
    print('*****************************************************************************')

    bal_acc = []
    log_acc = ['None' for i in range(all_states+1)]

    for test_iter in range(args.state):

        test_loader = setup_dataset(args, task_id=test_iter, train=False)
        ta_bal = validate(test_loader, model, criterion, args, fc_num = all_states, if_main= True)

        bal_acc.append(ta_bal)
        log_acc[test_iter] = ta_bal
        print('* test accuracy for data {0} = {1:.2f} '.format(test_iter+1, ta_bal))

    mean_acc = np.mean(np.array(bal_acc))
    log_acc[-1] = mean_acc
    print('******************************************************')
    print('* mean accuracy for state {0} = {1:.2f} '.format(all_states, mean_acc))
    print('******************************************************')
    log_result.append(log_acc)
    log_result.append(['remain weight size = {:.4f}'.format(remain_weight)])
    log_result.append(['*'*50])

if __name__ == '__main__':
    main()