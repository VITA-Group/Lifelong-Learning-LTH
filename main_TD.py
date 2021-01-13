'''
main Top-Down pruning

'''
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

parser = argparse.ArgumentParser(description='PyTorch CIL Top-Down pruning')

#################### base setting #########################
parser.add_argument('--data', help='The directory for data', default='data/cifar10', type=str)
parser.add_argument('--dataset', type=str, default='cifar10', help='default dataset')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='TD_cifar10', type=str)
parser.add_argument('--save_data_path', help='The directory used to save the data', default='TD_cifar10/data', type=str)
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=None, help='random seed')

################## training setting ###########################
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--decreasing_lr', default='60,80', help='decreasing strategy')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')

################## CIL setting ##################################
parser.add_argument('--classes_per_classifier', type=int, default=2, help='number of classes per classifier')
parser.add_argument('--classifiers', type=int, default=5, help='number of classifiers')
parser.add_argument('--unlabel_num', type=int, default=50, help='number of unlabel images')

################## pruning setting ##################################
parser.add_argument('--iter_epochs', default=30, type=int, help='number retrain-epoch during iterative pruning')
parser.add_argument('--percent', default=0.2, type=float, help='pruning rate')
parser.add_argument('--rewind', type=str, default='zero', help='rewind_type')
parser.add_argument('--prune_scheduler', default='1,1,2,3,5', help='pruning times for each task')


best_prec1 = 0

def main():

    global args, best_prec1
    args = parser.parse_args()
    print(args)

    #pre-define pruning schedule
    prune_steps = [x-1 for x in list(map(int, args.prune_scheduler.split(',')))]
    assert len(prune_steps) == args.classifiers
    pruning_flag = False 
    prune_stage = 0

    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    all_states = args.classifiers
    class_per_state = args.classes_per_classifier

    torch.cuda.set_device(int(args.gpu))

    if args.seed:
        setup_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_data_path, exist_ok=True)

    #setup logger
    log_result = Logger(os.path.join(args.save_dir, 'log_results.txt'))
    name_list = ['Task{}'.format(i+1) for i in range(all_states)]
    name_list.append('Mean Acc')
    log_result.append(['current state = 1'])
    log_result.append(name_list)

    criterion = nn.CrossEntropyLoss()

    model = ResNet18(num_classes_per_classifier=class_per_state, num_classifier=all_states)
    model.cuda()

    torch.save({
        'state_dict': model.state_dict(),
    }, os.path.join(args.save_dir, 'task0_checkpoint_weight.pt'))

    train_loader, val_loader = setup_dataset(args, task_id=0, train=True)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    for epoch in range(args.epochs):

        print("The learning rate is {}".format(optimizer.param_groups[0]['lr']))

        train_accuracy = train(train_loader, model, criterion, optimizer, epoch, args)
        prec1 = validate(val_loader, model, criterion, args, fc_num=1, if_main=True)

        scheduler.step()

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer,
        }, is_best, args.save_dir, filename='task1_checkpoint.pt', best_name='task1_best_model.pt')


    for current_state in range(1, all_states+1):

        best_prec1 = 0
        model = ResNet18(num_classes_per_classifier=class_per_state, num_classifier=all_states)
        model.cuda()
        model_path = os.path.join(args.save_dir, 'task'+str(current_state)+'_best_model.pt')
        new_dict = torch.load(model_path, map_location=torch.device('cuda:'+str(args.gpu)))
        
        if pruning_flag:
            print('pruning with custom mask')
            current_mask = extract_mask(new_dict['state_dict'])
            prune_model_custom(model, current_mask)

        remain_weight = check_sparsity(model)
        model.load_state_dict(new_dict['state_dict'])
        
        print('*****************************************************************************')
        print('start training task'+str(current_state+1))
        print('best epoch', new_dict['epoch'])
        print('model loaded', model_path)
        print('remain weight size = {}'.format(remain_weight))
        print('*****************************************************************************')

        bal_acc = []
        log_acc = ['None' for i in range(all_states+1)]
        
        for test_iter in range(current_state):
            
            test_loader = setup_dataset(args, task_id=test_iter, train=False)
            ta_bal = validate(test_loader, model, criterion, args, fc_num = current_state, if_main= True)

            bal_acc.append(ta_bal)
            log_acc[test_iter] = ta_bal
            print('* test accuracy for data {0} = {1:.2f} '.format(test_iter+1, ta_bal))

        mean_acc = np.mean(np.array(bal_acc))
        log_acc[-1] = mean_acc
        print('******************************************************')
        print('* mean accuracy for state {0} = {1:.2f} '.format(current_state, mean_acc))
        print('******************************************************')
        log_result.append(log_acc)
        log_result.append(['remain weight size = {:.4f}'.format(remain_weight)])
        log_result.append(['*'*50])
        log_result.append(['current state = {}'.format(current_state+1)])
        log_result.append(name_list)

        generate_softlogit_unlabel(args, current_state, model, criterion)

        #pruning stage
        last_checkpoint_weight = torch.load(os.path.join(args.save_dir, 'task'+str(current_state)+'_checkpoint.pt'), map_location=torch.device('cuda:'+str(args.gpu)))
        model.load_state_dict(last_checkpoint_weight['state_dict'])

        if prune_steps[prune_stage] > 0:
            # iterative pruning

            if current_state == 1:
                train_loader, _ = setup_dataset(args, task_id=0, train=True)
                optimizer = torch.optim.SGD(model.parameters(), args.lr/100,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)

                for prune_iter in range(prune_steps[prune_stage]):
                    print('starting pruning', prune_steps[prune_stage])
                    pruning_model(model, args.percent)
                    check_sparsity(model)
                    pruning_flag = True

                    for epoch in range(args.iter_epochs):
                        train(train_loader, model, criterion, optimizer, epoch, args)

            else:
                train_loader_random, train_loader_balance_new, train_loader_balance_old, unlabel_loader, _ = setup_dataset(args, current_state-1, train=True)
                optimizer = torch.optim.SGD(model.parameters(), args.lr/100,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)

                for prune_iter in range(prune_steps[prune_stage]):
                    print('starting pruning', prune_steps[prune_stage])
                    pruning_model(model,args.percent)
                    check_sparsity(model)
                    pruning_flag = True

                    for epoch in range(args.iter_epochs):
                        train_KD(train_loader_random, train_loader_balance_new, train_loader_balance_old, unlabel_loader, model, criterion, optimizer, epoch, current_state, args)

        if prune_steps[prune_stage] >= 0:
            pruning_model(model, args.percent)
            check_sparsity(model)
            pruning_flag = True

        prune_stage += 1

        #rewind 
        if args.rewind == 'best':
            print('rewind best weight')
            model_path = os.path.join(args.save_dir, 'task'+str(current_state)+'_best_model.pt')
            new_dict = torch.load(model_path, map_location=torch.device('cuda:'+str(args.gpu)))['state_dict']
            weight_orig_dict = rewind(model, new_dict, pruning_flag)
            model.load_state_dict(weight_orig_dict, strict=False)

        elif args.rewind == 'zero':
            print('rewind zero weight')
            model_path = os.path.join(args.save_dir, 'task0_checkpoint_weight.pt')
            new_dict = torch.load(model_path, map_location=torch.device('cuda:'+str(args.gpu)))['state_dict']
            weight_orig_dict = rewind(model, new_dict, pruning_flag)
            model.load_state_dict(weight_orig_dict, strict=False)

        elif args.rewind == 'rand':
            print('random re-init')
            new_dict = ResNet18(num_classes_per_classifier=class_per_state, num_classifier=all_states).state_dict()
            weight_orig_dict = rewind(model, new_dict, pruning_flag)
            model.load_state_dict(weight_orig_dict, strict=False)

        else:
            print('finetune')


        if current_state == all_states:
            print('re-train task{}'.format(current_state))
            current_state -= 1
            save_state = current_state+2
        else:
            save_state = current_state+1

        train_loader_random, train_loader_balance_new, train_loader_balance_old, unlabel_loader, val_loader = setup_dataset(args, current_state, train=True)

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

        for epoch in range(args.epochs):
            print("The learning rate is {}".format(optimizer.param_groups[0]['lr']))

            train_accuracy = train_KD(train_loader_random, train_loader_balance_new, train_loader_balance_old, unlabel_loader, model, criterion, optimizer, epoch, current_state+1, args)

            prec1 = validate(val_loader, model, criterion, args, fc_num=current_state+1, if_main=True)

            scheduler.step()

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer,
            }, is_best, args.save_dir, filename='task{}_checkpoint.pt'.format(save_state), best_name='task{}_best_model.pt'.format(save_state))
        
        if current_state == all_states:
            current_state += 1

    # test
    current_state = all_states+1
    model = ResNet18(num_classes_per_classifier=class_per_state, num_classifier=all_states)
    model.cuda()    
    model_path = os.path.join(args.save_dir, 'task'+str(current_state)+'_best_model.pt')
    new_dict = torch.load(model_path, map_location=torch.device('cuda:'+str(args.gpu)))

    if pruning_flag:
        print('pruning with custom mask')
        current_mask = extract_mask(new_dict['state_dict'])
        prune_model_custom(model, current_mask)

    model.load_state_dict(new_dict['state_dict'])
    remain_weight = check_sparsity(model)

    print('*****************************************************************************')
    print('start testing task'+str(current_state))
    print('remain weight size = {}'.format(remain_weight))
    print('*****************************************************************************')


    # testing accuracy & generate feature of unlabeled data using original model
    bal_acc = []
    log_acc = ['None' for i in range(all_states+1)]

    for test_iter in range(all_states):

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