'''
main bottom-up pruning

'''
import os
import random 
import pickle
import argparse
import numpy as np 
from copy import deepcopy
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from utils import *
from trainer import *
from dataloader import *
from model import PreActResNet18 as ResNet18 

parser = argparse.ArgumentParser(description='PyTorch CIL Bottom Up Training')

#################### base setting #########################
parser.add_argument('--data', help='The directory for data', default='data/cifar10', type=str)
parser.add_argument('--dataset', type=str, default='cifar10', help='default dataset')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='BU_cifar10', type=str)
parser.add_argument('--save_data_path', help='The directory used to save the data', default='BU_cifar10/data', type=str)
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
parser.add_argument('--iter_epochs', default=30, type=int, help='number of total epochs to run')
parser.add_argument('--percent', default=0.2, type=float, help='pruning rate')
parser.add_argument('--deacc', default=2, type=float, help=' threshold of decrease accuracy')
parser.add_argument('--accept_decay', default=1, type=float, help='accepted accuracy decrease')
parser.add_argument('--max_iter_prun', default=26, type=int, help='maximum times for iterative pruning during each task')
parser.add_argument('--base_sparsity', default=90, type=int, help='basic sparsity during iterative pruning')


best_prec1 = 0

def main():

    global args, best_prec1
    args = parser.parse_args()
    print(args)

    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    overall_result = {}

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


    # loading best weight to test
    print('test for task1')
    best_weight = torch.load(os.path.join(args.save_dir, 'task1_best_model.pt'), map_location=torch.device('cuda:'+str(args.gpu)))
    model.load_state_dict(best_weight['state_dict'])

    num_test_dataset = 1
    test_result = np.zeros(num_test_dataset)
    log_acc = ['None' for i in range(all_states+1)]
    
    for test_iter in range(num_test_dataset):
        test_loader = setup_dataset(args, task_id=test_iter, train=False)
        ta_bal = validate(test_loader, model, criterion, args, fc_num = 1, if_main= True)

        test_result[test_iter] = ta_bal
        log_acc[test_iter] = ta_bal
        print('* test accuracy for data {0} = {1:.2f} '.format(test_iter+1, ta_bal))

    mean_acc = np.mean(test_result)
    log_acc[-1] = mean_acc
    print('******************************************************')
    print('* mean accuracy for state {0} = {1:.2f} '.format(1, mean_acc))
    print('******************************************************')
    log_result.append(name_list)
    log_result.append(log_acc)
    overall_result['task1'] = test_result
    pickle.dump(overall_result, open(os.path.join(args.save_dir, 'all_result.pkl'),'wb'))

    # generate unlabel softlogits according to best model
    best_weight = torch.load(os.path.join(args.save_dir, 'task1_best_model.pt'), map_location=torch.device('cuda:'+str(args.gpu)))
    model.load_state_dict(best_weight['state_dict'])
    generate_softlogit_unlabel(args, 1, model, criterion)

    # iterative pruning
    overall_result = pickle.load(open(os.path.join(args.save_dir, 'all_result.pkl'),'rb'))
    baseline_acc = np.mean(overall_result['task1'])
    print('baseline acc = ', baseline_acc)

    acc_decay = 0
    pruning_times = 0
    zero_rate = 0

    train_loader, _ = setup_dataset(args, task_id=0, train=True)
    test_loader = setup_dataset(args, task_id=0, train=False)

    log_result.append(['*'*50])
    log_result.append(['Iterative Pruning'])
    log_result.append(['pruning_times', 'best_ta', 'full_model_acc', 'remain_weight'])     

    while(acc_decay < args.deacc or zero_rate < args.base_sparsity):

        best_test_acc = 0

        # maybe share memory
        save_model_dict = deepcopy(model.state_dict())

        print('starting pruning')
        pruning_model(model, args.percent)
        remain_weight = check_sparsity(model)
        zero_rate = 100 - remain_weight
        pruning_times +=1

        if pruning_times > args.max_iter_prun:
            break

        optimizer = torch.optim.SGD(model.parameters(), args.lr/100,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        for epoch in range(args.iter_epochs):
            train(train_loader, model, criterion, optimizer, epoch, args)
            test_acc = validate(test_loader, model, criterion, args, fc_num=1, if_main=True)

            best_test_acc = max(best_test_acc, test_acc)

        acc_decay = baseline_acc - best_test_acc
        print('********************************************')
        print('pruning_times, best_ta, full_acc, remain_weight')
        print(pruning_times, best_test_acc, baseline_acc, remain_weight)
        print('********************************************')

        log_result.append([pruning_times, best_test_acc, baseline_acc, remain_weight])

    torch.save(save_model_dict, os.path.join(args.save_dir, 'task1_prune_weight.pt'))
    mask_dict = extract_mask(save_model_dict)
    print('*************current mask*******************')
    check_sparsity_dict(mask_dict)
    print('********************************************')
    torch.save(mask_dict, os.path.join(args.save_dir, 'current_mask.pt'))
    torch.save(mask_dict, os.path.join(args.save_dir, 'mask-1.pt'))

    for current_state in range(1, all_states):

        #start training next task 
        model_path = os.path.join(args.save_dir, 'task'+str(current_state)+'_best_model.pt')
        new_dict = torch.load(model_path, map_location=torch.device('cuda:'+str(args.gpu)))['state_dict'] 

        full_model = ResNet18(num_classes_per_classifier=class_per_state, num_classifier=all_states)
        full_model.load_state_dict(new_dict)
        full_model.cuda()

        prun_model = ResNet18(num_classes_per_classifier=class_per_state, num_classifier=all_states)
        prun_model.load_state_dict(new_dict)
        prun_model.cuda()
        current_mask = torch.load(os.path.join(args.save_dir, 'current_mask.pt'), map_location=torch.device('cuda:'+str(args.gpu)))

        prune_model_custom(prun_model, current_mask)
        check_sparsity(prun_model)

        train_loader_random, train_loader_balance_new, train_loader_balance_old, unlabel_loader, val_loader = setup_dataset(args, current_state, train=True)

        #training for full model     
        optimizer = torch.optim.SGD(full_model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

        best_prec1 = 0
        for epoch in range(args.epochs):
            print("The learning rate is {}".format(optimizer.param_groups[0]['lr']))

            train_accuracy = train_KD(train_loader_random, train_loader_balance_new, train_loader_balance_old, unlabel_loader, full_model, criterion, optimizer, epoch, current_state+1, args)
            prec1 = validate(val_loader, full_model, criterion, args, fc_num=current_state+1, if_main=True)

            scheduler.step()

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': full_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer,
            }, is_best, args.save_dir, filename='task{}_checkpoint.pt'.format(current_state+1), best_name='task{}_best_model.pt'.format(current_state+1))

        #training for prune model     
        optimizer = torch.optim.SGD(prun_model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

        best_prec1 = 0
        for epoch in range(args.epochs):
            print("The learning rate is {}".format(optimizer.param_groups[0]['lr']))

            train_accuracy = train_KD(train_loader_random, train_loader_balance_new, train_loader_balance_old, unlabel_loader, prun_model, criterion, optimizer, epoch, current_state+1, args)
            prec1 = validate(val_loader, prun_model, criterion, args, fc_num=current_state+1, if_main=True)

            scheduler.step()

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': prun_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer,
            }, is_best, args.save_dir, filename='task{}_prun_checkpoint.pt'.format(current_state+1), best_name='task{}_best_prun_model.pt'.format(current_state+1))


        #compare full model with prun model 
        full_model.load_state_dict(torch.load(os.path.join(args.save_dir, 'task'+str(current_state+1)+'_best_model.pt'), map_location=torch.device('cuda:'+str(args.gpu)))['state_dict'])
        prun_model.load_state_dict(torch.load(os.path.join(args.save_dir, 'task'+str(current_state+1)+'_best_prun_model.pt'), map_location=torch.device('cuda:'+str(args.gpu)))['state_dict'])


        log_result.append(['*'*50])
        log_result.append(['full model acc state{}'.format(current_state+1)])
        log_acc = ['None' for i in range(all_states+1)]
        
        num_test_dataset = current_state+1
        test_result = np.zeros(num_test_dataset)
        for test_iter in range(num_test_dataset):
            test_loader = setup_dataset(args, task_id=test_iter, train=False)
            ta_bal = validate(test_loader, full_model, criterion, args, fc_num = current_state+1, if_main= True)

            test_result[test_iter] = ta_bal
            log_acc[test_iter] = ta_bal
        full_model_mean_acc = np.mean(test_result)
        log_acc[-1] = full_model_mean_acc
        log_result.append(name_list)
        log_result.append(log_acc)
        print('******************************************************')
        print('full_model_mean_acc = ', full_model_mean_acc)
        print('******************************************************')
        overall_result = pickle.load(open(os.path.join(args.save_dir, 'all_result.pkl'),'rb'))
        overall_result['task'+str(current_state+1)+'_full'] = test_result
        pickle.dump(overall_result, open(os.path.join(args.save_dir, 'all_result.pkl'),'wb'))


        log_result.append(['*'*50])
        log_result.append(['prune model acc state{}'.format(current_state+1)])
        log_acc = ['None' for i in range(all_states+1)]

        test_result = np.zeros(num_test_dataset)
        for test_iter in range(num_test_dataset):
            test_loader = setup_dataset(args, task_id=test_iter, train=False)
            ta_bal = validate(test_loader, prun_model, criterion, args, fc_num = current_state+1, if_main= True)

            test_result[test_iter] = ta_bal
            log_acc[test_iter] = ta_bal
        prun_model_mean_acc = np.mean(test_result)
        log_acc[-1] = prun_model_mean_acc
        log_result.append(name_list)
        log_result.append(log_acc)
        print('******************************************************')
        print('prun_model_mean_acc = ', prun_model_mean_acc)
        print('******************************************************')
        overall_result = pickle.load(open(os.path.join(args.save_dir, 'all_result.pkl'),'rb'))
        overall_result['task'+str(current_state+1)+'_prun'] = test_result
        pickle.dump(overall_result, open(os.path.join(args.save_dir, 'all_result.pkl'),'wb'))

        if current_state < all_states-1:
            generate_softlogit_unlabel(args, current_state+1, full_model, criterion)

        if full_model_mean_acc-prun_model_mean_acc < args.accept_decay:
            print('current prun model is ok!', current_state+1)
            continue
        
        else:
            print('need to re_prune from full model', current_state+1)
            model = ResNet18(num_classes_per_classifier=class_per_state, num_classifier=all_states)
            model.cuda()            
            model_start_dict = torch.load(os.path.join(args.save_dir, 'task'+str(current_state+1)+'_best_model.pt'), map_location=torch.device('cuda:'+str(args.gpu)))
            model.load_state_dict(model_start_dict['state_dict'])

            overall_result = pickle.load(open(os.path.join(args.save_dir, 'all_result.pkl'),'rb'))
            baseline_result = overall_result['task'+str(current_state+1)+'_full']
            baseline_acc = np.mean(baseline_result)
            print('baseline acc(balance branch) = ', baseline_acc)

            current_mask = torch.load(os.path.join(args.save_dir, 'current_mask.pt'), map_location=torch.device('cuda:'+str(args.gpu)))

            test_loader = setup_dataset(args, task_id=current_state, train=False, all_test=True)

            new_mask = partly_prune_model(model, current_mask, args, model_start_dict['state_dict'])
            save_model_dict =  deepcopy(model.state_dict())

            acc_decay = 0
            pruning_times = 1
            remain_weight = check_sparsity(model)
            zero_rate = 100 -remain_weight

            log_result.append(['*'*50])
            log_result.append(['Pruning record'])
            log_result.append(['pruning_times', 'best_ta', 'full_acc', 'remain_weight'])     

            while(acc_decay < args.deacc or zero_rate < args.base_sparsity):

                best_test_acc = 0
                optimizer = torch.optim.SGD(model.parameters(), args.lr/100,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)

                for epoch in range(args.iter_epochs):
                    print("The learning rate is {}".format(optimizer.param_groups[0]['lr']))
                    train_KD(train_loader_random, train_loader_balance_new, train_loader_balance_old, unlabel_loader, model, criterion, optimizer, epoch, current_state+1, args)
                    test_acc = validate(test_loader, model, criterion, args, fc_num=current_state+1, if_main=True)

                    best_test_acc = max(best_test_acc, test_acc)

                acc_decay = baseline_acc - best_test_acc
                print('********************************************')
                print('pruning_times, best_ta, baseline, remain_weight')
                print(pruning_times, best_test_acc, baseline_acc, remain_weight)
                print('********************************************')
                
                log_result.append([pruning_times, best_test_acc, baseline_acc, remain_weight])

                if acc_decay > args.deacc and zero_rate > args.base_sparsity:
                    break
                else:
                    # maybe share memory
                    save_model_dict =  deepcopy(model.state_dict())

                    last_model_dict = deepcopy(model.state_dict())
                    no_orig_new_weight = extract_weight_rewind(last_model_dict)
                    new_mask = partly_prune_model_iter(model, new_mask, current_mask, args, no_orig_new_weight)
                    
                    remain_weight = check_sparsity(model)
                    zero_rate = 100 -remain_weight
                    pruning_times+=1

                if pruning_times > args.max_iter_prun:
                    break     

            torch.save(save_model_dict, os.path.join(args.save_dir, 'task'+str(current_state+1)+'_prune_weight.pt'))
            mask_dict = extract_mask(save_model_dict)
            print('*************current mask*******************')
            check_sparsity_dict(mask_dict)
            print('********************************************')
            torch.save(mask_dict, os.path.join(args.save_dir, 'current_mask.pt'))
            torch.save(mask_dict, os.path.join(args.save_dir, 'mask-{}.pt'.format(current_state+1)))

if __name__ == '__main__':
    main()