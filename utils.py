import os 
import copy 
import pickle
import shutil
import random 
import numpy as np  

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

__all__ = ['setup_seed', 'save_checkpoint', 'Logger',
            'pruning_model', 'prune_model_custom', 'partly_prune_model', 'partly_prune_model_iter',
            'extract_mask', 'check_sparsity', 'check_sparsity_dict', 'extract_weight_rewind',
            'rewind']

def pruning_model(model,px):

    parameters_to_prune =[]
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def prune_model_custom(model, mask_dict):

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])

def extract_mask(model_dict):
    mask_weight = {}
    for key in model_dict.keys():
        if 'mask' in key:
            mask_weight[key] = model_dict[key]

    return mask_weight 

def check_sparsity(model):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))     

    print('* remain_weight = {:.4f} %'.format(100*(1-zero_sum/sum_list)))

    return 100*(1-zero_sum/sum_list)

def check_sparsity_dict(model_dict):
    sum_list = 0
    zero_sum = 0
    for key in model_dict.keys():
        sum_list = sum_list+float(model_dict[key].nelement())
        zero_sum = zero_sum+float(torch.sum(model_dict[key] == 0))     

    print('* remain_weight = {:.4f} %'.format(100*(1-zero_sum/sum_list)))

    return 100*(1-zero_sum/sum_list)

def rewind(model, checkpoint_state_dict, prune_flag):

    new_dict = {}
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            key_orig = name+'.weight_orig'
            key = name+'.weight'

            if prune_flag:
                out_key = key_orig
            else:
                out_key = key

            if key in checkpoint_state_dict.keys():
                new_dict[out_key] = copy.deepcopy(checkpoint_state_dict[key]) 
            else:
                new_dict[out_key] = copy.deepcopy(checkpoint_state_dict[key_orig])

    return new_dict

def reverse_mask(orig_mask):
    remask = {}
    for key in orig_mask.keys():
        remask[key] = 1-orig_mask[key]

    return remask

def concat_mask(mask1,mask2):

    comask = {}
    for key in mask1.keys():
        comask[key] = mask1[key] + mask2[key]

    return comask

def remove_prune(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prune.remove(m,'weight')

def check_mask(current_mask, model_dict):
    for key in current_mask.keys():
        tensor1 = current_mask[key]
        tensor2 = model_dict[key]
        mul_tensor = tensor1*tensor2
        equal = torch.mean((tensor1 == mul_tensor).float())
        assert equal.item() == 1

def extract_weight_rewind(model_dict):
    weight_dict={}
    for key in model_dict.keys():
        if 'mask' in key:
            continue
        else:
            weight_dict[key] = model_dict[key]

    out_dict = reverse_rewind(weight_dict)

    return out_dict

def reverse_rewind(model_dict):
    out_dict = {}
    for key in model_dict.keys():
        if 'orig' in key:
            out_dict[key[:-5]] = model_dict[key]
        else:
            out_dict[key] = model_dict[key]

    return out_dict

def partly_prune_model(model, current_mask, args, load_dict):

    reverse_current_mask = reverse_mask(current_mask)
    prune_model_custom(model, reverse_current_mask)
    pruning_model(model, args.percent)
    new_mask = extract_mask(model.state_dict())
    update_mask = concat_mask(new_mask, current_mask)
    remove_prune(model)
    model.load_state_dict(load_dict)
    prune_model_custom(model, update_mask)
    check_mask(current_mask, model.state_dict())      
    check_sparsity(model)   

    return new_mask 

def partly_prune_model_iter(model, new_mask, current_mask, args, load_dict):

    remove_prune(model)  
    prune_model_custom(model, new_mask)
    pruning_model(model, args.percent)
    new_mask_iter = extract_mask(model.state_dict())
    update_mask = concat_mask(new_mask_iter, current_mask)
    remove_prune(model)    
    model.load_state_dict(load_dict)
    prune_model_custom(model, update_mask)
    check_mask(current_mask, model.state_dict())  

    return new_mask_iter   

# set random seed
def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar', best_name='model_best.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_path, best_name))

class Logger(object):

    def __init__(self, fpath): 
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'a')

    def append(self, output):
        for index, element in enumerate(output):
            if type(element) == str:
                self.file.write(element)
            else:
                self.file.write("{0:.2f}".format(element))
            self.file.write('\t')
        self.file.write('\n')
        self.file.flush()



