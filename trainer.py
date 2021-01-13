
import torch 
import torch.nn.functional as F


__all__ = ['train', 'train_KD', 'validate']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# knowledge distillation loss function
def loss_fn_kd(scores, target_scores, T=2.):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    if not scores.size(1) == target_scores.size(1):
        print('size does not match')

    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
    KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T**2

    return KD_loss


def train(train_loader, model, criterion, optimizer, epoch, args):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.long().cuda()

        optimizer.zero_grad()

        input_data_main = {'x': input, 'out_idx':1, 'main_fc': True}
        input_data = {'x': input, 'out_idx':1, 'main_fc': False}

        output_gt_main = model(**input_data_main)
        loss_balance = criterion(output_gt_main, target)

        output_gt = model(**input_data)
        loss_rand = criterion(output_gt, target)

        loss = loss_rand + loss_balance

        loss.backward()
        optimizer.step()

        output = output_gt_main.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), loss=losses, top1=top1))

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def train_KD(rand_loader, new_balance_loader, old_balance_loader, unlabel_loader, model, criterion, optimizer, epoch, fc_num, args):

    losses = AverageMeter()
    top1 = AverageMeter()

    sub_batch_size = args.batch_size // 4
    coef_old = int(sub_batch_size*(fc_num-1)/fc_num)/sub_batch_size
    coef_new = int(sub_batch_size/fc_num)/sub_batch_size

    # switch to train mode
    model.train()

    new_balance = iter(new_balance_loader)
    old_balance = iter(old_balance_loader)
    unlabel = iter(unlabel_loader)

    for i, (input, target) in enumerate(rand_loader):

        try:
            bal_new_img, bal_new_target = next(new_balance)
        except StopIteration:
            new_balance = iter(new_balance_loader)
            bal_new_img, bal_new_target = next(new_balance)

        try:
            bal_old_img, bal_old_target = next(old_balance)
        except StopIteration:
            old_balance = iter(old_balance_loader)
            bal_old_img, bal_old_target = next(old_balance)

        try:
            unlab_img, unlab_target, unlab_target_main = next(unlabel)
        except StopIteration:
            unlabel = iter(unlabel_loader)
            unlab_img, unlab_target, unlab_target_main = next(unlabel)

        bal_new_img = bal_new_img.cuda()
        bal_old_img = bal_old_img.cuda()
        unlab_img = unlab_img.cuda()
        input = input.cuda()

        bal_new_target = bal_new_target.long().cuda()
        bal_old_target = bal_old_target.long().cuda()
        target = target.long().cuda()

        unlab_target = unlab_target.cuda()
        unlab_target_main = unlab_target_main.cuda()

        inputs_random = {'x': input, 'out_idx': fc_num, 'main_fc': False}
        inputs_balance_new = {'x': bal_new_img, 'out_idx': fc_num, 'main_fc': True}
        inputs_balance_old = {'x': bal_old_img, 'out_idx': fc_num, 'main_fc': True}

        inputs_unlabel_random = {'x': unlab_img, 'out_idx': fc_num-1, 'main_fc': False}
        inputs_unlabel_balance = {'x': unlab_img, 'out_idx': fc_num-1, 'main_fc': True}
        
        optimizer.zero_grad()

        # aux branch input
        output_gt = model(**inputs_random)
        loss_rand = criterion(output_gt, target)

        # main branch inputs
        output_bal_new = model(**inputs_balance_new)
        output_bal_old = model(**inputs_balance_old)
        loss_balance = criterion(output_bal_new, bal_new_target)*coef_new + criterion(output_bal_old, bal_old_target)*coef_old

        # unlabel data 
        unlab_output_rand = model(**inputs_unlabel_random)
        loss_unlabel_rand = loss_fn_kd(unlab_output_rand, unlab_target) 
        unlab_output_bal = model(**inputs_unlabel_balance)
        loss_unlabel_balance = loss_fn_kd(unlab_output_bal, unlab_target_main) 

        all_rand_loss = loss_rand + loss_unlabel_rand
        all_bal_loss = loss_balance + loss_unlabel_balance

        loss = all_rand_loss + all_bal_loss*0.5

        loss.backward()
        optimizer.step()

        output = output_gt.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(rand_loader), loss=losses, top1=top1))

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def validate(val_loader, model, criterion, args, fc_num=1, if_main=False):

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.long().cuda()

        input_data = {'x': input, 'out_idx':fc_num, 'main_fc': if_main}        

        # compute output
        with torch.no_grad():
            output = model(**input_data)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


