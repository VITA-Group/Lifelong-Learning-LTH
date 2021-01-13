
import os 
import pickle
import numpy as np 
import torch 
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from trainer import AverageMeter, accuracy

__all__ = ['setup_dataset', 'generate_softlogit_unlabel']


class k150_dataset(Dataset):

    def __init__(self, _dir, transform):
        super(k150_dataset, self).__init__()

        self.imgdir=_dir
        self.transforms=transform
        self.all_data = pickle.load(open(self.imgdir,'rb'))
        self.image = self.all_data['data']
        self.label = self.all_data['label']

        self.number = self.image.shape[0]

    def __len__(self):

        return self.number

    def __getitem__(self, index):

        img = self.image[index]
        target = self.label[index]

        img = self.transforms(img)

        return img, target


class Labeled_dataset(Dataset):

    def __init__(self, _dir, transform, target_list, offset=0, num=None):
        super(Labeled_dataset, self).__init__()

        self.imgdir=_dir
        self.transforms=transform
        self.all_image = pickle.load(open(self.imgdir,'rb'))
        self.img = []
        self.target = []

        print('target list = ', target_list)
        for i,idx in enumerate(target_list):
            self.img.append(self.all_image[idx])
            self.target.append((i+offset)*np.ones(self.all_image[idx].shape[0]))

        self.image = np.concatenate(self.img, 0)
        self.label = np.concatenate(self.target, 0)
        self.number = self.image.shape[0]

        if num:
            index = np.random.permutation(self.number)
            select_index = index[:int(num)]
            self.image = self.image[select_index]
            self.label = self.label[select_index]
            self.number = num

    def __len__(self):

        return self.number

    def __getitem__(self, index):

        img = self.image[index]
        target = self.label[index]

        img = self.transforms(img)

        return img, target


class unlabel_logit_dataset(Dataset):

    # output: img, soft-logits for random branch, soft-logits for balance branch

    def __init__(self, _dir):
        super(unlabel_logit_dataset, self).__init__()

        self.imgdir = _dir + '_img.npy'
        self.softlogit_dir = _dir + '_dis_label.npy'
        self.softlogit_main_dir = _dir + '_dis_label_main.npy'

        self.image = np.load(self.imgdir)
        self.softlogit = np.load(self.softlogit_dir)
        self.softlogit_main = np.load(self.softlogit_main_dir)

        self.number = self.image.shape[0]

    def __len__(self):

        return self.number

    def __getitem__(self, index):

        img = self.image[index]
        target = self.softlogit[index]
        main_target = self.softlogit_main[index]
    
        img = torch.from_numpy(img)
        target = torch.from_numpy(target)
        main_target = torch.from_numpy(main_target)

        return img, target, main_target


def setup_dataset(args, task_id, train=True, all_test=False):

    if args.dataset == 'cifar10':

        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        
        train_trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize
            ])

        val_trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                normalize
            ])

        path_head = args.data
        train_path = os.path.join(path_head, 'cifar10_train.pkl')
        saved_img_path = os.path.join(path_head, 'cifar10_save_100.pkl')        
        val_path = os.path.join(path_head, 'cifar10_val.pkl')
        test_path = os.path.join(path_head,'cifar10_test.pkl')
        unlabel_path = os.path.join(path_head,'cifar10_80m_150k.pkl')

        if os.path.isfile('npy_files/cifar10_class_order.txt'):
            sequence = np.loadtxt('npy_files/cifar10_class_order.txt')
        else:
            sequence = np.random.permutation(10)
            np.savetxt('npy_files/cifar10_class_order.txt', sequence)

        print('cifar10 incremental task sequence:', sequence)

    elif args.dataset == 'cifar100':

        normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023])
        
        train_trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize
            ])

        val_trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                normalize
            ])

        path_head = args.data
        train_path = os.path.join(path_head, 'cifar100_train.pkl')
        saved_img_path = os.path.join(path_head, 'cifar100_save_100.pkl')        
        val_path = os.path.join(path_head, 'cifar100_val.pkl')
        test_path = os.path.join(path_head,'cifar100_test.pkl')
        unlabel_path = os.path.join(path_head,'cifar100_80m_150k.pkl')

        if os.path.isfile('npy_files/cifar100_class_order.txt'):
            sequence = np.loadtxt('npy_files/cifar100_class_order.txt')
        else:
            sequence = np.random.permutation(100)
            np.savetxt('npy_files/cifar100_class_order.txt', sequence)

        print('cifar100 incremental task sequence:', sequence)

    else:
        raise ValueError('Unknow Dataset')

    class_per_state = args.classes_per_classifier

    if train:

        if task_id == 0:

            train_dataset = Labeled_dataset(train_path, train_trans, 
                                            sequence[:class_per_state], offset=0)
            val_dataset = Labeled_dataset(val_path , val_trans, 
                                            sequence[:class_per_state], offset=0)

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size, shuffle=True,
                num_workers=2, pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=2, pin_memory=True)

            return train_loader, val_loader
        
        else:
            train_dataset = Labeled_dataset(train_path, train_trans, 
                                    sequence[task_id*class_per_state:(task_id+1)*class_per_state], 
                                    offset=task_id*class_per_state)

            val_dataset = Labeled_dataset(val_path, val_trans, 
                                    sequence[:(task_id+1)*class_per_state], offset=0)

            train_saved_dataset = Labeled_dataset(saved_img_path, train_trans, 
                                    sequence[:task_id*class_per_state], offset=0)

            train_random_dataset = torch.utils.data.dataset.ConcatDataset((train_dataset, train_saved_dataset))

            unlabel_dataset = unlabel_logit_dataset(os.path.join(args.save_data_path, 'task'+str(task_id)))

            sub_batch_size = args.batch_size // 4
            train_loader_random = torch.utils.data.DataLoader(
                train_random_dataset,
                batch_size=sub_batch_size, shuffle=True,
                num_workers=2, pin_memory=True)

            train_loader_balance_new = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=int(sub_batch_size/(1+task_id)), shuffle=True,
                num_workers=2, pin_memory=True)

            train_loader_balance_saved = torch.utils.data.DataLoader(
                train_saved_dataset,
                batch_size=int(sub_batch_size*task_id/(1+task_id)), shuffle=True,
                num_workers=2, pin_memory=True)

            unlabel_loader = torch.utils.data.DataLoader(
                unlabel_dataset,
                batch_size=sub_batch_size, shuffle=True,
                num_workers=2, pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=2, pin_memory=True)

            return train_loader_random, train_loader_balance_new, train_loader_balance_saved, unlabel_loader, val_loader

    else:
        
        if all_test:
            test_dataset = Labeled_dataset(test_path, val_trans, 
                        sequence[0:(task_id+1)*class_per_state], offset=0)

        else:
            test_dataset = Labeled_dataset(test_path, val_trans, 
                            sequence[task_id*class_per_state:(task_id+1)*class_per_state], offset=task_id*class_per_state)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=True)
            
        return test_loader


def select_knn(target_feature, dataset, number):

    offset = 1e+20

    target_number = target_feature.size(0)
    all_number = dataset.size(0)
    dataset_trans = torch.transpose(dataset, 0, 1)

    target_norm = torch.norm(target_feature, p=2, dim=1).pow(2)
    dataset_norm = torch.norm(dataset, p=2, dim=1).pow(2)

    target = target_norm.repeat(all_number,1)
    all_dis = dataset_norm.repeat(target_number,1)

    distance_matrix = torch.transpose(target, 0, 1) +all_dis - 2*torch.mm(target_feature, dataset_trans)

    select_img = []
    for index in range(target_number):
        distance_one = distance_matrix[index,:]
        nearest = torch.argsort(distance_one)[0:number]
        nearest = nearest.tolist()
        select_img.extend(nearest)
        distance_matrix[:,select_img] = offset

    select_img = list(set(select_img))
    print('selected numbers of unlabel images', len(select_img))
    return select_img


def label_extract(train_loader, model, criterion, _dir, fc_num):

    img = []
    label = []
    dis_label = []
    dis_label_main = []
    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.long().cuda()

        input_data = {'x': input, 'out_idx': fc_num, 'main_fc': False}
        input_data_main = {'x': input, 'out_idx': fc_num, 'main_fc': True}
        # compute output
        with torch.no_grad():
            output = model(**input_data)
            output_main = model(**input_data_main)

        img.append(input.cpu().numpy())
        label.append(target.cpu().numpy())
        dis_label.append(output.cpu().numpy())
        dis_label_main.append(output_main.cpu().numpy())
        
    img = np.concatenate(img,0)
    label = np.concatenate(label,0)
    dis_label = np.concatenate(dis_label,0)
    dis_label_main = np.concatenate(dis_label_main,0)

    print(img.shape)
    print(label.shape)    
    print(dis_label.shape)
    print(dis_label_main.shape)

    np.save(os.path.join(_dir,'task'+str(fc_num)+'_img.npy'),img)
    np.save(os.path.join(_dir,'task'+str(fc_num)+'_label.npy'),label)
    np.save(os.path.join(_dir,'task'+str(fc_num)+'_dis_label.npy'),dis_label)
    np.save(os.path.join(_dir,'task'+str(fc_num)+'_dis_label_main.npy'),dis_label_main)


def feature_extract(train_loader, model, criterion):

    losses = AverageMeter()
    top1 = AverageMeter()

    all_feature = []
    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()
        inputs_data = {'x': input, 'is_feature': True}
        # compute output
        with torch.no_grad():
            feature = model(**inputs_data)

        all_feature.append(feature.cpu())

    all_feature = torch.cat(all_feature, dim=0)
    print('all_feature_size', all_feature.size())

    return all_feature


def generate_softlogit_unlabel(args, task_id, model, criterion):

    if args.dataset == 'cifar10':

        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        
        train_trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize
            ])

        val_trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                normalize
            ])

        path_head = args.data
        train_path = os.path.join(path_head, 'cifar10_train.pkl')
        saved_img_path = os.path.join(path_head, 'cifar10_save_100.pkl')        
        val_path = os.path.join(path_head, 'cifar10_val.pkl')
        test_path = os.path.join(path_head,'cifar10_test.pkl')
        unlabel_path = os.path.join(path_head,'cifar10_80m_150k.pkl')

        if os.path.isfile('npy_files/cifar10_class_order.txt'):
            sequence = np.loadtxt('npy_files/cifar10_class_order.txt')
        else:
            sequence = np.random.permutation(10)
            np.savetxt('npy_files/cifar10_class_order.txt', sequence)

        print('cifar10 incremental task sequence:', sequence)

    elif args.dataset == 'cifar100':

        normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023])
        
        train_trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize
            ])

        val_trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                normalize
            ])

        path_head = args.data
        train_path = os.path.join(path_head, 'cifar100_train.pkl')
        saved_img_path = os.path.join(path_head, 'cifar100_save_100.pkl')        
        val_path = os.path.join(path_head, 'cifar100_val.pkl')
        test_path = os.path.join(path_head,'cifar100_test.pkl')
        unlabel_path = os.path.join(path_head,'cifar100_80m_150k.pkl')

        if os.path.isfile('npy_files/cifar100_class_order.txt'):
            sequence = np.loadtxt('npy_files/cifar100_class_order.txt')
        else:
            sequence = np.random.permutation(100)
            np.savetxt('npy_files/cifar100_class_order.txt', sequence)

        print('cifar100 incremental task sequence:', sequence)

    else:
        raise ValueError('Unknow Dataset')

    class_per_state = args.classes_per_classifier

    saved_dataset = Labeled_dataset(saved_img_path, val_trans, sequence[:task_id*class_per_state], offset=0)
    k15_dataset = k150_dataset(unlabel_path, val_trans)
    saved_loader = torch.utils.data.DataLoader(
        saved_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    k15_loader = torch.utils.data.DataLoader(
        k15_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    all_feature = feature_extract(k15_loader, model, criterion)
    target_feature = feature_extract(saved_loader, model, criterion)

    index_select = select_knn(target_feature, all_feature, args.unlabel_num)
    np.save(os.path.join(args.save_data_path,'task'+str(task_id)+'_select_index.npy'), np.array(index_select))
    all_data = pickle.load(open(unlabel_path,'rb'))
    all_image = all_data['data']
    all_label = all_data['label']
    new_data = {}
    new_data['data'] = all_image[index_select,:,:,:]
    new_data['label'] = all_label[index_select]
    print(new_data['data'].shape)
    print(new_data['label'].shape)
    pickle.dump(new_data, open(os.path.join(args.save_data_path,'selected_unlabel_task'+str(task_id)+'.pkl'), 'wb'))
    
    extract_dataset = k150_dataset(os.path.join(args.save_data_path,'selected_unlabel_task'+str(task_id)+'.pkl'), train_trans)
    extract_loader = torch.utils.data.DataLoader(
        extract_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True)
    label_extract(extract_loader, model, criterion, args.save_data_path, fc_num=task_id)

















