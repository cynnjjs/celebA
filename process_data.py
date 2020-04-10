import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)
np.random.seed(0)
image_size = 64
transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ])

def preprocess_celebA(Y_attr, data_path):
    filepath = 'list_attr_celeba.txt'
    with open(filepath, 'r') as f:
        f.readline()
        attribute_names = f.readline().strip().split(' ')
        idx_y = attribute_names.index(Y_attr)
        for i, line in enumerate(f):
            fields = line.strip().replace('  ', ' ').split(' ')
            img_name = fields[0]
            f = os.path.join(data_path, img_name)
            attr_vec = np.array([int(x) for x in fields[1:]])


            d = datasets.ImageFolder(root=data_folder[idx], transform=transform)
            for i in range(num_list[idx]):
                self.data.append(d[i+start_idx[idx]][0])
            self.data = torch.stack(self.data)
            self.target = torch.cat((torch.ones(num_list[0]), torch.zeros(num_list[1]), torch.ones(num_list[2]), torch.zeros(num_list[3])),0).long()

            if (attr_vec[idx_x2]==1 and attr_vec[idx_y]==1):
                shutil.copy(f, x21y1_folder)
                c11 += 1
            elif (attr_vec[idx_x2]==1 and attr_vec[idx_y]==-1):
                shutil.copy(f, x21y0_folder)
                c10 += 1
            elif (attr_vec[idx_x2]==-1 and attr_vec[idx_y]==1):
                shutil.copy(f, x20y1_folder)
                c01 += 1
            elif (attr_vec[idx_x2]==-1 and attr_vec[idx_y]==-1):
                shutil.copy(f, x20y0_folder)
                c00 += 1

        print(c11, c10, c01, c00)

class celebADataset(Dataset):
    def __init__(self, Snum, Tnum, Y_attr, data_folder, source):

        if source:
            start_idx = 0
            num_list = Snum
        else:
            start_idx = Snum
            num_list = Tnum
        self.data = []
        d = datasets.ImageFolder(root=data_folder, transform=transform)
        for i in range(num_list):
            self.data.append(d[i+start_idx][0])
        self.data = torch.stack(self.data)

        # Assign Y labels
        self.target = torch.zeros(len(self.data))
        filepath = 'list_attr_celeba.txt'
        with open(filepath, 'r') as f:
            f.readline()
            attribute_names = f.readline().strip().split(' ')
            idx_y = attribute_names.index(Y_attr)
            if (source==False):
                for _ in range(Snum):
                    f.readline()
            for i, line in enumerate(f):
                if (i >= num_list):
                    break
                fields = line.strip().replace('  ', ' ').split(' ')
                attr_vec = np.array([int(x) for x in fields[1:]])
                if attr_vec[idx_y]==1:
                    self.target[i] = 1
        self.target = self.target.long()
        #self.target = torch.cat((torch.ones(num_list[0]), torch.zeros(num_list[1]), torch.ones(num_list[2]), torch.zeros(num_list[3])),0).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def split_dataset(dataset, batch_size, shuffle, validation_split):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader

def construct_CelebA_dataset(num_S_tot, num_T_tot, Y_attr, data_path):
    split = .1
    batch_size = 64

    S_dataset = celebADataset(num_S_tot, num_T_tot, Y_attr, data_path, source=True)
    T_dataset = celebADataset(num_S_tot, num_T_tot, Y_attr, data_path, source=False)
    S_train_loader, S_test_loader = split_dataset(S_dataset, batch_size=batch_size, shuffle=True, validation_split=split)
    T_train_loader, T_test_loader = split_dataset(T_dataset, batch_size=batch_size, shuffle=True, validation_split=split)

    return S_train_loader, S_test_loader, T_train_loader, T_test_loader

data_path = '../../../../scr-ssd/datasets/'
#num_tot = 202600
num_tot = 20000
num_S_tot = int(num_tot * 0.1)
num_T_tot = num_tot - num_S_tot
S_train_loader, S_test_loader, T_train_loader, T_test_loader = construct_CelebA_dataset(num_S_tot, num_T_tot, Y_attr='Male', data_path=data_path)
# Decide which device we want to run on
#device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Save DataLoaders
save_path1 = './semiTest/dataLoaders/S_train_loader.pt'
save_path2 = './semiTest/dataLoaders/S_test_loader.pt'
save_path3 = './semiTest/dataLoaders/T_train_loader.pt'
save_path4 = './semiTest/dataLoaders/T_test_loader.pt'
torch.save(S_train_loader, save_path1)
torch.save(S_test_loader, save_path2)
torch.save(T_train_loader, save_path3)
torch.save(T_test_loader, save_path4)

# Plot some training images
"""
real_batch = next(iter(T_train_loader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig('T_train_loader.png')
"""
