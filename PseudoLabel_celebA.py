import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.utils as vutils
import torch.nn.functional as F
from torch import nn, optim
import pickle
import shutil, os
from util import save_ckp, load_ckp, vat_loss
from model import get_model
from process_data import celebADataset

torch.manual_seed(42)
np.random.seed(0)

# Test Loss
def test_celebA(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        running_loss = 0
        acc = 0
        for b, (images, labels) in enumerate(test_loader):
            logits = model(images)
            loss = criterion(input=logits, target=labels)
            acc += torch.mean((torch.argmax(logits, 1)==labels).float())
            running_loss += loss.item()
    return running_loss / len(test_loader), acc.numpy() / len(test_loader)

# Train on Source
def train_Source_celebA(checkpoint_path, best_model_path, mixed, start_epochs, n_epochs, S_test_loss_min_input, S_train_loader, T_train_loader, S_test_loader, T_test_loader):
    criterion = nn.CrossEntropyLoss()
    time0 = time()
    print_iter = 100
    target_batch_num = 5
    S_loss_rec = []
    T_loss_rec = []
    S_acc_rec = []
    T_acc_rec = []
    train_loss_rec = []
     # initialize tracker for minimum validation loss
    S_test_loss_min = S_test_loss_min_input
    for e in range(start_epochs, n_epochs+1):
        running_loss = 0
        for b, (images, labels) in enumerate(S_train_loader):
            #images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(input=logits, target=labels)
            if mixed:
                T_images = []
                for j in range(target_batch_num):
                    T_images_j, _, _ = next(iter(T_train_loader))
                    T_images.append(T_images_j)
                T_images = torch.cat(T_images, dim=0)

                # Todo: Try larger batch_size for target (5x)
                logits_T = model(T_images)
                loss_T = torch.mean(torch.sum(- F.log_softmax(logits_T, 1) * F.softmax(logits_T, 1), 1))
                loss_T = loss_T + vat_loss(T_images, logits_T, model)
                loss += loss_T * 0.1

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if b % print_iter == 0:
                print("Epoch {} Iter {} - Training loss: {:.4f}".format(e, b, loss.item()))

        train_loss = running_loss / len(S_train_loader)
        train_loss_rec.append(train_loss)
        print("Epoch {} - Training loss: {:.4f}".format(e, train_loss))
        S_test_loss, S_test_acc = test_celebA(model, S_test_loader)
        T_test_loss, T_test_acc = test_celebA(model, T_test_loader)
        print("S, T Test loss: {:.4f}, {:.4f}".format(S_test_loss, T_test_loss))
        print("S, T Test Acc: {:.4f}, {:.4f}".format(S_test_acc, T_test_acc))

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': e+1,
            'S_test_loss_min': S_test_loss_min,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        ## Save the model if validation loss has decreased
        if S_test_loss < S_test_loss_min:
            print('S_test_loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(S_test_loss_min,S_test_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            S_test_loss_min = S_test_loss

        S_loss_rec.append(S_test_loss)
        T_loss_rec.append(T_test_loss)
        S_acc_rec.append(S_test_acc)
        T_acc_rec.append(T_test_acc)
    print("\nTraining Time (in minutes) =",(time()-time0)/60)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(S_loss_rec, label = 'Source Test')
    plt.plot(T_loss_rec, label = 'Target Test')
    plt.plot(train_loss_rec, label = 'Source Train')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.subplot(212)
    plt.plot(S_acc_rec, label = 'Source Test')
    plt.plot(T_acc_rec, label = 'Target Test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    plt.savefig('train_Source.png')

def finetune_Target_celebA(checkpoint_path, best_model_path, mixed, start_epochs, n_epochs, T_test_loss_min_input, S_train_loader, T_train_loader, S_test_loader, T_test_loader):
    criterion = nn.CrossEntropyLoss()
    time0 = time()
    print_iter = 5
    S_loss_rec = []
    T_loss_rec = []
    S_acc_rec = []
    T_acc_rec = []
    train_loss_rec = []
     # initialize tracker for minimum validation loss
    T_test_loss_min = T_test_loss_min_input
    for e in range(start_epochs, n_epochs+1):
        running_loss = 0
        for b, (T_images, _) in enumerate(T_train_loader):
            optimizer.zero_grad()
            logits_T = model(T_images)
            loss_T = torch.mean(torch.sum(- F.log_softmax(logits_T, 1) * F.softmax(logits_T, 1), 1))

            # Uncomment to add vat loss
            loss = loss_T #+ vat_loss(T_images, logits_T, model)

            # Option to mix source labeled loss with target unlabeled loss
            if mixed:
                S_images, S_labels, _ = next(iter(S_train_loader))
                logits_S = model(S_images)
                loss_S = criterion(input=logits_S, target=S_labels)
                loss += loss_S

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if b % print_iter == 0:
                print("Epoch {} Iter {} - Training loss: {:.4f}".format(e, b, loss.item()))
                if mixed:
                    print("loss_T, loss_S =", loss.item()-loss_S.item(), loss_S.item())

                S_test_loss, S_test_acc = test_celebA(model, S_test_loader)
                T_test_loss, T_test_acc = test_celebA(model, T_test_loader)
                print("S, T Test loss: {:.4f}, {:.4f}".format(S_test_loss, T_test_loss))
                print("S, T Test Acc: {:.4f}, {:.4f}".format(S_test_acc, T_test_acc))


        train_loss = running_loss / len(T_train_loader)
        train_loss_rec.append(train_loss)
        print("Epoch {} - Training loss: {:.4f}".format(e, train_loss))
        S_test_loss, S_test_acc = test_celebA(model, S_test_loader)
        T_test_loss, T_test_acc = test_celebA(model, T_test_loader)
        print("S, T Test loss: {:.4f}, {:.4f}".format(S_test_loss, T_test_loss))
        print("S, T Test Acc: {:.4f}, {:.4f}".format(S_test_acc, T_test_acc))

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': e+1,
            'T_test_loss_min': T_test_loss_min,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        ## Save the model if validation loss has decreased
        if T_test_loss < T_test_loss_min:
            print('T_test_loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(T_test_loss_min,T_test_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            T_test_loss_min = T_test_loss

        S_loss_rec.append(S_test_loss)
        T_loss_rec.append(T_test_loss)
        S_acc_rec.append(S_test_acc)
        T_acc_rec.append(T_test_acc)
    print("\nTraining Time (in minutes) =",(time()-time0)/60)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(S_loss_rec, label = 'Source Test')
    plt.plot(T_loss_rec, label = 'Target Test')
    plt.plot(train_loss_rec, label = 'Source Train')
    plt.ylabel('Entropy Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.subplot(212)
    plt.plot(S_acc_rec, label = 'Source Test')
    plt.plot(T_acc_rec, label = 'Target Test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    plt.savefig('finetune_Target.png')

#======= Main Algorithm Below ======

# CelebA dataset
n_way = 2
input_size = 3 * 64 * 64 # Color is relative ratio between 2 channels

num_tot = 20000
num_S_tot = int(num_tot * 0.1)
num_T_tot = num_tot - num_S_tot
# Testing purposes (10%)
#num_S_tot = [int(x/10) for x in num_S_tot]
#num_T_tot = [int(x/10) for x in num_T_tot]
train_Source = False
new_train = False
mixed = False # Bundle target and source in one batch
n_epoch = 120
checkpoint_get_path = './semiTest/checkpoints/checkpoint.pt'
checkpoint_save_path = './semiTest/checkpoints/checkpoint.pt'
bestmodel_save_path = './semiTest/checkpoints/best.pt'

save_path1 = './semiTest/dataLoaders/S_train_loader.pt'
save_path2 = './semiTest/dataLoaders/S_test_loader.pt'
save_path3 = './semiTest/dataLoaders/T_train_loader.pt'
save_path4 = './semiTest/dataLoaders/T_test_loader.pt'
model_type = 'myConvNet' # Choose from '3-layer', 'Linear', 'myConvNet'

time0 = time()
S_train_loader = torch.load(save_path1)
S_test_loader = torch.load(save_path2)
T_train_loader = torch.load(save_path3)
T_test_loader = torch.load(save_path4)
print("Data loading time (in minutes) =",(time()-time0)/60)

model = get_model(model_type=model_type, input_size=input_size, output_size=n_way)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.002)
if new_train:
    if train_Source:
        start_epoch = 0
        S_test_loss_min = 999
    else:
        model, _, start_epoch, _ = load_ckp(checkpoint_get_path, model, optimizer, True)
        T_test_loss_min = 999
        optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=0.02)
else:
    if train_Source:
        # load the saved checkpoint
        model, optimizer, start_epoch, S_test_loss_min = load_ckp(checkpoint_get_path, model, optimizer, True)
    else:
        model, optimizer, start_epoch, T_test_loss_min = load_ckp(checkpoint_get_path, model, optimizer, False)
if train_Source:
    train_Source_celebA(checkpoint_save_path, bestmodel_save_path, mixed, start_epoch, n_epoch, S_test_loss_min, S_train_loader, T_train_loader, S_test_loader, T_test_loader)
else:
    finetune_Target_celebA(checkpoint_save_path, bestmodel_save_path, mixed, start_epoch, n_epoch, T_test_loss_min, S_train_loader, T_train_loader, S_test_loader, T_test_loader)
