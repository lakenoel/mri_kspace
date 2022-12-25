"""
code adapted from
https://github.com/farazahmeds/Classification-of-brain-tumor-using-Spatiotemporal-models/blob/main/dataset/datasets.py
"""
import os
#import multiprocessing
from pathlib import Path
import numpy as np
import torchio as tio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
#import torchvision
#from unet import UNet
import sys
sys.path.append('../3D_CNN_PyTorch')
import generate_model
#import convert_kspace
from torchmetrics.classification import BinaryAUROC
#from torchmetrics import AUROC
from tqdm import tqdm
#import argparse
import torch.nn.functional as F


N_SAMPLES = 98 #1160
TRAIN_SPLIT_RATIO = 0.8
NUM_EPOCHS = 2
NUM_WORKERS = 4
BATCH_SIZE = 1
SEED = 42

# gbm T1 r.t. T1GD; no-test-transform[augmentation]
RESULTS_FILE = f'train_results_{N_SAMPLES}-samples_{NUM_EPOCHS}-epochs_T1vT1_unstripped.csv'

IXI_DIR = Path('./Data/IXI/T1')  # T1, T2 (can also download MRA, PD, and DTI)
GBM_DIR = Path('./Data/UPennGBM/images_structural_unstripped/')  # T1, T1GD, FLAIR, T2

def get_ixi():
    ixi = []  # healthy samples

    for dirpath, dirnames, filenames in sorted(os.walk(IXI_DIR)):  # os.walk(os.path.join(IXI_DIR, 'T1'))
        for file in filenames:
            if file.endswith('-T1.nii.gz'):
                img_path = os.path.join(dirpath, file)
                ixi.append(tio.Subject(t1=tio.ScalarImage(img_path), label=0,))
    return ixi

def get_gbm():
    gbm = []  # glioblastoma samples

    for dirpath, dirnames, filenames in sorted(os.walk(GBM_DIR)):
        for file in filenames:
            # exclude post op follow up scans
            if '_21_' not in file and file.endswith('_T1_unstripped.nii.gz'):
                img_path = os.path.join(dirpath, file)
                gbm.append(tio.Subject(t1=tio.ScalarImage(img_path), label=1,))
    return gbm

#gbm_subjects_dataset = tio.SubjectsDataset(gbm, transform=transform)
#ixi_subjects_dataset = tio.SubjectsDataset(ixi, transform=transform)


def train(model, criterion, loader, optimizer, device, results):
    count = 0
    total_acc, total_loss = 0, 0
    progress = tqdm(loader)
    for i, batch in enumerate(progress):
        data = batch['t1'][tio.DATA].type(torch.FloatTensor).to(device)
        labels = batch['label'].to(device)
        outputs = model(data) # without converting to float: RuntimeError: Input type (torch.cuda.ShortTensor) and weight type (torch.cuda.FloatTensor) should be the same
        
        loss = criterion(outputs, labels)
        #log_softmax = F.log_softmax(outputs, dim=1)
        #loss = F.nll_loss(log_softmax, labels)
        total_loss += loss.item() * data.size(0)

        preds = torch.argmax(outputs.data, 1)
        total_acc += torch.sum(preds == labels).item()
    
        count += data.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        progress.set_description('train loss: %.3f | train acc: %.3f' % (total_loss / count, total_acc / count))

    print('\tTrain | loss: %.3f, acc: %.3f' % (total_loss / count, total_acc / count))

    results['train_loss']=f'{(total_loss / count):.3f}'
    results['train_acc']=f'{(total_acc / count):.3f}'

def test(model, criterion, loader, metric, device, predlist, lbllist, results):
    count = 0
    total_acc, total_loss = 0, 0

    progress = tqdm(loader)
    for i, batch in enumerate(progress):
        data = batch['t1'][tio.DATA].type(torch.FloatTensor).to(device)
        labels = batch['label'].to(device)
        outputs = model(data) # without converting to float: RuntimeError: Input type (torch.cuda.ShortTensor) and weight type (torch.cuda.FloatTensor) should be the same
        
        loss = criterion(outputs, labels)
        #log_softmax = F.log_softmax(outputs, dim=1)
        #loss = F.nll_loss(log_softmax, labels)
        total_loss += loss.item() * data.size(0)

        preds = torch.argmax(outputs.data, 1)
        total_acc += torch.sum(preds == labels).item()
    
        count += data.size(0)

        predlist = torch.cat([predlist, preds.view(-1)])  # Append batch prediction results; for AUC
        lbllist = torch.cat([lbllist, labels.view(-1)])
        progress.set_description(' test loss: %.3f |  test acc: %.3f' % (total_loss / count, total_acc / count))

    auroc = metric(predlist, lbllist)
    print('\t Test | loss: %.3f, acc: %.3f, auc: %.3f' % (total_loss / count, total_acc / count, auroc.item()))

    results['test_loss'] = f'{(total_loss / count):.3f}'
    results['test_acc']=f'{(total_acc / count):.3f}'
    results['test_auc']=f'{auroc.item():.3f}'

def run_epoch(epoch, model, criterion, trainloader, testloader, optimizer, metric, device):
    results = {}
    # results = {'train_loss': [],
    #             'train_acc': [],
    #             'test_loss': [],
    #             'test_acc': [],
    #             'test_auc': []}
    
    predlist = torch.zeros(0, dtype=torch.long).to(device)
    lbllist = torch.zeros(0, dtype=torch.long).to(device)

    train(model, criterion, trainloader, optimizer, device, results)
    test(model, criterion, testloader, metric, device, predlist, lbllist, results)
    write_results(results, epoch)


def write_results(results, epoch):
    with open(RESULTS_FILE, 'a') as f:
        res = str(epoch)+','+','.join(results.values())
        f.write(res+'\n')


def main():
    torch.manual_seed(SEED)
    gbm = get_gbm()  # 611 samples
    ixi = get_ixi()  # 581 samples

    print('Samples in UPennGBM dataset:', len(gbm))
    print('Samples in IXI dataset:', len(ixi))
    print()
    print('Total samples requested:', N_SAMPLES)
    print()

    n_train = int(N_SAMPLES * TRAIN_SPLIT_RATIO)
    n_test = N_SAMPLES - n_train

    from torchio.transforms import (
        CropOrPad,
        OneOf,
        RescaleIntensity,
        RandomAffine,
        RandomElasticDeformation,
        RandomFlip,
        Compose,
    )

    rescale = RescaleIntensity((0.05, 99.5))
    randaffine = tio.RandomAffine(scales=(0.9,1.2),degrees=10, isotropic=True, image_interpolation='nearest')
    flip = tio.RandomFlip(axes=('LR'), p=0.5)
    #pad = CropOrPad([256,256,150])
    pad = CropOrPad([240,240,155])
    train_transform = Compose([rescale, flip, randaffine, pad])
    test_transform = Compose([pad])

    samples_per_dataset = N_SAMPLES // 2

    print('Getting', samples_per_dataset, 'samples from each dataset\n')
    subjects_list = gbm[:samples_per_dataset] + ixi[:samples_per_dataset]

    train_subjects, test_subjects = random_split(subjects_list, [n_train, n_test], generator=torch.Generator().manual_seed(SEED))

    trainset = tio.SubjectsDataset(train_subjects, transform=train_transform)
    testset = tio.SubjectsDataset(test_subjects, transform=test_transform)

    def count_split(dataset):
        n_ixi, n_gbm = 0, 0
        for subj in dataset:
            if subj['label'] == 0:
                n_ixi += 1
            elif subj['label'] == 1:
                n_gbm += 1
            else:
                print(subj['label'], "invalid label")
        
        print('\tIXI:', n_ixi)
        print('\tGBM:', n_gbm)
        print()

    print('Training set:', len(trainset), 'samples')
    count_split(trainset)

    print('Testing set:', len(testset), 'samples')
    count_split(testset)

    trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


    # # Visualize axial slices of one batch
    # layer = 100
    # batch_t1 = one_batch['t1'][tio.DATA][..., layer]
    # batch_label = one_batch['label'][tio.DATA][:, 1:, ..., layer]
    # slices = torch.cat((batch_mri, batch_label))
    # image_path = 'batch_whole_images.png'
    # torchvision.utils.save_image(
    #     slices,
    #     image_path,
    #     nrow=BATCH_SIZE // 2,
    #     normalize=True,
    #     scale_each=True,
    #     padding=0
    # )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cnn_name = 'resnet'
    model_depth = 101 # (18|34|50|101|152|200)
    n_classes = 2  # output classes
    in_channels = 1 # model input channels (1|3)
    sample_size = 240*240*155  # image size

    model = generate_model.main(cnn_name, model_depth, n_classes, in_channels, sample_size).to(device)
    #model = UNet().to(device)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters())#, lr=1e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    #criterion = F.nll_loss()
    metric = BinaryAUROC().to(device)


    # results = {'train_loss': [],
    #            'train_acc': [],
    #            'test_loss': [],
    #            'test_acc': [],
    #            'test_auc': []}

    header = ['epoch','train_loss','train_acc','test_loss','test_acc','test_auc']
    with open(RESULTS_FILE, 'w') as f:
        f.write(','.join(header)+'\n')

    for epoch in range(NUM_EPOCHS):
        predlist = torch.zeros(0, dtype=torch.float).to(device)
        lbllist = torch.zeros(0, dtype=torch.float).to(device)
        print('epoch %d:' % epoch)
        run_epoch(epoch, model, criterion, trainloader, testloader, optimizer, metric, device)

    #torch.save(model.state_dict(), f'{}.pth')


if __name__ == '__main__':
    main()
