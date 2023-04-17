import os
import pandas as pd
import torchio as tio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append('../../3D_CNN_PyTorch')
import generate_model
#import convert_kspace
from torchmetrics.classification import BinaryAUROC
#from torchmetrics import AUROC
from tqdm import tqdm
import pydicom
import glob
import numpy as np

TRAIN_SPLIT_RATIO = 0.8
NUM_WORKERS = 4
BATCH_SIZE = 4
SEED = 42
NUM_EPOCHS = 50
RESULTS_FILE = f'CNvAD_train_results_{NUM_EPOCHS}-epochs_{BATCH_SIZE}-batch-size.csv'

BASE_DIR = './data'

nc_mci_ad_csv = os.path.join(BASE_DIR, 'CSV', 'NCvMCIvAD.csv')
nc_mci_ad_df = pd.read_csv(nc_mci_ad_csv)
nc_mci_ad_df = nc_mci_ad_df.drop_duplicates()


def set_loc(df):
    def get_loc(row):
        return os.path.join(BASE_DIR, 'MRI', row.PTID)

    location = df.apply(get_loc, axis=1)
    df['loc'] = location

set_loc(nc_mci_ad_df)


def get_subjects(dx_0, dx_1):
    assert dx_0 in ('CN', 'MCI')
    assert dx_1 in ('MCI', 'AD')

    subjects = []

    for _, row in nc_mci_ad_df.iterrows():
        if dx_0 == 'CN' and dx_1 == 'MCI':
            dx = dx_0 if row.DX_bl == 0 else dx_1
        elif dx_0 == 'CN' and dx_1 == 'AD':
            dx = dx_0 if row.DX_bl == 0 else dx_1
        elif dx_0 == 'MCI' and dx_1 == 'AD':
            dx = dx_0 if row.DX_bl == 1 else dx_1
        label = 0 if dx == dx_0 else 1

        img_path = os.path.join(row['loc'], 't2')

        # skip files with no SliceLocation (eg scout views)
        slices = []
        skipcount = 0

        files = []
        for fname in glob.glob(os.path.join(img_path, '*'), recursive=False):
        #     #print("loading: {}".format(fname))
             files.append(pydicom.dcmread(fname))

        for f in files:
            if hasattr(f, 'SliceLocation'):
                slices.append(f)
            else:
                skipcount = skipcount + 1

        #print("skipped, no SliceLocation: {}".format(skipcount))

        # ensure they are in the correct order
        slices = sorted(slices, key=lambda s: s.SliceLocation)

        if len(slices) == 0:
            print('No slices for', img_path, '\n')
            continue
        if skipcount > 0:
            print('Missing slices for', img_path, '\n\tSkipped', skipcount, 'slices.\n')
            continue

        subjects.append(
            tio.Subject(
                image=tio.ScalarImage(img_path),
                #diagnosis=dx,
                label=label)
            )
    return subjects


def run_epoch(model, criterion, loader, device, results, optimizer=None, metric=None):
    predlist = torch.zeros(0, dtype=torch.long).to(device)
    lbllist = torch.zeros(0, dtype=torch.long).to(device)

    count = 0
    total_acc, total_loss = 0, 0
    progress = tqdm(loader)

    for i, batch in enumerate(progress):
        data = batch['image'][tio.DATA].type(torch.FloatTensor).to(device)
        labels = batch['label'].to(device)

        #print(f'\tdata min is {data.min()}, data max is {data.max()}')

        outputs = model(data) # without converting to float: RuntimeError: Input type (torch.cuda.ShortTensor) and weight type (torch.cuda.FloatTensor) should be the same
        
        loss = criterion(outputs, labels)
        #log_softmax = F.log_softmax(outputs, dim=1)
        #loss = F.nll_loss(log_softmax, labels)
        total_loss += loss.item() * data.size(0)

        preds = torch.argmax(outputs.data, 1)
        total_acc += torch.sum(preds == labels).item()
    
        count += data.size(0)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress.set_description('train loss: %.3f | train acc: %.3f' % (total_loss / count, total_acc / count))
        else:
            predlist = torch.cat([predlist, preds.view(-1)])  # Append batch prediction results; for AUC
            lbllist = torch.cat([lbllist, labels.view(-1)])
            progress.set_description(' test loss: %.3f |  test acc: %.3f' % (total_loss / count, total_acc / count))

    if optimizer is not None:
        print('\tTrain | loss: %.3f, acc: %.3f' % (total_loss / count, total_acc / count))
        # print('loss: %.3f | acc: %.3f' % (total_loss / count, total_acc / count))
        # print('\t\ttrain--total_loss/count:', total_loss / count)
        # print('\t\ttrain--total_loss/len(loader):', total_loss / len(loader))
        # print('\t\ttrain--total_acc/count:', total_acc / count)
        # print('\t\ttrain--total_acc/len(loader):', total_acc / len(loader))
        results['train_loss']=f'{(total_loss / count):.3f}'
        results['train_acc']=f'{(total_acc / count):.3f}'
    else:
        auroc = metric(predlist, lbllist)
        print('\t Test | loss: %.3f, acc: %.3f, auc: %.3f' % (total_loss / count, total_acc / count, auroc.item()))

        # print('\t\ttest--total_loss/count:', total_loss / count)
        # print('\t\ttest--total_loss/len(loader):', total_loss / len(loader))
        # print('\t\ttest--total_acc/count:', total_acc / count)
        # print('\t\ttest--total_acc/len(loader):', total_acc / len(loader))
        results['test_loss'] = f'{(total_loss / count):.3f}'
        results['test_acc']=f'{(total_acc / count):.3f}'
        results['test_auc']=f'{auroc.item():.3f}'

def write_results(results, epoch):
    with open(RESULTS_FILE, 'a') as f:
        res = str(epoch)+','+','.join(results.values())
        f.write(res+'\n')

def main():
    torch.manual_seed(SEED)

    nc_ad_subjects = get_subjects('CN', 'AD')
    mci_ad_subjects = get_subjects('MCI', 'AD')
    nc_mci_subjects = get_subjects('CN', 'MCI')

    pairwise_subjects = {'CNvAD': nc_ad_subjects, 'MCIvAD': mci_ad_subjects, 'CNvMCI': nc_mci_subjects}

    for subj_pair_name, subj_pair in pairwise_subjects.items():
        global RESULTS_FILE
        RESULTS_FILE = f'{subj_pair_name}_train_results_{NUM_EPOCHS}-epochs_{BATCH_SIZE}-batch_size.csv'

        print('\nStarting', subj_pair_name)

        N_SAMPLES = len(subj_pair)
        n_train = int(N_SAMPLES * TRAIN_SPLIT_RATIO)
        n_test = N_SAMPLES - n_train

        print('Total subjects:', N_SAMPLES, '\n')

        from torchio.transforms import (
            CropOrPad,
            OneOf,
            RescaleIntensity,
            RandomAffine,
            RandomElasticDeformation,
            RandomFlip,
            Compose,
            Resample,
            Resize
        )

        IMAGE_SIZE = [256, 256, 48]
        rescale = RescaleIntensity((0.05, 99.5))
        resample = tio.Resample(1)
        
        half_res = [dim // 2 for dim in IMAGE_SIZE[:2]]
        half_res.append(IMAGE_SIZE[-1])
        resize = tio.Resize(half_res)

        randaffine = tio.RandomAffine(scales=(0.9,1.2),degrees=10, isotropic=True, image_interpolation='nearest')
        flip = tio.RandomFlip(axes=('LR'), p=0.5)
        pad = CropOrPad(IMAGE_SIZE) # crop/pad to most common shape
        #train_transform = Compose([rescale, flip, randaffine, pad])
        #test_transform = Compose([rescale, pad])
        #train_transform = Compose([flip, randaffine, pad])
        #test_transform = Compose([pad])

        train_transform = Compose([rescale, flip, randaffine, resample, resize])
        test_transform = Compose([rescale, resample, resize])

        train_subjects, test_subjects = random_split(subj_pair, [n_train, n_test], generator=torch.Generator().manual_seed(SEED))
        trainset = tio.SubjectsDataset(train_subjects, transform=train_transform)
        testset = tio.SubjectsDataset(test_subjects, transform=test_transform)

        def count_split(dataset):
            n_0, n_1 = 0, 0
            for subj in dataset:
                if subj['label'] == 0:
                    n_0 += 1
                elif subj['label'] == 1:
                    n_1 += 1
                else:
                    print(subj['label'], "invalid label")
            
            name_0, name_1 = subj_pair_name.split('v')
            print(f'\t{name_0}: {n_0}')
            print(f'\t{name_1}: {n_1}')
            print()

        print('Training set:', len(trainset), 'samples')
        count_split(trainset)

        print('Testing set:', len(testset), 'samples')
        count_split(testset)

        trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        print('batch size:', BATCH_SIZE)
        print('length trainloader:', len(trainloader))
        print('length testloader:', len(testloader), '\n')

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
        #sample_size = np.prod(IMAGE_SIZE)  # image size, i.e., h*w*d

        sample_size = np.prod(half_res)  # image size, i.e., h*w*d
        
        model = generate_model.main(cnn_name, model_depth, n_classes, in_channels, sample_size).to(device)
        #model = UNet().to(device)
        #model.to(device)

        optimizer = torch.optim.AdamW(model.parameters())#, lr=1e-3, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        #criterion = F.nll_loss()
        metric = BinaryAUROC().to(device)


        header = ['epoch','train_loss','train_acc','test_loss','test_acc','test_auc']
        with open(RESULTS_FILE, 'w') as f:
            f.write(','.join(header)+'\n')

        for epoch in range(NUM_EPOCHS):
            results = {}

            print('epoch %d:' % epoch)
            run_epoch(model, criterion, trainloader, device, results, optimizer=optimizer)
            run_epoch(model, criterion, testloader, device, results, metric=metric)

            write_results(results, epoch)

        print('----------------------------------\n')


if __name__ == '__main__':
   main()
