#import torchvision
import glob
import logging
import os
import sys
import pydicom
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torchmetrics.classification import AUROC
from torchmetrics import Accuracy
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchio as tio
sys.path.append('../../3D_CNN_PyTorch')
import generate_model

#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter('lightning_logs/mri')
#import convert_kspace
#from torchmetrics.classification import BinaryAUROC

TRAIN_SPLIT_RATIO = 0.8
NUM_WORKERS = 4
BATCH_SIZE = 8 
SEED = 42
NUM_EPOCHS = 20
RESULTS_FILE = f'ADNI_multiclass_train_results_{NUM_EPOCHS}-epochs_{BATCH_SIZE}-batch-size_pl.csv'  # pl = pytorch_lightning
BASE_DIR = './data'

class MRImageClassifier(pl.LightningModule):
    def __init__(self, cnn_name, model_depth, n_classes, in_channels, sample_size):
        super().__init__()
        self.n_classes = n_classes
        self.model = generate_model.main(cnn_name, model_depth, n_classes, in_channels, sample_size)
        self.auroc = AUROC(task='multiclass', num_classes=n_classes)
        self.accuracy = Accuracy(task='multiclass', num_classes=n_classes)
        self.report_f = open(RESULTS_FILE, 'w')
        self.report_f.write('epoch,train_loss,train_acc,test_loss,test_acc,test_auc\n')
        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, x):
        return self.model(x)


    def on_validation_epoch_start(self):
        self.val_labels = []
        self.val_scores = []
  

    def step_impl(self, batch, prefix):
        image = batch['image']['data']
        label = batch['label']
        y_pred = self.model(image)
        loss = self.loss_fn(y_pred, label)
        acc = self.accuracy(y_pred.data, label)
    
        # AUROC
        y_prob = torch.softmax(y_pred.data, dim=1)#[:, 1]
        metric = AUROC(task='multiclass', num_classes=self.n_classes)
        #auroc = metric(label, y_prob)

        auroc = metric(y_prob, label)

        #self.log('val_auroc', auroc)
        #print('calculated batch size:', image.shape[0]) # == BATCH_SIZE
        self.log('%s_acc' % prefix, acc, batch_size=image.shape[0], on_step=False, on_epoch=True)
        self.log("%s_loss" % prefix, loss, batch_size=image.shape[0], on_step=False, on_epoch=True)

        return loss, label, y_prob#[:, 1]


    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step_impl(batch, 'train')
        #self.train_labels.append(label)
        #self.train_scores.append(score.view(-1))
        #x, y = batch
        #y_pred = self(x)
        #loss = self.loss_fn(y_pred, y)
        #self.log('train_loss', loss)
        return loss
    

    def validation_step(self, batch, batch_idx):
        loss, label, score = self.step_impl(batch, 'val')
        self.val_labels.append(label)
        self.val_scores.append(score)
        # x, y = batch
        # y_pred = self.model(x)
        # loss = self.loss_fn(y_pred, y)
        # self.log('val_loss', loss)

        # # AUROC
        # y_prob = torch.softmax(y_pred.data, dim=1)[:, 1]
        # y_true =y.data
        # metric = AUROC(task='multiclass', num_classes=n_classes)
        # auroc = metric(y_true, y_prob)
        # self.log('val_auroc', auroc)
        return loss
    

    def on_train_epoch_end(self):
        labels = torch.cat(self.val_labels)
        scores = torch.cat(self.val_scores)
        auc = self.auroc(scores, labels)
        self.log('auc', auc)

        epoch = self.trainer.current_epoch
        metrics = self.trainer.callback_metrics
        logging.info('%03d: train_loss: %.3f train_acc: %.3f val_loss: %.3f val_acc: %.3f auc: %.3f' % (
            epoch, metrics['train_loss'], metrics['train_acc'], metrics['val_loss'], metrics['val_acc'], auc))
        if self.report_f is not None:
            self.report_f.write('%d,%g,%g,%g,%g,%g\n' % (
                epoch,
                metrics['train_loss'],
                metrics['train_acc'],
                metrics['val_loss'],
                metrics['val_acc'],
                auc)
            )


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())#, lr=1e-3, weight_decay=1e-3)


    # def train_dataloader(self):
    #     transform = []
    #     trainset = MRImageDataset(data_path=os.path.join(BASE_DIR, 'MRI'), transform=transform)
    #     trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    #     return trainloader    

    # def val_dataloader(self):
    #     transform = []
    #     testset = MRImageDataset(data_path=os.path.join(BASE_DIR, 'MRI'), transform=transform)
    #     testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    #     return testloader

def set_loc(df):
    def get_loc(row):
        return os.path.join(BASE_DIR, 'MRI', row.PTID)

    location = df.apply(get_loc, axis=1)
    df['loc'] = location


def get_patient_image_locs():
    DIG_LEN = 4
    for row, _ in patient_df.iterrows():
        if row % 10 == 0:
            num_digs = int(np.log10(row)+1)
        else:
            num_digs = int(np.ceil(np.log10(row)))
        #num_digs = int(np.log10(1000 % row))
        num_zeroes = DIG_LEN - num_digs  # num zeroes to pad with
        if num_zeroes < 0:
            num_zeroes = 1

        match_pattern = r'\d+_S_[0]{%d}%d'%(num_zeroes, row)
        matches = nc_mci_ad_df['PTID'].str.match(match_pattern)
        if matches is not None:
            if (len(nc_mci_ad_df[matches]['loc']) > 1):
                break
            patient_df.loc[row, 'image'] = nc_mci_ad_df[matches].iloc[0]['loc']


def drop_bad_scans():
    print('Dropping bad scans...')
    bad_scans = ['006_S_1130', '068_S_0476', '068_S_0127', '068_S_0109', '100_S_0047', '068_S_0210']

    bad_indices = []
    for scan in bad_scans:
        index_val = patient_df[patient_df['image'].str.contains(scan)].index
        try:
            bad_indices.append(index_val.item())
        except:
            print(f'index is {index_val}. Skipping')
            continue
    
    patient_df.drop(bad_indices, inplace=True)


def get_subjects():
    # 1 = Normal Control
    # 2 = Mild Cognitive Impairment
    # 3 = Alzheimers Disease

    ## drop some mci samples to balance classes
    #mci_samples = self.patient_df[self.patient_df.DX_bl == 2]
    #patient_df.drop(mci_samples.sample(n=60, axis=0, random_state=SEED).index, inplace=True)#.reset_index()

    subjects = []

    for _, row in patient_df.iterrows():
        assert(row.DX_bl in (1, 2, 3))
        if row.DX_bl == 1:
            dx = 'CN'
        elif row.DX_bl == 2:
            dx = 'MCI'
        elif row.DX_bl == 3:
            dx = 'AD'

        img_path = os.path.join(row['image'], 't2')

        # skip files with no SliceLocation (eg scout views))
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
                diagnosis=dx,
                label=int(row.DX_bl-1))
            )
    return subjects

    # def __getitem__(self, index):
    #     subject = self.subjects[index]
    #     image = subject['image']
    #     if self.transform:
    #         image = self.transform(image)
    #     return image.data, torch.tensor(subject['label'])

    # def __len__(self):
    #     return len(self.subjects)


def show_counts(trainset, testset, trainloader=None, testloader=None):
    def count_split(dataset):
        n_0, n_1, n_2 = 0, 0, 0
        for subj in dataset:
            if subj['label'] == 0:
                n_0 += 1
            elif subj['label'] == 1:
                n_1 += 1
            elif subj['label'] == 2:
                n_2 += 1
            else:
                print(subj['label'], "invalid label")
        
        print(f'\tCN: {n_0}')
        print(f'\tMCI: {n_1}')
        print(f'\tAD: {n_2}')
        print()

    print('Training set:', len(trainset), 'samples')
    count_split(trainset)

    print('\nTesting set:', len(testset), 'samples')
    count_split(testset)

    print('\nbatch size:', BATCH_SIZE)
    print('length trainloader:', len(trainloader))
    print('length testloader:', len(testloader), '\n')


nc_mci_ad_csv = os.path.join(BASE_DIR, 'CSV', 'NCvMCIvAD.csv')
nc_mci_ad_df = pd.read_csv(nc_mci_ad_csv)
nc_mci_ad_df = nc_mci_ad_df.drop_duplicates()

#mci_samples = self.nc_mci_ad_df[nc_mci_ad_df.DX_bl == 1]
#nc_mci_ad_df.drop(self.mci_samples.sample(n=60, axis=0, random_state=SEED).index, inplace=True)

# TODO: compare DX_bl to DX_latest
patient_csv = os.path.join(BASE_DIR, 'patient_index_and_fold_mapping.csv')
patient_df = pd.read_csv(patient_csv, index_col='RID')

def main():
    #print('pl version:', pl.__version__)  # 2.0.2

    set_loc(nc_mci_ad_df)
    get_patient_image_locs()
    drop_bad_scans()

    subjects = get_subjects()

    n_samples = len(subjects)
    n_train = int(n_samples * TRAIN_SPLIT_RATIO)
    n_test = n_samples - n_train

    (train_subjects,
     test_subjects) = random_split(
                        subjects,
                        [n_train, n_test],
                        generator=torch.Generator().manual_seed(SEED))

    #test_subjects = torch.utils.data.Subset(list(train_subjects), [i for i in range(n_train)])
    #if not set(subj['image']['path'] for subj in list(test_subjects)).issubset(set(subj['image']['path'] for subj in list(train_subjects))):
    #    sys.exit()
    # print('len train:', len(train_subjects))
    # print('len test:', len(test_subjects))
    # print(next(iter(train_subjects)))
    # print('train paths == test paths:', set([subj['image']['path'] for subj in list(train_subjects)]) == set([subj['image']['path'] for subj in list(test_subjects)]))
    # sys.exit()
    # Transforms
    IMAGE_SIZE = [256, 256, 48]
    rescale = tio.RescaleIntensity((0.05, 99.5))
    #resample = tio.Resample(1)
    
    half_res = [dim // 2 for dim in IMAGE_SIZE[:2]]
    half_res.append(IMAGE_SIZE[-1])
    resize = tio.Resize(half_res)

    randaffine = tio.RandomAffine(scales=(0.9,1.2),degrees=10, isotropic=True, image_interpolation='nearest')
    flip = tio.RandomFlip(axes=('LR'), p=0.5)
    pad = tio.CropOrPad(IMAGE_SIZE) # crop/pad to most common shape

    train_transform = tio.Compose([resize, rescale, flip, randaffine])
    trainset = tio.SubjectsDataset(train_subjects, transform=train_transform)
    
    test_transform = tio.Compose([resize, rescale])
    testset = tio.SubjectsDataset(test_subjects, transform=test_transform)

    trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # examples = iter(testloader)
    # example_data = next(iter(testloader))#, example_targets = next(examples)
    # print(example_data['image'])
    # img_grid = torchvision.utils.make_grid(example_data['image'][tio.DATA][..., example_data['image'][tio.DATA].shape[-1]//2])
    # writer.add_image('mri images', img_grid)
    # writer.close()
    # sys.exit()
    

    show_counts(trainset, testset, trainloader, testloader)


    LOG_FMT = "[%(levelno)d] %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    

    #sample_size = np.prod(IMAGE_SIZE)  # image size, i.e., h*w*d
    sample_size = np.prod(half_res)  # image size, i.e., h*w*d
    model = MRImageClassifier(cnn_name='resnet',
                              model_depth=18,
                              n_classes=3,
                              in_channels=1,
                              sample_size=sample_size)
    
    ck = pl.callbacks.ModelCheckpoint(dirpath="models", save_top_k=1, monitor="val_loss")
    #trainer = pl.Trainer(callbacks=[ck], gpus=1, enable_progress_bar = True, max_epochs=NUM_EPOCHS)
    #trainer = pl.Trainer(max_epochs=NUM_EPOCHS,)
                #callbacks=[ck], enable_progress_bar=True)
    
    trainer = pl.Trainer(
        #logger=logger,
        callbacks=[ck],
        max_epochs=NUM_EPOCHS,
        #gpus=1,
        #progress_bar_refresh_rate=30
    )
    trainer.fit(model, trainloader, testloader)
    
if __name__ == '__main__':
    main()
