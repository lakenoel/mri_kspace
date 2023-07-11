# https://github.com/mvrl/alzheimer-project/blob/master/Dynamic%2BAttention%20for%20AD%20MRI%20classification/scripts/Baseline-3DResNet.ipynb
import torch
from torch import nn
from torch.utils.data import Dataset
#import torchvision.models as models
import os
import numpy as np
from sklearn import metrics
from tqdm import trange, tqdm
#import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import torchio as tio
import pandas as pd
import pydicom
import glob
import random
from torch.utils.data import DataLoader
from collections import Counter
import sys
sys.path.append('../../3D_CNN_PyTorch/') # https://github.com/xmuyzz/3D-CNN-PyTorch
import generate_model
from convert_kspace_dcm import get_kspace, get_reduced_image

BATCH_SIZE = 32 
EPOCHS = 50
NUM_FOLDS = 5 
NUM_WORKERS = 4
SEED = 42

KSPACE = False  # True if raw k-space run, False otherwise
REDUCED_KSPACE = False  # True if reduced percentage MRI run, False otherwise
MRI = True  # True if raw MRI run, False otherwise

if KSPACE:
    REDUCED_KSPACE = MRI = False
elif REDUCED_KSPACE:
    KSPACE = MRI = False
elif MRI:
    KSPACE = REDUCED_KSPACE = False
assert(KSPACE or REDUCED_KSPACE or MRI)

# required if REDUCED_KSPACE == True
REDUCED_PERCENTAGE = 80

# LR = 0.0001
# LOSS_WEIGHTS = torch.tensor([1., 1.]) 

BASE_DIR = '../data'

nc_mci_ad_csv = os.path.join(BASE_DIR, 'CSV', 'NCvMCIvAD.csv')
nc_mci_ad_df = pd.read_csv(nc_mci_ad_csv)
nc_mci_ad_df = nc_mci_ad_df.drop_duplicates()

def set_loc(df):
    def get_loc(row):
        return os.path.join(BASE_DIR, 'MRI', row.PTID)

    location = df.apply(get_loc, axis=1)
    df['loc'] = location

#LABELS = {0: 'NC', 1: 'MCI', 2: 'AD'}
LABELS = {0: 'NC', 1: 'AD'}#'MCI', 2: 'AD'}
# 1 = Normal Control
# 2 = Mild Cognitive Impariment
# 3 = Alzhiemers Disease

# TODO: compare DX_bl to DX_latest
patient_csv = os.path.join(BASE_DIR, 'patient_index_and_fold_mapping.csv')
patient_df = pd.read_csv(patient_csv, index_col='RID')

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
    bad_scans = ['062_S_0535', '033_S_0739', '006_S_1130', '068_S_0476', '068_S_0127', '068_S_0109', '100_S_0047', '068_S_0210']
    bad_indices = []
    for scan in bad_scans:
        index_val = patient_df[patient_df['image'].str.contains(scan)].index
        try:
            bad_indices.append(index_val.item())
        except:
            print('index is', index_val, '. Skipping')
            continue
    
    patient_df.drop(bad_indices, inplace=True)


def get_subjects():
    print('Getting subjects...')
    subjects = []

    for _, row in patient_df.iterrows():
        assert(row.DX_bl in (1, 2, 3))
        if row.DX_bl == 1:
            dx = 'NC'
            label = 0
        elif row.DX_bl == 2:
            continue
            #dx = 'MCI'
        elif row.DX_bl == 3:
            dx = 'AD'
            label = 1

        img_path = os.path.join(row['image'], 't2')

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
            print('\tNo slices for', img_path, '\n')
            continue
        if skipcount > 0:
            print('\tMissing slices for', img_path, '\n\tSkipped', skipcount, 'slices.\n')
            continue

        if KSPACE: 
            print('Converting slices to k-space for img', img_path)
            slices = [get_kspace(s.pixel_array) for s in slices]
            img = np.expand_dims(np.stack(slices).transpose(1,2,0), axis=0)
        elif REDUCED_KSPACE:
            print('Converting slices to reduced k-space for img', img_path)
            img = get_reduced_image(slices, REDUCED_PERCENTAGE)
        elif MRI: 
            slices = [s.pixel_array for s in slices]
            #print('shape:', np.stack(slices).shape)
            img = np.expand_dims(np.stack(slices).transpose(1,2,0), axis=0) 
        
        #label = int(row.DX_bl)-1
        diagnosis=dx
        subjects.append({
            'image': img,
            'label': label,
            'diagnosis': diagnosis
        })

    return subjects

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out

class ResNet3D(nn.Module):
    def __init__(self, num_classes=2, input_shape=(1,110,110,110)): # input: input_shape:	[num_of_filters, kernel_size] (e.g. [256, 25])
        super(ResNet3D, self).__init__()
        #stage 1
        self.conv1 = nn.Sequential(
            nn.Conv3d(
            in_channels=input_shape[0],        
            out_channels=32,       
            kernel_size=(3,3,3),         
            padding=1
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(
            in_channels=32,       
            out_channels=32,      
            kernel_size=(3,3,3),          
            padding=1              
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),                  
            nn.Conv3d(
            in_channels=32,       
            out_channels=64,       
            kernel_size=(3,3,3), 
            stride=2,
            padding=1              
            )
        )
        #stage 2
        self.bot2=Bottleneck(64,64,1)
        #stage 3
        self.bot3=Bottleneck(64,64,1)
        #stage 4
        self.conv4=nn.Sequential(
            nn.BatchNorm3d(64),
            nn.Conv3d(
            in_channels=64,        # input height
            out_channels=64,       # n_filters
            kernel_size=(3,3,3),          # filter size
            padding=1,
            stride=2
            )
        )
        #stage 5
        self.bot5=Bottleneck(64,64,1)
        #stage 6
        self.bot6=Bottleneck(64,64,1)
        #stage 7
        self.conv7=nn.Sequential(
            nn.BatchNorm3d(64),
            nn.Conv3d(
            in_channels=64,        # input height
            out_channels=128,       # n_filters
            kernel_size=(3,3,3),          # filter size
            padding=1,
            stride=2
            )
        )
        #stage 8
        self.bot8=Bottleneck(128,128,1)
        
        #stage 9
        self.bot9=Bottleneck(128,128,1)

        #stage 10
        self.conv10=nn.Sequential(
            nn.MaxPool3d(kernel_size=(7,7,7)))
        
        fc1_output_features=128     
        self.fc1 = nn.Sequential(
             nn.Linear(1024, 128),
             nn.ReLU()
        )

        fc2_output_features=2           
        self.fc2 = nn.Sequential(
        nn.Linear(fc1_output_features, fc2_output_features),
        nn.Sigmoid()
        )

    def forward(self, x, drop_prob=0.8):
        x = self.conv1(x)
        #print(x.shape)  
        x = self.bot2(x)
        #print(x.shape)
        x = self.bot3(x)
        #print(x.shape)
        x = self.conv4(x)
        #print(x.shape) 
        x = self.bot5(x)
        #print(x.shape)
        x = self.bot6(x)
        #print(x.shape)
        x = self.conv7(x)
        #print(x.shape)        
        x = self.bot8(x)
        #print(x.shape) 
        x = self.bot9(x)
        #print(x.shape)
        x = self.conv10(x)
        #print(x.shape) 
        x = x.view(x.size(0), -1) # flatten the output of conv2 to (batch_size, num_filter * w * h)
        #print(x.shape)        
        x = self.fc1(x)
        x = self.fc2(x)
        #prob = self.out(x) # probability
        return x

class Dataset(Dataset):
    def __init__(self, subjects, transform):
        self.subjects = subjects
        self.transform = transform

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, i):
        x = {}
        x.update(self.subjects[i])
        x['image'] = self.transform(x['image'])
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(train_dataloader, val_dataloader):
    def assemble_labels(step, y_true, y_pred, label, out):
        if step==0:
            y_true = label
            y_pred = out
        else:
            y_true = torch.cat((y_true, label), 0)
            y_pred = torch.cat((y_pred, out))
        return y_true, y_pred


    ## Model setup
    cnn_name = 'resnet'
    model_depth = 18 # (18|34|50|101|152|200)
    n_classes = 2  # output classes
    in_channels = 1 # model input channels (1|3)
    ##sample_size = np.prod(IMAGE_SIZE)  # image size, i.e., h*w*d
    IMAGE_SIZE = [256, 256, 48]

    half_res = [dim // 2 for dim in IMAGE_SIZE[:2]]
    half_res.append(IMAGE_SIZE[-1])
    sample_size = np.prod(half_res)  # image size, i.e., h*w*d
    #net = ResNet3D(num_classes=3, input_shape=[1]+sample_size).to(device)
    net = generate_model.main(cnn_name, model_depth, n_classes, in_channels, sample_size)
    # opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)
    opt = torch.optim.AdamW(net.parameters())#, lr=1e-3, weight_decay=1e-3)
    #opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma= 0.985)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(opt, 
                                                #   base_lr=LR, 
                                                #   max_lr=0.001, 
                                                #   step_size_up=100,
                                                #   cycle_momentum=False)
    #loss_fcn = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHTS.to(device))
    loss_fcn = torch.nn.CrossEntropyLoss()

    t = trange(EPOCHS, desc=' ', leave=True)

    train_hist = []
    val_hist = []
    pred_result = []
    old_acc = 0
    old_auc = 0
    test_acc = 0
    best_epoch = 0
    for e in t:    
        y_true = []
        y_pred = []
        
        val_y_true = []
        val_y_pred = []                
        
        train_loss = 0
        val_loss = 0

        # training
        net.train()
        for step, batch in enumerate(train_dataloader):
        #for step, (img, label, _) in enumerate(train_dataloader):
            #img = img.float().to(device)
            img = batch['image'].float().to(device)
            #label = label.long().to(device)
            label = batch['label'].long().to(device)
            opt.zero_grad()
            out = net(img)
            loss = loss_fcn(out, label)

            loss.backward()
            opt.step()
            
            label = label.cpu().detach()
            out = out.cpu().detach()
            y_true, y_pred = assemble_labels(step, y_true, y_pred, label, out)        

            train_loss += loss.item()

        train_loss = train_loss/(step+1)
        acc = float(torch.sum(torch.max(y_pred, 1)[1]==y_true))/ float(len(y_pred))
        auc = metrics.roc_auc_score(y_true, y_pred[:,1])
        f1 = metrics.f1_score(y_true, torch.max(y_pred, 1)[1])
        precision = metrics.precision_score(y_true, torch.max(y_pred, 1)[1])
        recall = metrics.recall_score(y_true, torch.max(y_pred, 1)[1])
        ap = metrics.average_precision_score(y_true, torch.max(y_pred, 1)[1]) #average_precision

        #scheduler.step()

        # val
        net.eval()
        #full_path = []
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
            #for step, (img, label, _) in enumerate(val_dataloader):
                #img = img.float().to(device)
                img = batch['image'].float().to(device)
                label = batch['label'].long().to(device)
                out = net(img)
                loss = loss_fcn(out, label)
                val_loss += loss.item()

                label = label.cpu().detach()
                out = out.cpu().detach()
                val_y_true, val_y_pred = assemble_labels(step, val_y_true, val_y_pred, label, out)
                
                # for item in batch:
                #     full_path.append(item)
                # print('full path:', full_path)
        val_loss = val_loss/(step+1)
        val_acc = float(torch.sum(torch.max(val_y_pred, 1)[1]==val_y_true))/ float(len(val_y_pred))
        val_auc = metrics.roc_auc_score(val_y_true, val_y_pred[:,1])
        val_f1 = metrics.f1_score(val_y_true, torch.max(val_y_pred, 1)[1])
        val_precision = metrics.precision_score(val_y_true, torch.max(val_y_pred, 1)[1])
        val_recall = metrics.recall_score(val_y_true, torch.max(val_y_pred, 1)[1])
        val_ap = metrics.average_precision_score(val_y_true, torch.max(val_y_pred, 1)[1]) #average_precision


        train_hist.append([train_loss, acc, auc])#, f1, precision, recall, ap])
        val_hist.append([val_loss, val_acc, val_auc])#, val_f1, val_precision, val_recall, val_ap])             

        t.set_description("Epoch: %i, train auc: %.4f, train loss: %.4f, train acc: %.4f, val auc: %.4f, val loss: %.4f, val acc: %.4f, test acc: %.4f" 
                          %(e, auc, train_loss, acc, val_auc, val_loss, val_acc, test_acc))

        if(old_acc<val_acc):
            old_acc = val_acc
            best_epoch = e
            test_loss = 0
            test_y_true = val_y_true
            test_y_pred = val_y_pred            

            test_loss = val_loss
            test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1]==test_y_true))/ float(len(test_y_pred))
            test_auc = metrics.roc_auc_score(test_y_true, test_y_pred[:,1])
            test_f1 = metrics.f1_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_precision = metrics.precision_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_recall = metrics.recall_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_ap = metrics.average_precision_score(test_y_true, torch.max(test_y_pred, 1)[1]) #average_precision
            
        if(old_acc==val_acc) and (old_auc<val_auc):
            old_acc = val_acc
            old_auc = val_auc
            best_epoch = e
            test_loss = 0
            test_y_true = val_y_true
            test_y_pred = val_y_pred            

            test_loss = val_loss
            test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1]==test_y_true))/ float(len(test_y_pred))
            test_auc = metrics.roc_auc_score(test_y_true, test_y_pred[:,1])
            test_f1 = metrics.f1_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_precision = metrics.precision_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_recall = metrics.recall_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_ap = metrics.average_precision_score(test_y_true, torch.max(test_y_pred, 1)[1]) #average_precision

            
            test_performance = [best_epoch, test_loss, test_acc, test_auc, test_f1, test_precision, test_recall, test_ap]

    return train_hist, val_hist, test_performance, test_y_true, test_y_pred#, full_path

RESULTS_FILE = 'test.txt'
def write_results(train_results, val_results, fold):
    results = zip(train_results, val_results)

    with open(RESULTS_FILE, 'a') as f:
        for epoch, values in enumerate(results):
            res = str(fold)+','+str(epoch)+','+','.join([str(val) for val in values[0]]+[str(val) for val in values[1]])
            f.write(res+'\n')

def main():
#DATA_PATH = '/data/scratch/gliang/data/adni/ADNI2_MRI_Feature/Alex_Layer-9_DynamicImage'
#FEATURE_SHAPE=(256,5,5)
#print('DATA_PATH:',DATA_PATH)
    set_loc(nc_mci_ad_df)
    get_patient_image_locs()
    drop_bad_scans()

    ## Writeout setup
    #global RESULTS_FILE
    #RESULTS_FILE = f'ADNI_kfold_train_results_{NUM_FOLDS}-folds_{NUM_EPOCHS}-epochs_{BATCH_SIZE}-batch_size.csv'
    #header = ['fold', 'epoch','train_loss','train_acc','test_loss','test_acc','test_auc']
    #with open(RESULTS_FILE, 'w') as f:
        #f.write(','.join(header)+'\n')
    
    torch.manual_seed(SEED)

    # Data setup
    subjects = get_subjects()
    random.seed(SEED)
    random.shuffle(subjects)

    print('\nStarting...')

    N_SAMPLES = len(subjects)

    print('Total subjects:', N_SAMPLES, '\n')

    rescale = tio.RescaleIntensity((0.05, 99.5))
    resample = tio.Resample(1)
    IMAGE_SIZE = [256, 256, 48]
    half_res = [dim // 2 for dim in IMAGE_SIZE[:2]]
    half_res.append(IMAGE_SIZE[-1])
    resize = tio.Resize(half_res)
    # resize = tio.Resize([110,110,110])
    randaffine = tio.RandomAffine(scales=(0.9,1.2),degrees=10, isotropic=True, image_interpolation='nearest')
    flip = tio.RandomFlip(axes=('LR'), p=0.5)
    #pad = tio.CropOrPad(IMAGE_SIZE) # crop/pad to most common shape
    
    train_transform = tio.Compose([resize]) #tio.Compose([rescale, flip, randaffine, resample, resize])
    test_transform = tio.Compose([resize]) #tio.Compose([rescale, resample, resize])

    train_hist = []
    val_hist = []
    test_performance = []
    test_y_true = np.asarray([])
    test_y_pred = np.asarray([])
    #full_path = np.asarray([])

    kf = StratifiedKFold(n_splits=NUM_FOLDS,
                        shuffle=True,
                        random_state=SEED)

    # Loop over folds
    ## Split data into folds
    X = [subject['image'] for subject in subjects]
    y = [subject['label'] for subject in subjects]

    global RESULTS_FILE
    if KSPACE:
        RESULTS_FILE = f'_train_results_kspace_ADvNC_{NUM_FOLDS}-folds_{EPOCHS}-epochs_{BATCH_SIZE}-batch_size.csv'
    elif REDUCED_KSPACE:
        RESULTS_FILE = f'_train_results_ADvNC_{REDUCED_PERCENTAGE}prcnt-reduced-kspace_results_{NUM_FOLDS}-folds_{EPOCHS}-epochs_{BATCH_SIZE}-batch_size.csv'
    elif MRI:
        RESULTS_FILE = f'_train_results_ADvNC_{NUM_FOLDS}-folds_{EPOCHS}-epochs_{BATCH_SIZE}-batch_size.csv'

    #header = ['fold','epoch','train_loss','train_acc','train_auc','test_loss','test_acc','test_auc']
    with open(RESULTS_FILE, 'w') as f:
        f.write('fold,epoch,train_loss,train_acc,train_auc,val_loss,val_acc,val_auc\n')

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        print(f'Fold [{fold + 1}/{NUM_FOLDS}]\n==========================\n')

        train_subjects = [subjects[i] for i in train_idx]
        test_subjects = [subjects[i] for i in test_idx]

        print(f'\tTrain: {Counter([LABELS[subj["label"]] for subj in train_subjects])}')
        print(f'\tTest: {Counter([LABELS[subj["label"]] for subj in test_subjects])}')

        # Create SubjectsDataset for train and test sets
        train_dataset = Dataset(train_subjects, transform=train_transform)
        test_dataset = Dataset(test_subjects, transform=test_transform)

        # Create data loaders for train and test sets
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        print('Training set:', len(train_dataset), 'samples')

        print('Testing set:', len(test_dataset), 'samples')

        print('batch size:', BATCH_SIZE)
        print('length trainloader:', len(train_dataloader))
        print('length testloader:', len(val_dataloader), '\n')

        #model = generate_model.main(cnn_name, model_depth, n_classes, in_channels, sample_size)
        #model.to(device)

        #optimizer = torch.optim.AdamW(model.parameters())#, lr=1e-3, weight_decay=1e-3)
        #criterion = nn.CrossEntropyLoss()
        #metric = AUROC(task='multiclass', num_classes=3) 
    #for i in range(0, 5):
        #print('Train Fold', i)
        
        #TEST_NUM = i
        #TRAIN_LABEL, TEST_LABEL = prep_data(LABEL_PATH, TEST_NUM)
        
        #train_dataset = Dataset(label_file=TRAIN_LABEL)
        #train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        #val_dataset = Dataset(label_file=TEST_LABEL)
        #val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
            
        cur_result = train(train_dataloader, val_dataloader)
        
        train_hist.append(cur_result[0])
        val_hist.append(cur_result[1]) 

        write_results(cur_result[0], cur_result[1], fold)

        test_performance.append(cur_result[2]) 
        test_y_true = np.concatenate((test_y_true, cur_result[3].numpy()))
        if(len(test_y_pred) == 0):
            test_y_pred = cur_result[4].numpy()
        else:
            test_y_pred = np.vstack((test_y_pred, cur_result[4].numpy()))
        #full_path = np.concatenate((full_path, np.asarray(cur_result[5])))
        print('finish\n')

    print(test_performance)

    test_y_true = torch.tensor(test_y_true)
    test_y_pred = torch.tensor(test_y_pred)

    test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1]==test_y_true.long()))/ float(len(test_y_pred))
    test_auc = metrics.roc_auc_score(test_y_true, test_y_pred[:,1])
    test_f1 = metrics.f1_score(test_y_true, torch.max(test_y_pred, 1)[1])
    test_precision = metrics.precision_score(test_y_true, torch.max(test_y_pred, 1)[1])
    test_recall = metrics.recall_score(test_y_true, torch.max(test_y_pred, 1)[1])
    test_ap = metrics.average_precision_score(test_y_true, torch.max(test_y_pred, 1)[1])

    print('ACC %.4f, AUC %.4f, F1 %.4f, Prec %.4f, Recall %.4f, AP %.4f' 
        %(test_acc, test_auc, test_f1, test_precision, test_recall, test_ap))


if __name__ == '__main__':
    main()
