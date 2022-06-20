from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import json
import os
import torch
from torchnet.meter import AUCMeter
import wandb


def load_fault_data():
    """
    Load fault_data of bearing from a sample frequency of 12k
    including rolling element, inner race and outer race fault with 7-mil
    14-mil,21-mil diameter, we use DE variable for working
    Reshape the raw data in a format of (samples,6000)
    """

    path = './CRWU_dataset/12kDriveEnd'
    files = os.listdir(path)
    temp = []
    label = []
    for mat in files:
        if ('28' not in mat) and ('.mat' in mat):
            temp1 = sio.loadmat(os.path.join(path, mat))
            for key in temp1.keys():
                if 'DE' in key:
                    temp.append(temp1[key][:120000])
                    if 'B' in mat:
                        if '07' in mat:
                            label.append([0] * 20)
                        if '14' in mat:
                            label.append([1] * 20)
                        if '21' in mat:
                            label.append([2] * 20)
                    if 'IR' in mat:
                        if '07' in mat:
                            label.append([3] * 20)
                        if '14' in mat:
                            label.append([4] * 20)
                        if '21' in mat:
                            label.append([5] * 20)
                    if 'OR' in mat:
                        if '07' in mat:
                            label.append([6] * 20)
                        if '14' in mat:
                            label.append([7] * 20)
                        if '21' in mat:
                            label.append([8] * 20)
    temp = np.asarray(temp)
    data1 = temp.reshape((-1, 6000))
    label1 = np.asarray(label)
    label1 = label1.reshape((-1, 1))
    return data1, label1


def load_normal_data():
    """
    Load normal_data of bearing
    we use DE variable for working
    Reshape the raw data in a format of (samples,6000)
    """

    path = './CRWU_dataset/Normal_Baseline_Data'
    files = os.listdir(path)
    temp = []
    label2 = []
    for mat in files:
        temp1 = sio.loadmat(os.path.join(path, mat))
        for key in temp1.keys():
            if 'DE' in key:
                if 240000 < len(temp1[key]) < 480000:
                    temp.append(temp1[key][:240000])
                if len(temp1[key]) > 480000:
                    temp.append(temp1[key][:480000])
    temp2 = np.concatenate((temp[0], temp[1], temp[2], temp[3]))
    data2 = temp2.reshape((-1, 6000))
    label2 = np.ones((data2.shape[0], 1)) * 9
    return data2, label2


def load_crwu_data():
    """
    combine all data to be a set, split train set and test set
    """

    data1, label1 = load_fault_data()
    data2, label2 = load_normal_data()
    data = np.concatenate((data1, data2))
    label = np.concatenate((label1, label2)).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=42, stratify=label)
    return X_train, X_test, y_train, y_test


class crwu_dataset(Dataset):
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[],
                 log='', flag='', epoch=-1):

        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise

        if self.mode == 'test':
            if dataset == 'crwu':
                _, self.test_data, _, self.test_label = load_crwu_data()
                self.test_label = self.test_label.flatten().tolist()
        else:
            if dataset == 'crwu':
                train_data, _, train_label, _ = load_crwu_data()
                train_label = train_label.flatten().tolist()

            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file, "r"))
            else:  # inject noise
                noise_label = []
                idx = list(range(len(train_data)))
                random.shuffle(idx)
                num_noise = int(self.r * len(train_data))
                noise_idx = idx[:num_noise]
                for i in range(len(train_data)):
                    if i in noise_idx:
                        if noise_mode == 'sym':
                            if dataset == 'crwu':
                                noiselabel = random.randint(0, 9)
                            noise_label.append(noiselabel)
                        elif noise_mode == 'asym':
                            noiselabel = self.transition[train_label[i]]
                            noise_label.append(noiselabel)
                    else:
                        noise_label.append(train_label[i])
                print("save noisy labels to %s ..." % noise_file)
                # print(noise_label)
                json.dump(noise_label, open(noise_file, "w"))

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                    clean = (np.array(noise_label) == np.array(train_label))
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability, clean)
                    auc, _, _ = auc_meter.value()
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n' % (pred.sum(), auc))
                    log.flush()
                    wandb.log({(flag + " AUC"): auc}, step=epoch)

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]
                print("%s data has a size of %d" % (self.mode, len(self.noise_label)))

    def __getitem__(self, index):
        if self.mode == 'labeled':
            series, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            series1 = self.transform(series)
            series2 = self.transform(series)
            return series1, series2, target, prob
        elif self.mode == 'unlabeled':
            series = self.train_data[index]
            series1 = self.transform(series)
            series2 = self.transform(series)
            return series1, series2
        elif self.mode == 'all':
            series, target = self.train_data[index], self.noise_label[index]
            series = self.transform(series)
            return series, target, index
        elif self.mode == 'test':
            series, target = self.test_data[index], self.test_label[index]
            series = self.transform(series)
            return series, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class Jitter:
    def __init__(self, sigma=0.03):
        self.sigma = sigma

    def __call__(self, x):
        return x + np.random.normal(loc=0., scale=self.sigma, size=x.shape)


class Scaling:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, x):
        factor = np.random.normal(loc=1., scale=self.sigma, size=x.shape)
        return np.multiply(x, factor)


class ToTensor:
    def __call__(self, x):
        return torch.from_numpy(x).to(dtype=torch.float32)


class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class crwu_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        self.labeled_dataset = None
        if self.dataset == 'crwu':
            self.transform_train = ComposeTransform([
                Jitter(),
                Scaling(),
                ToTensor(),
            ])
            self.transform_test = ComposeTransform([
                ToTensor(),
            ])

    def run(self, mode, pred=[], prob=[], flag='', epoch=-1):
        if mode == 'warmup':
            all_dataset = crwu_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                       root_dir=self.root_dir, transform=self.transform_train, mode="all",
                                       noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = crwu_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                           root_dir=self.root_dir, transform=self.transform_train,
                                           mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,
                                           log=self.log, flag=flag, epoch=epoch)
            if len(labeled_dataset) == 0:
                labeled_dataset = self.labeled_dataset
            else:
                self.labeled_dataset = labeled_dataset
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = crwu_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                             root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled",
                                             noise_file=self.noise_file, pred=pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = crwu_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                        root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = crwu_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                        root_dir=self.root_dir, transform=self.transform_test, mode='all',
                                        noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
