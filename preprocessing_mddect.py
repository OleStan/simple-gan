import torch
import numpy as np
import os
import matplotlib.pyplot as plt


class DatasetNpy():
    def __init__(self, npy_file, target_length=1024, classes=None, angles=None, channel=0):
        self.npy_file = npy_file
        self.target_length = target_length
        self.classes = classes
        self.angles = angles
        self.channel = channel
        self.dataset = self.build_dataset()
        self.length = self.dataset.shape[1]
        self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.dataset[:, idx]
        step = torch.unsqueeze(step, 0)
        target = 0
        return step, target

    def build_dataset(self):
        data = np.load(self.npy_file)
        
        if self.classes is None:
            self.classes = list(range(data.shape[4]))
        if self.angles is None:
            self.angles = list(range(data.shape[1]))
        
        signals = []
        for exp in range(data.shape[0]):
            for angle in self.angles:
                for direction in range(data.shape[2]):
                    for repeat in range(data.shape[3]):
                        for class_idx in self.classes:
                            signal = data[exp, angle, direction, repeat, class_idx, :, self.channel]
                            
                            if len(signal) < self.target_length:
                                signal = np.pad(signal, (0, self.target_length - len(signal)), mode='edge')
                            elif len(signal) > self.target_length:
                                signal = signal[:self.target_length]
                            
                            signals.append(signal)
        
        dataset = np.vstack(signals).T
        dataset = torch.from_numpy(dataset).float()
        return dataset

    def minmax_normalize(self):
        for index in range(self.length):
            self.dataset[:, index] = (self.dataset[:, index] - self.dataset[:, index].min()) / (
                self.dataset[:, index].max() - self.dataset[:, index].min())


class Dataset():
    def __init__(self, root, target_length=1024):
        self.root = root
        self.target_length = target_length
        self.dataset = self.build_dataset()
        self.length = self.dataset.shape[1]
        self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.dataset[:, idx]
        step = torch.unsqueeze(step, 0)
        target = 0
        return step, target

    def build_dataset(self):
        '''get dataset of signal'''
        dataset = []
        for _file in os.listdir(self.root):
            if not _file.endswith('.txt'):
                continue
            sample = np.loadtxt(os.path.join(self.root, _file)).T
            
            if len(sample) == 1250:
                sample = sample[:self.target_length]
            elif len(sample) < self.target_length:
                sample = np.pad(sample, (0, self.target_length - len(sample)), mode='edge')
            elif len(sample) > self.target_length:
                sample = sample[:self.target_length]
            
            dataset.append(sample)
        dataset = np.vstack(dataset).T
        dataset = torch.from_numpy(dataset).float()

        return dataset

    def minmax_normalize(self):
        '''return minmax normalize dataset'''
        for index in range(self.length):
            self.dataset[:, index] = (self.dataset[:, index] - self.dataset[:, index].min()) / (
                self.dataset[:, index].max() - self.dataset[:, index].min())


if __name__ == '__main__':
    dataset = Dataset('./data/mddect')
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Signal shape: {dataset.dataset.shape}")
    plt.plot(dataset.dataset[:, 0].T)
    plt.title('Sample MDDECT Signal')
    plt.xlabel('Time Points')
    plt.ylabel('Normalized Amplitude')
    plt.grid(True, alpha=0.3)
    plt.show()
