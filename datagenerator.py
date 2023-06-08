from PIL import Image
import random
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from prefetch_generator import BackgroundGenerator
from settings import Path, Config

class MyDataset(Dataset):
    def __init__(self, mode="train", file_root="", shuffle=True):  # __init__是初始化该类的一些基础参数
        self.mode = mode

        file = open(file_root)
        file_content = file.readlines()

        if shuffle:
            random.shuffle(file_content)

        if self.mode == 'main':
            self.images = [Path.ROOT_PIC_DATASET + '/' + self.mode + '/' + file[:-1] for file in file_content]
        else:
            self.images = [Path.ROOT_PIC_DATASET + '/' + self.mode + '/' + file[:-3] for file in file_content]
            self.labels = [int(file[-2]) for file in file_content]

        print('get ' + self.mode + ' data : ' + str(len(self.images)))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize([512,512]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.Normalize(mean=[0.53308, 0.32708, 0.15955], std=[0.22700, 0.17025, 0.13329])
        ])

        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize([512,512]),
            transforms.Normalize(mean=[0.53308, 0.32708, 0.15955], std=[0.22700, 0.17025, 0.13329])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.mode == 'train':
            img = self.transform(img)
            return img, self.labels[idx]
        elif self.mode == 'valid' or self.mode == 'test':
            img = self.transform2(img)
            return img, self.labels[idx]
        else:
            img = self.transform2(img)
            return img, self.images[idx]

class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def get_data_loader(mode, file_root, shuffle=True):
    dataset = MyDataset(mode=mode, file_root=file_root, shuffle=shuffle)
    dataloader = DataLoaderX(dataset=dataset, shuffle=shuffle, batch_size=Config.BATCH_SIZE,
                             num_workers=Config.WORKERS_NUM,drop_last=True)
    return dataloader
