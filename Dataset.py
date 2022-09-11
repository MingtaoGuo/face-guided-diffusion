import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from PIL import Image 
import numpy as np 
import os 
import scipy.io as sio 

class Dataset_celeba(TensorDataset):
    def __init__(self, path, image_size):
        self.path = path
        self.image_size = image_size
        self.datasets = os.listdir(path)
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        img = Image.open(self.path + self.datasets[item]).resize([self.image_size, self.image_size])
        return self.transforms(img), np.zeros([0])

    def __len__(self):
        return len(self.datasets)

class Dataset_celeba_with_label(TensorDataset):
    def __init__(self, img_path, anno_path, image_size):
        self.path = img_path
        self.data_lines = open(anno_path).readlines()
        self.image_size = image_size
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        data = self.data_lines[item].strip().split()
        path = self.path + data[0]
        img = Image.open(path).resize([self.image_size, self.image_size])
        label = int(data[1])
        return self.transforms(img), label

    def __len__(self):
        return len(self.data_lines)

class Dataset_cifar(TensorDataset):
    def __init__(self, path):
         self.data = np.concatenate([sio.loadmat(path + "/data_batch_1.mat")['data'], 
                                     sio.loadmat(path + "/data_batch_2.mat")['data'],
                                     sio.loadmat(path + "/data_batch_3.mat")['data'],
                                     sio.loadmat(path + "/data_batch_4.mat")['data'],
                                     sio.loadmat(path + "/data_batch_5.mat")['data']], axis=0)
         self.label = np.concatenate([sio.loadmat(path + "/data_batch_1.mat")['labels'], 
                                      sio.loadmat(path + "/data_batch_2.mat")['labels'],
                                      sio.loadmat(path + "/data_batch_3.mat")['labels'],
                                      sio.loadmat(path + "/data_batch_4.mat")['labels'],
                                      sio.loadmat(path + "/data_batch_5.mat")['labels']], axis=0)
         self.data = np.reshape(self.data, [-1, 3, 32, 32])
         self.labels = np.reshape(self.data, [-1])

    def __getitem__(self, item):
        data = np.float32(self.data[item] / 127.5 - 1.0)
        label = np.zeros([10], dtype=np.float32)
        label[self.label[item]] = 1
        return data, label

    def __len__(self):
        return self.data.shape[0]

# f = open("/Data_2/gmt/Dataset/list_attr_celeba.txt")
# lines = f.readlines()
# size = lines[0].strip()
# meta = lines[1].strip().split()
# count = {"Black_Hair": 0, "Blond_Hair": 0, "Brown_Hair": 0, "Gray_Hair": 0}

# print(meta.index('Gray_Hair'))
# lines = open("/Data_2/gmt/Dataset/list_attr_celeba.txt").readlines()
# path = "/Data_2/gmt/Dataset/img_align_celeba/"
# lines = open("/Data_2/gmt/Dataset/list_attr_celeba.txt").readlines()
# data_lines = []
# for line in lines[2:]:
#     line = line.split()
#     name = line[0]
#     line = line[1:]
#     if line[8] == '1':
#         data_lines.append([name, 0])
#     elif line[9] == '1':
#         data_lines.append([name, 1])
#     elif line[11] == '1':
#         data_lines.append([name, 2])
#     elif line[17] == '1':
#         data_lines.append([name, 3])

# num = len(data_lines)
# idx = list(range(num))
# np.random.shuffle(idx)
# shuffled_data = []
# for i in idx:
#     shuffled_data.append(data_lines[i])
# val_data_lines = shuffled_data[:10000]
# train_data_lines = shuffled_data[10000:]
# f = open("val.txt", "w")
# for val in val_data_lines:
#     f.write(val[0] + " " + str(val[1]) + "\n")

# f = open("train.txt", "w")
# for val in train_data_lines:
#     f.write(val[0] + " " + str(val[1]) + "\n")
# a = 0

