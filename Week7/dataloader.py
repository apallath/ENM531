import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2

class Cell_dataset(Dataset):
    """Pocked Cell dataset."""
    def __init__(self, file_location):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.location = file_location
        self.filenames = os.listdir(self.location)
        # self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.filenames[idx]
        img = cv2.imread(self.location+name, cv2.IMREAD_GRAYSCALE)
        img = np.array(img)
        img = img.astype('float')
        img = np.reshape(img,(1,200,200))
        nm = name.split("_")
        if nm[0]=="Pocked":
            label = 1
        elif nm[0]=="Unpocked":
            label = 0
        else:
            print("String Parsing Error")
        
        label = np.array(label).astype('float')
        label = np.reshape(label,(1))
        # if self.transform:
        #     train_x = self.transform(train_x)
        #     train_y = self.transform(train_y)

        return torch.from_numpy(img), torch.from_numpy(label)

# Test
if __name__ == '__main__':
    test_dataset = Cell_dataset('test/')
    test_loader = DataLoader(test_dataset,batch_size=2, shuffle=True)
    for idx, data in enumerate(test_loader):
        if idx%500==0:
            print(data)