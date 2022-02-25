from os.path import splitext
from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import os, cv2

class BasicDataset(Dataset):
    def __init__(self, folder, transform):

        self.imgs_dir = f"{folder}/imgs/"
        self.masks_dir = f"{folder}/masks/"
        self.transform = transform

        self.ids = [splitext(file)[0] for file in listdir(self.imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def pre_process(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = img.transpose((2, 0, 1))
        if img.max() > 1:
            img = img / 255    
        return img    

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = f"{self.masks_dir}{idx}.png"
        img_file = f"{self.imgs_dir}{idx}.png"

        img = np.array(Image.open(img_file).convert("RGB"))
        mask = np.array(Image.open(mask_file).convert("L"))
        
        data = {"image": img, "mask": mask}
        augmented = self.transform(**data)
        img, mask = augmented["image"], augmented["mask"] 


        #cv2.imshow("img", img)
        #cv2.waitKey(0)

        # HWC to CHW
        img = self.pre_process(img)
        mask = self.pre_process(mask)

        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        return torch.from_numpy(img).type(torch.FloatTensor), torch.from_numpy(mask).type(torch.FloatTensor)





class BasicDatasetTarget(Dataset):
    def __init__(self, folder, img_size):

        self.imgs_dir = os.path.join(folder, "imgs")
        self.img_size = [img_size[1], img_size[0]]

        self.ids = [splitext(file)[0] for file in listdir(self.imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def pre_process(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = img.transpose((2, 0, 1))
        if img.max() > 1:
            img = img / 255    
        return img    

    def __getitem__(self, i):
        idx = self.ids[i]

        try:
            img_file = os.path.join(self.imgs_dir, idx + ".png")
            img = Image.open(img_file).convert("RGB")
        except OSError:
            img_file = os.path.join(self.imgs_dir, idx + ".jpg")
            img = Image.open(img_file).convert("RGB")          

        img = img.resize(self.img_size)
        img = np.array(img)
        
        # HWC to CHW
        img = self.pre_process(img)

        return torch.from_numpy(img).type(torch.FloatTensor)
                



class BasicDatasetTargetPseudo(Dataset):
    def __init__(self, folder, pseudo_folder, img_size):

        self.imgs_dir = f"{folder}/imgs/"
        self.pseudo_dir = pseudo_folder
        self.img_size = [img_size[1], img_size[0]]

        self.ids = [splitext(file)[0] for file in listdir(self.imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def pre_process(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = img.transpose((2, 0, 1))
        if img.max() > 1:
            img = img / 255    
        return img    

    def __getitem__(self, i):
        idx = self.ids[i]
        try:
            img_file = os.path.join(self.imgs_dir, idx + ".png")
            img = Image.open(img_file).convert("RGB")
        except OSError:
            img_file = os.path.join(self.imgs_dir, idx + ".jpg")
            img = Image.open(img_file).convert("RGB")          


        pseudo_file = os.path.join(self.pseudo_dir, idx + ".png")

        img = img.resize(self.img_size)
        img = np.array(img)

        mask = Image.open(pseudo_file).convert("L")
        mask = mask.resize(self.img_size)
        mask = np.array(mask)
        
        # HWC to CHW
        img = self.pre_process(img)
        mask = self.pre_process(mask)

        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        return torch.from_numpy(img).type(torch.FloatTensor), torch.from_numpy(mask).type(torch.FloatTensor)
                
