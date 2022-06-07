from numpy import dtype
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
import cv2

class cityscapes_data(Dataset):
    def __init__(self, imgs, masks):
        self.img_folder = imgs
        self.mask_folder = masks
        self.img_files = [f for _,_,file in os.walk(imgs) for f in file if not f.startswith('.')]
        self.mask_files = [f for _,_,file in os.walk(masks) for f in file if not f.startswith('.')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self,id):
        img_f = os.path.join(self.img_folder,self.img_files[id].split('_')[0],self.img_files[id])
        mask_f = os.path.join(self.mask_folder, img_f.split('/')[-2],img_f.split('/')[-1].split('leftImg8bit')[0]+'gtFine_labelIds.png')
        img = torch.from_numpy(cv2.imread(img_f))
        mask = torch.from_numpy(cv2.imread(mask_f))[:,:,0]
        img = img.to(torch.float).permute(2,0,1)
        mask = mask.to(torch.float).permute(0,1)
        return img, mask

if __name__ == "__main__":

    train_imgs = '/Users/breenda/Desktop/my_unet/data/imgs/train'
    train_masks = '/Users/breenda/Desktop/my_unet/data/masks/train'
    valid_imgs = '/Users/breenda/Desktop/my_unet/data/imgs/valid'
    valid_masks = '/Users/breenda/Desktop/my_unet/data/masks/valid'

    train_dataset = cityscapes_data(train_imgs, train_masks)
    valid_dataset = cityscapes_data(valid_imgs,valid_masks)

    train_loader = DataLoader(cityscapes_data(train_imgs, train_masks),batch_size=1,shuffle=True)
    valid_loader = DataLoader(cityscapes_data(valid_imgs,valid_masks),batch_size=1, shuffle=False)

    for i,m in train_loader:
        print(i,m)