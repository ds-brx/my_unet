from numpy import dtype
import torch
import os
from unet_model import UNet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from data_load import cityscapes_data
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, device, epochs, loss_f, opt, train_loader, valid_loader,checkpoint_F):
    model.to(device = device)
    model.train()
    for e in range(epochs):
        print("Trainng Epoch {}".format(e))
        for img, mask in tqdm(train_loader):
            img = img.to(device= device, dtype= torch.float32)
            mask = mask.to(device= device, dtype= torch.long)
            pred = model(img)
            loss = loss_f(pred, mask)

            opt.zero_grad()
            loss.backward()
            opt.step()
            print('Epoch {} Train Loss {}'.format(e,loss))

        if e%10 == 0:
            validate_model(model,loss_f,valid_loader)    

        torch.save(model.state_dict(),os.path.join(checkpoint_F, 'model_epoch_{}.pth'.format(e)))


def validate_model(model, loss_f, valid_loader):
    print("Validating")
    model.eval()
    for b, img, mask in tqdm(valid_loader):
        img.to(device)
        mask.to(device)
        pred = model(img)
        loss = loss_f(pred, mask)
        print('Valid Loss {}'.format(loss))


if __name__ == "__main__":

    inp_channels = 3
    out_classes = 34
    epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using Device : ',device)
    train_imgs = '/Users/breenda/Desktop/my_unet/data/imgs/train'
    train_masks = '/Users/breenda/Desktop/my_unet/data/masks/train'
    valid_imgs = '/Users/breenda/Desktop/my_unet/data/imgs/valid'
    valid_masks = '/Users/breenda/Desktop/my_unet/data/masks/valid'
    checkpoint_F = '/Users/breenda/Desktop/my_unet/checkpoints'

    model = UNet(inp_channels, out_classes)
    loss_f = CrossEntropyLoss()
    opt = Adam(model.parameters(), lr = 0.001)

    train_loader = DataLoader(cityscapes_data(train_imgs, train_masks),batch_size=1,shuffle=True)
    valid_loader = DataLoader(cityscapes_data(valid_imgs,valid_masks),batch_size=1, shuffle=False)

    train_model(model, device, epochs, loss_f, opt, train_loader, valid_loader, checkpoint_F)


