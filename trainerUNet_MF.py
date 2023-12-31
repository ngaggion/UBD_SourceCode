import os
import torch
import torch.nn.functional as F
import argparse
import random

from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from utils import CrossVal
from dataset_2D import LandmarksDataset, ToTensorSeg, RandomScale, AugColor, Rotate

from models.unet import UNet, OneClassDiceLoss
from torch.nn import CrossEntropyLoss, BCELoss

from metrics import dc
import time
import numpy as np

def evalImageMetricsL(output, target):
    dcp = dc(output == 1, target == 1)
    return dcp

def evalImageMetricsLH(output, target):
    dcp = dc(output == 1, target == 1)
    dcc = dc(output == 2, target == 2)
    return dcp, dcc


def randomSample(iter_lungs, iter_heart, c_lungs, c_heart, train_loader_lungs, train_loader_heart, len_lungs, len_heart):
    coin = np.random.uniform(0, 1)

    if coin > 0.5:
        SAMPLE = "LUNGS"
        sample_batched = next(iter_lungs)
        c_lungs += 1
        
        if c_lungs == len_lungs:
            iter_lungs = iter(train_loader_lungs)
            c_lungs = 0
            
    else:
        SAMPLE = "HEART"
        sample_batched = next(iter_heart)
        c_heart += 1
        
        if c_heart == len_heart:
            iter_heart = iter(train_loader_heart)
            c_heart = 0
    
    return SAMPLE, sample_batched, iter_lungs, iter_heart, c_lungs, c_heart

def trainer(train_dataset_lungs, val_dataset_lungs, train_dataset_heart, val_dataset_heart, model, config):
    torch.manual_seed(420)

    dev = str(config['device'])
    
    device = torch.device("cuda:%s"%dev if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)

    train_loader_lungs = torch.utils.data.DataLoader(train_dataset_lungs, batch_size = config['batch_size'], shuffle = True, num_workers = 2)
    val_loader_lungs = torch.utils.data.DataLoader(val_dataset_lungs, batch_size = config['val_batch_size'], num_workers = 0)

    train_loader_heart = torch.utils.data.DataLoader(train_dataset_heart, batch_size = config['batch_size'], shuffle = True, num_workers = 2)
    val_loader_heart = torch.utils.data.DataLoader(val_dataset_heart, batch_size = config['val_batch_size'], num_workers = 0)

    optimizer = torch.optim.Adam(params = model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])

    train_loss_avg = []
    val_loss_avg = []
    val_dicelungs_avg = []
    val_diceheart_avg = []
    val_dicecla_avg = []

    tensorboard = "Training"
        
    folder = os.path.join(tensorboard, config['name'])

    try:
        os.mkdir(folder)
    except:
        pass 

    writer = SummaryWriter(log_dir = folder)  

    best = 0
    suffix = ".pt"
    
    print('Training ...')
    
    dice_loss = OneClassDiceLoss().to(device)
    ce_loss = BCELoss().to(device)
    
    scheduler = StepLR(optimizer, step_size=config['stepsize'], gamma=config['gamma'])

    train_iter_lungs = iter(train_loader_lungs)
    train_iter_heart = iter(train_loader_heart)
    train_c_lungs = 0
    train_len_lungs = len(train_loader_lungs)
    train_c_heart = 0
    train_len_heart = len(train_loader_heart)
        
    for epoch in range(config['epochs']):
        model.train()

        train_loss_avg.append(0)
        num_batches = 0
        
        t = time.time()
                
        for j in range(0, 60):
            SAMPLE, sample_batched, train_iter_lungs, train_iter_heart, train_c_lungs, train_c_heart = randomSample(
                                                                                        train_iter_lungs, train_iter_heart, 
                                                                                        train_c_lungs, train_c_heart, 
                                                                                        train_loader_lungs, train_loader_heart, 
                                                                                        train_len_lungs, train_len_heart)

            image, target = sample_batched['image'].to(device), sample_batched['seg'].to(device)
            
            out = model(image)
            sigmoid = torch.sigmoid(out)

            # backpropagation
            optimizer.zero_grad()

            if SAMPLE == "HEART":
                loss_heart = dice_loss(sigmoid[:,1,:,:], (target == 2).float()) + ce_loss(sigmoid[:,1,:,:], (target == 2).float())
            else:
                loss_heart = 0            

            loss_lungs = dice_loss(sigmoid[:,0,:,:], (target == 1).float()) + ce_loss(sigmoid[:,0,:,:], (target == 1).float())

            loss = loss_lungs + loss_heart

            train_loss_avg[-1] += loss.item()

            loss.backward()
            optimizer.step()

            num_batches += 1
        
        t2 = time.time()
        
        print('Training epoch took %.3f seconds' %(t2-t))

        train_loss_avg[-1] /= num_batches
        num_batches = 0
        num_batches_h = 0

        model.eval()
        val_loss_avg.append(0)
        val_dicelungs_avg.append(0)
        val_diceheart_avg.append(0)
        
        t = time.time()
        with torch.no_grad():
            for sample_batched in val_loader_lungs:
                image, target = sample_batched['image'].to(device), sample_batched['seg'].to(device)

                out = model(image)
                sigmoid = torch.sigmoid(out)
                
                seg = torch.zeros([1024,1024])
                seg[sigmoid[0,0,:,:] > 0.5] = 1
                seg[sigmoid[0,1,:,:] > 0.5] = 2

                dcl = evalImageMetricsL(seg.cpu().numpy(), target[0,:,:].cpu().numpy())
                val_dicelungs_avg[-1] += dcl
                val_loss_avg[-1] += dcl

                num_batches += 1
            
            for sample_batched in val_loader_heart:
                image, target = sample_batched['image'].to(device), sample_batched['seg'].to(device)
                out = model(image)
                sigmoid = F.sigmoid(out)
                
                seg = torch.zeros([1024,1024])
                seg[sigmoid[0,0,:,:] > 0.5] = 1
                seg[sigmoid[0,1,:,:] > 0.5] = 2

                dcl, dch = evalImageMetricsLH(seg.cpu().numpy(), target[0,:,:].cpu().numpy())
                val_dicelungs_avg[-1] += dcl
                val_diceheart_avg[-1] += dch
                val_loss_avg[-1] += (dcl + dch) / 2

                num_batches += 1
                num_batches_h += 1

        val_loss_avg[-1] /= num_batches
        val_dicelungs_avg[-1] /= num_batches
        val_diceheart_avg[-1] /= num_batches_h
        
        t2 = time.time()
        
        print('Epoch [%d / %d] validation Dice: %.3f, took %.3f seconds' % (epoch+1, config['epochs'], val_loss_avg[-1], t2-t))
        print('Dice Lungs %.3f. Dice Heart %.3f' %(val_dicelungs_avg[-1],val_diceheart_avg[-1]))
        
        writer.add_scalar('Train/Loss', train_loss_avg[-1], epoch)
        writer.add_scalar('Validation/Dice', val_loss_avg[-1], epoch)
        writer.add_scalar('Validation/Dice Lungs', val_dicelungs_avg[-1], epoch)
        writer.add_scalar('Validation/Dice Heart', val_diceheart_avg[-1], epoch)
        
        if val_loss_avg[-1] > best:
            best = val_loss_avg[-1]
            print('Model Saved Dice')
            out = "bestDice.pt"
            torch.save(model.state_dict(), os.path.join(folder, out))

        scheduler.step()

        print('')
        
        torch.save(model.state_dict(), os.path.join(folder, "final.pt"))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type=str)    
    parser.add_argument("--load", help="enter the folder where the weights are saved", default = "None", type=str)
    parser.add_argument("--inputsize", default = 1024, type=int)
    parser.add_argument("--epochs", default = 1000, type = int)
    parser.add_argument("--lr", default = 1e-4, type = float)
    parser.add_argument("--stepsize", default = 3000, type = int)
    parser.add_argument("--gamma", default = 0.1, type = float)
    
    ## 5-fold Cross validation fold
    parser.add_argument("--fold", default = 1, type = int)
    parser.add_argument("--device", default = 0, type = int)
    parser.add_argument('--organs', type=str, default = 'LH')
    
    # Define the output: only lungs, or lungs and heart by default
    parser.add_argument('--original', dest='original', action='store_true')
    parser.set_defaults(original = False)
    
    parser.add_argument('--female', dest='female', action='store_true')
    parser.set_defaults(female = False)
    parser.add_argument('--all', dest='all', action='store_true')
    parser.set_defaults(all = False)

    config = parser.parse_args()
    config = vars(config)

    inputSize = config['inputsize']

    if not config['female'] and not config['all']:
        images_lungs = open("splits/train_images_lungs_M.txt",'r').read().splitlines()
        images_heart = open("splits/train_images_heart_M.txt",'r').read().splitlines()
    elif config['female'] and not config['all']:
        images_lungs = open("splits/train_images_lungs_F.txt",'r').read().splitlines()
        images_heart = open("splits/train_images_heart_F.txt",'r').read().splitlines()
    elif config['all']:
        images_lungs = open("splits/train_images_lungs.txt",'r').read().splitlines()
        images_heart = open("splits/train_images_heart.txt",'r').read().splitlines()
    
    print("Lungs:", len(images_lungs))
    random.Random(13).shuffle(images_lungs)
    print("Heart:", len(images_heart))
    random.Random(13).shuffle(images_heart)
        
    print('Fold %s'%config['fold'], 'of 5')
    images_train_L, images_val_L = CrossVal(images_lungs, config['fold'])
    images_train_H, images_val_H = CrossVal(images_heart, config['fold'])
    
    train_dataset_lungs = LandmarksDataset(images=images_train_L,
                                     img_path="../Chest-xray-landmark-dataset/Images",
                                     label_path="../Chest-xray-landmark-dataset/landmarks",
                                     heart = False,
                                     transform = transforms.Compose([
                                                 Rotate(5),
                                                 RandomScale(),
                                                 #AugColor(0.50),
                                                 ToTensorSeg()])
                                     )

    val_dataset_lungs = LandmarksDataset(images=images_val_L,
                                     img_path="../Chest-xray-landmark-dataset/Images",
                                     label_path="../Chest-xray-landmark-dataset/landmarks",
                                     heart = False,
                                     transform = ToTensorSeg()
                                     )

    train_dataset_heart = LandmarksDataset(images=images_train_H,
                                     img_path="../Chest-xray-landmark-dataset/Images",
                                     label_path="../Chest-xray-landmark-dataset/landmarks",
                                     heart = True,
                                     transform = transforms.Compose([
                                                 Rotate(5),
                                                 RandomScale(),
                                                 #AugColor(0.50),
                                                 ToTensorSeg()])
                                     )

    val_dataset_heart = LandmarksDataset(images=images_val_H,
                                     img_path="../Chest-xray-landmark-dataset/Images",
                                     label_path="../Chest-xray-landmark-dataset/landmarks",
                                     heart = True,
                                     transform = ToTensorSeg()
                                     )

    config['latents'] = 64
    config['batch_size'] = 4
    config['val_batch_size'] = 1
    config['weight_decay'] = 1e-5
    
    n_classes = len(config['organs'])

    model = UNet(n_classes = n_classes)    
    trainer(train_dataset_lungs, val_dataset_lungs, train_dataset_heart, val_dataset_heart, model, config)