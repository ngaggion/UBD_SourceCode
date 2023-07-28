import torch 
from models.unet import UNet
import cv2 
import os 
import numpy as np

modelPreds = []
modelNames = []

names = os.listdir("Training/")

for i in range(len(names)):
    modelPred = UNet(n_classes=2).to('cuda:0').float()
    modelPred.load_state_dict(torch.load(f"Training/{names[i]}/final.pt", map_location=torch.device('cuda:0')), strict=False)
    modelPred.eval()
    modelPreds.append(modelPred)
    modelNames.append(f"{names[i]}")

splits_M = ["splits/test_images_heart_M.txt", "splits/test_images_lungs_M.txt"]
splits_F = ["splits/test_images_heart_F.txt", "splits/test_images_lungs_F.txt"]

with torch.no_grad():
        
    for split in splits_M:
        files = open(split, "r").readlines()
        
        for file in files:
            file = file.strip()
            image = cv2.imread("../Chest-xray-landmark-dataset/Images/" + file, cv2.IMREAD_GRAYSCALE) / 255.0
            data = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to('cuda:0').float()
            
            for i in range(0, len(modelPreds)):
                
                seg = modelPreds[i](data)[0, :, :, :]
                seg = seg.cpu().numpy()
            
                seg_ = np.zeros([1024, 1024])
                
                lungs = seg[0,:,:] > 0.5
                heart = seg[1,:,:] > 0.5
                
                seg_ = 1 * lungs.astype('int') + 2 * heart.astype('int')
                seg_ = np.clip(seg_, 0, 2)
                seg_ = seg_.astype(np.uint8)
                                
                # Save in Predictions/M/modelName/file
                try:
                    os.makedirs(f"Predictions/M/{modelNames[i]}")
                except:
                    pass
                cv2.imwrite(f"Predictions/M/{modelNames[i]}/{file}", seg_)

    for split in splits_F:
        files = open(split, "r").readlines()
        
        for file in files:
            file = file.strip() 
            image = cv2.imread("../Chest-xray-landmark-dataset/Images/" + file, cv2.IMREAD_GRAYSCALE) / 255.0
            data = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to('cuda:0').float()
            
            for i in range(0, len(modelPreds)):
                
                seg = modelPreds[i](data)[0, :, :, :]
                seg = seg.cpu().numpy()
                
                seg_ = np.zeros([1024, 1024])
                
                lungs = seg[0,:,:] > 0.5
                heart = seg[1,:,:] > 0.5
                
                seg_ = 1 * lungs.astype('int') + 2 * heart.astype('int')
                seg_ = np.clip(seg_, 0, 2)
                seg_ = seg_.astype(np.uint8)
                                
                # Save in Predictions/M/modelName/file
                try:
                    os.makedirs(f"Predictions/F/{modelNames[i]}")
                except:
                    pass
                cv2.imwrite(f"Predictions/F/{modelNames[i]}/{file}", seg_)
        