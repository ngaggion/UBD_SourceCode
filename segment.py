import torch 
from models.unet import UNet
import cv2 
import os 

modelPreds = []
modelNames = []

for i in range(12):
    modelPred = UNet(n_classes=3).to('cuda:0').float()
    modelPred.load_state_dict(torch.load(f"weights/epoch{i}.pt"), strict=False)
    modelPred.eval()
    modelPreds.append(modelPred)
    modelNames.append(f"epoch{i}")
    
splits_M = ["splits/test_images_heart_M.txt", "splits/test_images_lungs_M.txt"]
splits_F = ["splits/test_images_heart_F.txt", "splits/test_images_lungs_F.txt"]

with torch.no_grad():
        
    for split in splits_M:
        files = open(split, "r").readlines()
        
        for file in files:
            file = file.strip()
            image = cv2.imread("../Chest-xray-landmark-dataset/Images/" + file, cv2.IMREAD_GRAYSCALE) / 255.0
            data = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to('cuda:0').float()
            
            for i in range(0, 12):
                
                seg = modelPreds[i](data)[0, :, :, :]
                seg = seg.argmax(dim=0)
                seg = seg.cpu().numpy()
                                
                # Save in Predictions/M/modelName/file
                try:
                    os.makedirs(f"Predictions/M/{modelNames[i]}")
                except:
                    pass
                cv2.imwrite(f"Predictions/M/{modelNames[i]}/{file}", seg)

    for split in splits_F:
        files = open(split, "r").readlines()
        
        for file in files:
            file = file.strip() 
            image = cv2.imread("../Chest-xray-landmark-dataset/Images/" + file, cv2.IMREAD_GRAYSCALE) / 255.0
            data = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to('cuda:0').float()
            
            for i in range(0, 12):
                
                seg = modelPreds[i](data)[0, :, :, :]
                seg = seg.argmax(dim=0)
                seg = seg.cpu().numpy()
                                
                # Save in Predictions/M/modelName/file
                try:
                    os.makedirs(f"Predictions/F/{modelNames[i]}")
                except:
                    pass
                cv2.imwrite(f"Predictions/F/{modelNames[i]}/{file}", seg)
        