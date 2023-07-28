import cv2 
import os 
import numpy as np 
from medpy.metric.binary import dc 
import pandas as pd 

def load_landmarks(path, heart = False):
    RL_path = "../Chest-xray-landmark-dataset/landmarks/RL/" + path.replace("png", "npy")
    LL_path = "../Chest-xray-landmark-dataset/landmarks/LL/" + path.replace("png", "npy")
    RL = np.load(RL_path)
    LL = np.load(LL_path)
    
    if heart:
        H_path = "../Chest-xray-landmark-dataset/landmarks/H/" + path.replace("png", "npy")
        H = np.load(H_path)
    else:
        H = None
        
    return RL, LL, H


def landmark_to_mask(RL, LL, H):
    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')
    
    mask = np.zeros((1024, 1024))
    mask = cv2.drawContours(mask, [RL], -1, 1, -1)
    mask = cv2.drawContours(mask, [LL], -1, 1, -1)
    
    if H is not None:
        H = H.reshape(-1, 1, 2).astype('int')
        mask = cv2.drawContours(mask, [H], -1, 2, -1)

    return mask

splits_M = ["splits/test_images_heart_M.txt", "splits/test_images_lungs_M.txt"]
splits_F = ["splits/test_images_heart_F.txt", "splits/test_images_lungs_F.txt"]

for i in range(0, 2):
    split = splits_F[i]
    
    if i == 0:
        heart = True
        results = pd.DataFrame(columns = ["Model", "File", "Dice Lungs", "Dice Heart"])
    else:
        heart = False
        results = pd.DataFrame(columns = ["Model", "File", "Dice Lungs"])
        
    files = open(split, "r").readlines()
    for file in files:
        file = file.strip()
        landmarks = load_landmarks(file, heart = heart)
        mask = landmark_to_mask(*landmarks)
        
        models = os.listdir(f"Predictions/F")
        models = [model for model in models if not model.endswith(".csv")]
        
        for model in models:
            prediction = cv2.imread(f"Predictions/F/{model}/{file}", cv2.IMREAD_UNCHANGED)
            
            if not heart:
                dice = dc(prediction == 1, mask == 1)   
                results = results.append({"Model": model, "File": file, "Dice Lungs": dice}, ignore_index = True)
            else:
                dice_lungs = dc(prediction == 1, mask == 1)
                dice_heart = dc(prediction == 2, mask == 2)
                results = results.append({"Model": model, "File": file, "Dice Lungs": dice_lungs, "Dice Heart": dice_heart}, ignore_index = True)
    
    results.to_csv(f"Predictions/F/{split.split('_')[2].split('.')[0]}.csv", index = False)

for i in range(0, 2):
    split = splits_M[i]
    
    if i == 0:
        heart = True
        results = pd.DataFrame(columns = ["Model", "File", "Dice Lungs", "Dice Heart"])
    else:
        heart = False
        results = pd.DataFrame(columns = ["Model", "File", "Dice Lungs"])
        
    files = open(split, "r").readlines()
    for file in files:
        file = file.strip()
        landmarks = load_landmarks(file, heart = heart)
        mask = landmark_to_mask(*landmarks)
        
        models = os.listdir(f"Predictions/M")
        # ignore csv files
        models = [model for model in models if not model.endswith(".csv")]
        
        for model in models:
            prediction = cv2.imread(f"Predictions/M/{model}/{file}", cv2.IMREAD_UNCHANGED)
            
            if not heart:
                dice = dc(prediction == 1, mask == 1)   
                results = results.append({"Model": model, "File": file, "Dice Lungs": dice}, ignore_index = True)
            else:
                dice_lungs = dc(prediction == 1, mask == 1)
                dice_heart = dc(prediction == 2, mask == 2)
                results = results.append({"Model": model, "File": file, "Dice Lungs": dice_lungs, "Dice Heart": dice_heart}, ignore_index = True)
    
    results.to_csv(f"Predictions/M/{split.split('_')[2].split('.')[0]}.csv", index = False)
