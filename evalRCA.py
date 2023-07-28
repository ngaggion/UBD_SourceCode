import cv2 
import os 
import numpy as np 
from medpy.metric.binary import dc 
import pandas as pd 
from models.affine import SiameseReg
from models.deformableNet import DeformableNet, GridSampler
import torch

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


def apply_registration_affine(img, params):
    params = params.cpu().numpy()
    affine_matrix = np.zeros((2, 3))
    affine_matrix[0, :] = params[0, 0:3]  # Changed from 0:3 to 0:2
    affine_matrix[1, :] = params[0, 3:6]  # Changed from 3:6 to 2:4
    affine_matrix[:2, 2] = affine_matrix[:2, 2] * 1024
    img = cv2.warpAffine(img.astype('float'), affine_matrix, (img.shape[1], img.shape[0]))

    return img


def apply_registration_deformable(img, params, modelResampler):
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(config['device']).float()
    img = modelResampler(img, params)

    return img.cpu().numpy().squeeze()


def load_models(config):
    modelAffine = SiameseReg(config).float().to(config['device'])
    modelAffine.load_state_dict(torch.load("weights/IndividualRCA/Affine/bestMSE.pt"), strict=False)
    modelAffine.eval()

    modelDeformable = DeformableNet((1024, 1024), batchnorm=True).to(config['device'])
    modelDeformable.load_state_dict(torch.load("weights/IndividualRCA/Deformable/modelDeformable.pt"))
    modelDeformable.eval()

    modelResampler = GridSampler((1024, 1024), mode='nearest').to(config['device'])
    modelResampler.eval()

    modelFinder = SiameseReg(config).float().to(config['device'])
    modelFinder.load_state_dict(torch.load("weights/IndividualRCA/Selector/bestMSE.pt"), strict=False)
    modelFinder.eval()

    return modelAffine, modelDeformable, modelResampler, modelFinder


def calculate_ground_truth(image_names, config, modelAffine, modelDeformable, source, heart):
    gt_params = []
    gt_masks = []

    for img_near in image_names:
        img_target = cv2.imread("../Chest-xray-landmark-dataset/Images/" + img_near, 0) / 255.0
        target = torch.from_numpy(img_target).unsqueeze(0).unsqueeze(0).to(config['device']).float()

        RL_, LL_, H_ = load_landmarks(img_near, heart)
        mask_gt = landmark_to_mask(RL_, LL_, H_)

        params1 = modelAffine(target, source).detach()
        src_affine = apply_registration_affine(source.cpu().numpy().squeeze(), params1)
        src_affine = torch.from_numpy(src_affine).unsqueeze(0).unsqueeze(0).to(config['device']).float()
        params2 = modelDeformable(src_affine, target)[1].detach()

        gt_params = gt_params + [(params1, params2)]
        gt_masks = gt_masks + [mask_gt]

    return gt_params, gt_masks

def find_nearest(mu, latent_matrix, images_train, n_near = 5):
    distances = latent_matrix @ mu.T
    _, sorted_distances_indices = torch.sort(distances, dim=0, descending=True)

    # Select indices of the top 5 nearest images in the latent space
    idxs = sorted_distances_indices[:n_near].squeeze().cpu().numpy()
    target_image_names = [images_train[i] for i in idxs]
    
    return target_image_names

config = {
    'latents': 64,
    'inputsize': 1024,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'sampling': False
}

latent_space = np.load("models/train_space.npy")
latent_matrix = torch.from_numpy(latent_space).to(config['device'])

images_train = open("splits/train_images_lungs.txt", 'r').read().splitlines()
N_Lungs = len(images_train)
images_train += open("splits/train_images_heart.txt", 'r').read().splitlines()

images_train_heart = open("splits/train_images_heart.txt", 'r').read().splitlines()

# Load the models
modelAffine, modelDeformable, modelResampler, modelFinder = load_models(config)

splits_M = ["splits/test_images_heart_M.txt", "splits/test_images_lungs_M.txt"]
splits_F = ["splits/test_images_heart_F.txt", "splits/test_images_lungs_F.txt"]

with torch.no_grad():
    for i in range(0, 2):
        split = splits_F[i]
        
        if i == 0:
            heart = True
            results = pd.DataFrame(columns = ["Model", "File", "Dice Lungs", "Dice Heart"])
            results_max = pd.DataFrame(columns = ["Model", "File", "Dice Lungs", "Dice Heart"])
        else:
            heart = False
            results = pd.DataFrame(columns = ["Model", "File", "Dice Lungs"])
            results_max = pd.DataFrame(columns = ["Model", "File", "Dice Lungs"])
            
        files = open(split, "r").readlines()
        for file in files:
            file = file.strip()
            
            img = cv2.imread(f"../Chest-xray-landmark-dataset/Images/{file}", 0) / 255.0
            
            source = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(config['device']).float()
            # Calculate the latent vector for the current image using the model finder
            mu, _ = modelFinder.encoder(source)
        
            if not heart:
                nearests = find_nearest(mu, latent_matrix, images_train)
            else:
                nearests = find_nearest(mu, latent_matrix[N_Lungs:], images_train_heart)
        
            # Calculate ground truth parameters and masks for the nearest images
            gt_params, gt_masks = calculate_ground_truth(
                nearests, config, modelAffine, modelDeformable, source, heart)
            
            models = os.listdir(f"Predictions/F")
            models = [model for model in models if not model.endswith(".csv")]
            
            for model in models:
                pred = cv2.imread(f"Predictions/F/{model}/{file}", cv2.IMREAD_UNCHANGED)
                    
                rca_dice_lung_list = []
                rca_dice_heart_list = []
                    
                if not heart:
                    for j in range(0, len(gt_params)):
                        params = gt_params[j]
                        mask = gt_masks[j]

                        pred_reg = apply_registration_affine(pred, params[0])
                        pred_reg = apply_registration_deformable(pred_reg, params[1], modelResampler)

                        lung_mask = pred_reg == 1
                
                        rca_dice_lung = dc(lung_mask, mask == 1)
                        rca_dice_lung_list.append(rca_dice_lung)
                    
                    results = results.append({"Model": model, "File": file, "Dice Lungs": np.mean(rca_dice_lung_list)}, ignore_index = True)
                    results_max = results_max.append({"Model": model, "File": file, "Dice Lungs": np.max(rca_dice_lung_list)}, ignore_index = True)
            
                else:
                    for j in range(0, len(gt_params)):
                        params = gt_params[j]
                        mask = gt_masks[j]

                        pred_reg = apply_registration_affine(pred, params[0])
                        pred_reg = apply_registration_deformable(pred_reg, params[1], modelResampler)

                        lung_mask = pred_reg == 1
                        heart_mask = pred_reg == 2
                        rca_dice_lung = dc(lung_mask, mask == 1)
                        rca_dice_heart = dc(heart_mask, mask == 2)
                        rca_dice_lung_list.append(rca_dice_lung)
                        rca_dice_heart_list.append(rca_dice_heart)
                        
                    results = results.append({"Model": model, "File": file, "Dice Lungs": np.mean(rca_dice_lung_list), "Dice Heart": np.mean(rca_dice_heart_list)}, ignore_index = True)
                    results_max = results_max.append({"Model": model, "File": file, "Dice Lungs": np.max(rca_dice_lung_list), "Dice Heart": np.max(rca_dice_heart_list)}, ignore_index = True)
                        
        results.to_csv(f"Predictions/F/rca_{split.split('_')[2].split('.')[0]}.csv", index = False)
        results_max.to_csv(f"Predictions/F/rca_{split.split('_')[2].split('.')[0]}_max.csv", index = False)

    for i in range(0, 2):
        split = splits_M[i]
        
        if i == 0:
            heart = True
            results = pd.DataFrame(columns = ["Model", "File", "Dice Lungs", "Dice Heart"])
            results_max = pd.DataFrame(columns = ["Model", "File", "Dice Lungs", "Dice Heart"])
        else:
            heart = False
            results = pd.DataFrame(columns = ["Model", "File", "Dice Lungs"])
            results_max = pd.DataFrame(columns = ["Model", "File", "Dice Lungs"])
            
        files = open(split, "r").readlines()
        
        for file in files:
            file = file.strip()
            
            img = cv2.imread(f"../Chest-xray-landmark-dataset/Images/{file}", 0) / 255.0
            
            source = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(config['device']).float()
            # Calculate the latent vector for the current image using the model finder
            mu, _ = modelFinder.encoder(source)
        
            if not heart:
                nearests = find_nearest(mu, latent_matrix, images_train)
            else:
                nearests = find_nearest(mu, latent_matrix[N_Lungs:], images_train_heart)
        
            # Calculate ground truth parameters and masks for the nearest images
            gt_params, gt_masks = calculate_ground_truth(
                nearests, config, modelAffine, modelDeformable, source, heart)
            
            models = os.listdir(f"Predictions/M")
            models = [model for model in models if not model.endswith(".csv")]
            
            for model in models:
                pred = cv2.imread(f"Predictions/M/{model}/{file}", cv2.IMREAD_UNCHANGED)
                
                rca_dice_lung_list = []
                rca_dice_heart_list = []
                    
                if not heart:
                    for j in range(0, len(gt_params)):
                        params = gt_params[j]
                        mask = gt_masks[j]

                        pred_reg = apply_registration_affine(pred, params[0])
                        pred_reg = apply_registration_deformable(pred_reg, params[1], modelResampler)

                        lung_mask = pred_reg == 1
                
                        rca_dice_lung = dc(lung_mask, mask == 1)
                        rca_dice_lung_list.append(rca_dice_lung)
                    
                    results = results.append({"Model": model, "File": file, "Dice Lungs": np.mean(rca_dice_lung_list)}, ignore_index = True)
                    results_max = results_max.append({"Model": model, "File": file, "Dice Lungs": np.max(rca_dice_lung_list)}, ignore_index = True)
            
                else:
                    for j in range(0, len(gt_params)):
                        params = gt_params[j]
                        mask = gt_masks[j]

                        pred_reg = apply_registration_affine(pred, params[0])
                        pred_reg = apply_registration_deformable(pred_reg, params[1], modelResampler)

                        lung_mask = pred_reg == 1
                        heart_mask = pred_reg == 2
                        rca_dice_lung = dc(lung_mask, mask == 1)
                        rca_dice_heart = dc(heart_mask, mask == 2)
                        rca_dice_lung_list.append(rca_dice_lung)
                        rca_dice_heart_list.append(rca_dice_heart)
                        
                    results = results.append({"Model": model, "File": file, "Dice Lungs": np.mean(rca_dice_lung_list), "Dice Heart": np.mean(rca_dice_heart_list)}, ignore_index = True)
                    results_max = results_max.append({"Model": model, "File": file, "Dice Lungs": np.max(rca_dice_lung_list), "Dice Heart": np.max(rca_dice_heart_list)}, ignore_index = True)
                        
                        
        results.to_csv(f"Predictions/M/rca_{split.split('_')[2].split('.')[0]}.csv", index = False)
        results_max.to_csv(f"Predictions/M/rca_{split.split('_')[2].split('.')[0]}_max.csv", index = False)

