import os 

import torch
from skimage import io, transform

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 

class LandmarksDataset(Dataset):
    def __init__(self, images, img_path, label_path, transform=None, heart = False):
        
        self.images = images
        self.img_path = img_path
        self.label_path = label_path
        
        self.RL_path = os.path.join(self.label_path, 'RL')
        self.LL_path = os.path.join(self.label_path, 'LL')
        
        if heart:
            self.H_path = os.path.join(self.label_path, 'H')
            
        self.heart = heart

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        
        img_path = os.path.join(self.img_path, img_name)
                
        RL_path = os.path.join(self.RL_path, img_name.replace('.png', '.npy'))       
        LL_path = os.path.join(self.LL_path, img_name.replace('.png', '.npy'))
                         
        image = cv2.imread(img_path, 0).astype('float') / 255.0
                
        RL = np.load(RL_path).astype('float')
        LL = np.load(LL_path).astype('float')
        
        if self.heart: 
            H_path = os.path.join(self.H_path, img_name.replace('.png', '.npy'))
            H = np.load(H_path).astype('float')
            landmarks = np.concatenate([RL, LL, H], axis = 0)
        else:
            landmarks = np.concatenate([RL, LL], axis = 0)
                
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

    
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return np.float32(cv2.LUT(image.astype('uint8'), table))

class AugColor(object):
    def __init__(self, gammaFactor):
        self.gammaf = gammaFactor

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        # Gamma
        gamma = np.random.uniform(1 - self.gammaf, 1 + self.gammaf / 2)
        
        image = adjust_gamma(image * 255, gamma) / 255
        
        # Adds a little noise
        image = image + np.random.normal(0, 1/128, image.shape)
        
        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
                
        h, w = image.shape[:2]
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)
        else:
            image = np.transpose(image, (2, 0, 1))
        
        landmarks[:,0] /= w
        landmarks[:,1] /= h 
        
        # concatenate a vectors of 1 to the landmarks 
        landmarks = np.concatenate((landmarks, np.ones((landmarks.shape[0], 1))), axis=1)
        
        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks).float()}
   
def getDenseMask(RL, LL, H = None):
    img = np.zeros([1024,1024])
    
    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')

    img = cv2.drawContours(img, [RL], -1, 1, -1)
    img = cv2.drawContours(img, [LL], -1, 1, -1)
    
    if H is not None:
        H = H.reshape(-1, 1, 2).astype('int')
        img = cv2.drawContours(img, [H], -1, 2, -1)
        
    return img

class ToTensorSeg(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        h, w = image.shape[:2]
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)
        else:
            image = np.transpose(image, (2, 0, 1)) 

        RL = landmarks[:44,:]
        LL = landmarks[44:94,:]

        if landmarks.shape[0] == 94:
            seg = getDenseMask(RL, LL)
        elif landmarks.shape[0] == 120:
            H = landmarks[94:120,:]
            seg = getDenseMask(RL, LL, H)

        return {'image': torch.from_numpy(image).float(),
                'seg': torch.from_numpy(seg).long()}
    
    
class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
                
        # Get a random angle on a normal distribution, with the given standard deviation
        angle = np.random.normal(0, self.angle / 3)
        
        # Compute the padding size based on the image diagonal length
        h, w = image.shape[:2]
        diagonal = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))
        pad_x = (diagonal - w) // 2
        pad_y = (diagonal - h) // 2

        # Pad the image
        padded_image = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
        padded_h, padded_w = padded_image.shape[:2]

        # Rotate the padded image
        center = (padded_w // 2, padded_h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_image = cv2.warpAffine(padded_image, rotation_matrix, (padded_w, padded_h))
        
        # Compute the rotated landmarks
        landmarks += np.array([pad_x, pad_y])  # Account for padding
        ones = np.ones(shape=(len(landmarks), 1))
        landmarks_hom = np.hstack([landmarks, ones])
        rotated_landmarks = np.dot(rotation_matrix, landmarks_hom.T).T

        # Calculate the bounding box that includes all the landmarks
        x_min, y_min = np.min(rotated_landmarks, axis=0).astype(int)
        x_max, y_max = np.max(rotated_landmarks, axis=0).astype(int)

        extra_width = 1024 - (x_max - x_min)
        extra_height = 1024 - (y_max - y_min)
        
        if extra_width > 0:
            x_min -= extra_width // 2
            x_max += extra_width // 2 + extra_width % 2
        
        if extra_height > 0:
            y_min -= extra_height // 2
            y_max += extra_height // 2 + extra_height % 2
                
        # Check if the bounding box is square, if it is not, make it square and centered
        
        if (x_max - x_min) > (y_max - y_min):
            extra = (x_max - x_min) - (y_max - y_min)
            y_min -= extra // 2
            y_max += extra // 2 + extra % 2
        else:
            extra = (y_max - y_min) - (x_max - x_min)
            x_min -= extra // 2
            x_max += extra // 2 + extra % 2
       
        # Crop the rotated image to the bounding box size while keeping all landmarks
        image = rotated_image[y_min:y_max, x_min:x_max]
        landmarks = rotated_landmarks - np.array([x_min, y_min])

        h, w = image.shape[:2]
        if h != w:
            print(image.shape)
            raise ValueError('Image is not square')
        
        if h > 1024:
            image = cv2.resize(image, (1024, 1024))
            landmarks *= 1024 / h
                
        return {'image': image, 'landmarks': landmarks}

    
class RandomScale(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']       
                
        # Pongo limites para evitar que los landmarks salgan del contorno
        min_x = np.min(landmarks[:,0]) 
        max_x = np.max(landmarks[:,0])
        ancho = max_x - min_x
        
        min_y = np.min(landmarks[:,1])
        max_y = np.max(landmarks[:,1])
        alto = max_y - min_y
        
        max_var_x = 1024 / ancho 
        max_var_y = 1024 / alto
                
        min_var_x = 0.80
        min_var_y = 0.80
                                
        varx = np.random.uniform(min_var_x, max_var_x)
        vary = np.random.uniform(min_var_x, max_var_y)
                
        landmarks[:,0] = landmarks[:,0] * varx
        landmarks[:,1] = landmarks[:,1] * vary
        
        h, w = image.shape[:2]
        new_h = np.round(h * vary).astype('int')
        new_w = np.round(w * varx).astype('int')

        img = transform.resize(image, (new_h, new_w))
        
        # Cropeo o padeo aleatoriamente
        min_x = np.round(np.min(landmarks[:,0])).astype('int')
        max_x = np.round(np.max(landmarks[:,0])).astype('int')
        
        min_y = np.round(np.min(landmarks[:,1])).astype('int')
        max_y = np.round(np.max(landmarks[:,1])).astype('int')
        
        if new_h > 1024:
            rango = 1024 - (max_y - min_y)
            maxl0y = new_h - 1025
            
            if rango > 0 and min_y > 0:
                l0y = min_y - np.random.randint(0, min(rango, min_y))
                l0y = min(maxl0y, l0y)
            else:
                l0y = min_y
                
            l1y = l0y + 1024
            
            img = img[l0y:l1y,:]
            landmarks[:,1] -= l0y
            
        elif new_h < 1024:
            pad = h - new_h
            p0 = np.random.randint(np.floor(pad/4), np.ceil(3*pad/4))
            p1 = pad - p0
            
            img = np.pad(img, ((p0, p1), (0, 0)), mode='constant', constant_values=0)
            landmarks[:,1] += p0
        
        if new_w > 1024:
            rango = 1024 - (max_x - min_x)
            maxl0x = new_w - 1025
            
            if rango > 0 and min_x > 0:
                l0x = min_x - np.random.randint(0, min(rango, min_x))
                l0x = min(maxl0x, l0x)
            else:
                l0x = min_x
            
            l1x = l0x + 1024
                
            img = img[:, l0x:l1x]
            landmarks[:,0] -= l0x
            
        elif new_w < 1024:
            pad = w - new_w
            p0 = np.random.randint(np.floor(pad/4), np.ceil(3*pad/4))
            p1 = pad - p0
            
            img = np.pad(img, ((0, 0), (p0, p1)), mode='constant', constant_values=0)
            landmarks[:,0] += p0
        
        if img.shape[0] != 1024 or img.shape[1] != 1024:
            print('Original', [new_h,new_w])
            print('Salida', img.shape)
            raise Exception('Error')
                
            
        return {'image': img, 'landmarks': landmarks}