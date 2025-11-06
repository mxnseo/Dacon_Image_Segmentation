import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.dataset import SatelliteDataset
from utils.utils import rle_encode
from model.deeplabv3 import get_model

def main():

    model = get_model(num_classes=1, pretrained=False).to(device)
    
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    result = []
    with torch.no_grad():
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            outputs = model(images)
            
            masks = torch.sigmoid(outputs['out']).cpu().numpy() 
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.35).astype(np.uint8) # Threshold = 0.35
            
            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '': 
                    result.append(-1)
                else:
                    result.append(mask_rle)

    # Submission
    submit = pd.read_csv('../data/sample_submission.csv')
    submit['mask_rle'] = result
    submit.to_csv('./deeplabv3_submit.csv', index=False)
    print("Submission file created")

if __name__ == '__main__':
    main()