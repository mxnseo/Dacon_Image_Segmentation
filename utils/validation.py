import torch
import torch.nn.functional as F
from tqdm import tqdm

def dice_score(preds, masks, smooth=1e-6):
    """
    Dice Score (Dice Coefficient) 계산 함수
    preds: 모델 예측값 (0 또는 1)
    masks: 실제 마스크 (0 또는 1)
    """
    preds = preds.contiguous().view(-1)
    masks = masks.contiguous().view(-1)
    
    intersection = (preds * masks).sum()
    total_pixels = preds.sum() + masks.sum()
    
    dice = (2. * intersection + smooth) / (total_pixels + smooth)
    return dice.item()

def evaluate(model, dataloader, device):
    """
    모델을 평가하고 평균 Dice Score를 반환하는 함수
    """
    model.eval() # 모델을 평가 모드로 설정
    total_dice = 0.0
    
    with torch.no_grad(): # Gradient 계산 비활성화
        pbar = tqdm(dataloader, desc="Validating", leave=False, dynamic_ncols=True)
        
        for images, masks in pbar:
            images = images.float().to(device)
            masks = masks.float().to(device) # (B, H, W)

            # 추론
            outputs = model(images)
            
            # DeepLabv3는 딕셔너리('out')로 반환
            logits = outputs['out'] # (B, 1, H, W)
            
            # Sigmoid + Threshold (0.5)
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float() # (B, 1, H, W)
            
            # 마스크 형태 (B, H, W) -> (B, 1, H, W)로 통일
            masks_unsqueezed = masks.unsqueeze(1)
            
            # 배치 Dice Score 계산
            batch_dice = dice_score(preds, masks_unsqueezed)
            total_dice += batch_dice
            
            pbar.set_postfix({'val_dice_batch': f'{batch_dice:.4f}'})

    model.train() # 모델을 다시 학습 모드로 설정
    
    avg_dice = total_dice / len(dataloader)
    return avg_dice