import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
def tensor_to_pil_images(tensor):
    """
    image tensor to pil image list
    """
    tensor = tensor.float().cpu()
    N = tensor.shape[0]
    pil_images = []
    
    for i in range(N):
        img_tensor = tensor[i] * 255.0
        img_array = img_tensor.numpy().astype(np.uint8)
        if img_array.shape[-1] == 3:
            pil_img = Image.fromarray(img_array, mode='RGB')
        elif img_array.shape[-1] == 4:
            pil_img = Image.fromarray(img_array, mode='RGBA')
        
        elif img_array.shape[-1] == 1:
            img_array = img_array.squeeze(-1)
            pil_img = Image.fromarray(img_array, mode='L')
        else:
            raise ValueError(f"not support color channel: {img_array.shape[-1]}")
        
        pil_images.append(pil_img)
    
    return pil_images


def resize_tensor_images(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Resize tensor images with shape (N, H, W, C) to a target resolution while maintaining the aspect ratio.
    """
    tensor = tensor.cpu()
    N, H, W, C = tensor.shape
    
    if H >= W:
        new_H = target_size
        new_W = int(W * (target_size / H))
    else:
        new_W = target_size
        new_H = int(H * (target_size / W))
        
    resize_transform = transforms.Resize((new_H, new_W), interpolation=transforms.InterpolationMode.BILINEAR)
    resized_tensors = []
    
    for i in range(N):
        img_tensor = tensor[i].permute(2, 0, 1)  # 调整通道顺序为 (C, H, W)
        resized_img = resize_transform(img_tensor.unsqueeze(0)).squeeze(0)
        resized_img = resized_img.permute(1, 2, 0)  # 恢复通道顺序为 (H, W, C)
        resized_tensors.append(resized_img)
    
    resized_tensor = torch.stack(resized_tensors)
    
    return resized_tensor