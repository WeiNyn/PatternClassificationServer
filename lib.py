import sys
from typing import List, Dict

import torch
from torch._C import Value

sys.setrecursionlimit(10**6)

from PIL import Image
import numpy as np

import lpips
from src.model import PatternClassifier
from src.setting import Setting

model = PatternClassifier(sample_dir=Setting.sample_dir, model=Setting.model_path, device=Setting.device, max_image_size=Setting.max_image_size)

def detect(images: List, masks: List) -> Dict:
    """detect - detect pattern and color for masked images

    Args:
        images (List): list of Pillow Image
        masks (List): list of Pillow Image

    Returns:
        Dict: result in dict format
    """
    color_code = None
    color_name = None
    pattern_dict = dict()
    
    if len(images) != len(masks):
        raise ValueError('images and masks must have the same lenhgt')
    
    for index, image, mask in zip(range(len(images)), images, masks):
        result = model.detect(image, mask)
        
        pattern = result['pattern']
        color = result['color']
        pattern_id = pattern['label']
        
        pattern_dict[pattern_id] = pattern_dict.get(pattern_id, 0) + 1
        
        if index == len(images)//2:
            color_code = color['rgb']
            color_name = color['color_name']
    
    max_id = max(pattern_dict.keys(), key=lambda x: pattern_dict[x])
    
    return dict(
        function='pattern_matching',
        results=dict(
            pattern_id=max_id,
            color_code=color_code,
            color_name=color_name
        )
    )
    
def change_samples(sample_dir: str):
    """change database for classification

    Args:
        sample_dir (str): sample_dir to images folder (ImageFolder from torchvision format)

    Returns:
    """
    try:
        global model
        lpips_model = model.get_model()
        
        setting = Setting()
        setting.data_module['folder'] = sample_dir
        
        lpips_model = lpips.LPIPS()
        
        model = None
        torch.cuda.empty_cache()
        
        setting.sample_dir = sample_dir
        model = PatternClassifier(sample_dir=setting.sample_dir, model=lpips_model, device=setting.device, max_image_size=setting.max_image_size)
    
    
    except Exception as ex:
        raise RuntimeError(f"Cannot change samples databse: {ex}")