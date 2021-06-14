import json
from typing import Optional, List
import os
from io import BytesIO
import sys
from os import path
import asyncio
import lpips
from pydantic.types import Json
from time import time

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, early_stopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms

sys.setrecursionlimit(10**6)

from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from PIL import Image
import numpy as np

from src.model import PatternClassifier
from src.setting import Setting
from src.LPIPSTrainer.src.model.model import PatternModel
from src.LPIPSTrainer.src.data.data_module import PatternDataModule

model = PatternClassifier(sample_dir=Setting.sample_dir, model=Setting.model_path, device=Setting.device, max_image_size=Setting.max_image_size)

app = FastAPI()


@app.post("/demo-pipeline")
async def demo_pipeline(images: List[UploadFile] = File(...), masks: List[UploadFile] = File(...)):
    star_time = time()
    if len(images) != len(masks):
        return JSONResponse(jsonable_encoder({'error': f'Number of masks and images must be the same'}), status_code=400)
    
    try:
        
        images = [Image.open(BytesIO(await image.read())).convert('RGB') for image in images]
        masks = [Image.open(BytesIO(await mask.read())) for mask in masks]
        
    except Exception as ex:
        return JSONResponse(jsonable_encoder({'error': f"Upload file error {ex}"}), status_code=400)
    
    color_code = None
    color_name = None
    pattern_dict = dict()
    
    for index, image, mask in zip(range(len(images)), images, masks):
        result = await model(image, mask)
        
        pattern = result['pattern']
        color = result['color']
        pattern_id = pattern['label']
        
        pattern_dict[pattern_id] = pattern_dict.get(pattern_id, 0) + 1
        
        if index == len(images)//2:
            color_code = color['rgb']
            color_name = color['color_name']
    
    print(pattern_dict)   
    max_id = max(pattern_dict.keys(), key=lambda x: pattern_dict[x])
    
    print(time() - star_time)
    
    return dict(
        function='pattern_matching',
        results=dict(
            pattern_id=max_id,
            color_code=color_code,
            color_name=color_name
        )
    )
    

class Data(BaseModel):
    image: str
    image_mask: str
    output_dir: str


@app.post("/demo")
async def demo_pattern(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert('RGB')
    
    except Exception as ex:
        return JSONResponse(jsonable_encoder({'error': f"Upload file error {ex}"}), status_code=400)
    
    w, h = img.size
    
    left = int(3200.0/4032.0*w)
    right = int(3712.0/4032.0*w)
    top = int(1700.0/3024.0*h)
    bottom = int(2212.0/3024.0*h)

    img = img.crop((left, top, right, bottom))    
    
    try:
        result = await model(img, None)
    
    except Exception as ex:
        return JSONResponse(jsonable_encoder({'error': f"Error in detection: {ex}"}), status_code=500)
    
    pattern = result['pattern']
    color = result['color']
    
    result = dict(
        function='pattern_matching',
        results=dict(
            pattern_id=pattern['label'],
            color_code=color['rgb'],
            color_name=color['color_name'],
        )
    )
    
    return JSONResponse(jsonable_encoder(result), status_code=200)

class DataImage(BaseModel):
    image: str

@app.post("/demo-path")
async def demo_pattern(data: DataImage):
    image = data.image 
    if not path.exists(image):
        return JSONResponse(jsonable_encoder({'error': f"{image} is not exists"}), status_code=400)
    
    img = Image.open(image).convert('RGB')
    # img = img.crop((3200, 1700, 3712, 2212))
    
    tfs = transforms.RandomCrop(512)
    img = tfs(img)
    
    try:
        result = await model(img, None)
    
    except Exception as ex:
        return JSONResponse(jsonable_encoder({'error': f"Error in detection: {ex}"}), status_code=500)
    
    pattern = result['pattern']
    color = result['color']
    
    result = dict(
        function='pattern_matching',
        results=dict(
            pattern_id=pattern['label'],
            color_code=color['rgb'],
            color_name=color['color_name'],
        )
    )
    
    return JSONResponse(jsonable_encoder(result), status_code=200)


@app.post("/")
async def pattern(data: Data):
    image = data.image
    image_mask = data.image_mask
    output_dir = data.output_dir
    
    if not path.exists(image):
        return JSONResponse(jsonable_encoder({'error': f"{image} is not exists"}), status_code=400)
    
    if len(image_mask) > 0 and not path.exists(image_mask):
        return JSONResponse(jsonable_encoder({'error': f"{image_mask} is not exists"}), status_code=400)
    
    if len(output_dir) > 0 and not path.exists(output_dir):
        return JSONResponse(jsonable_encoder({'error': f"{output_dir} is not exists"}), status_code=400)
    
    
    img = Image.open(image).convert('RGB')
    
    if len(image_mask) > 0:
        mask = Image.open(image_mask)
    
    else:
        mask = None
    
    result = await model(img, mask)
    
    pattern = result['pattern']
    color = result['color']
    
    if len(output_dir) > 0:
        loop = asyncio.get_event_loop()
        loop.create_task(make_image(pattern, color, output_dir, image))
    
    result = dict(
        function='pattern_matching',
        results=dict(
            pattern_id=pattern['label'],
            color_code=color['rgb'],
            color_name=color['color_name'],
            image=path.join(output_dir, image.split("/")[-1])
        )
    )
    
    return JSONResponse(jsonable_encoder(result), status_code=200)


async def make_image(pattern, color, output_dir, image_path):
    select_pattern = pattern['result']
    
    color_code = color['rgb']
    
    select_pattern = Image.fromarray(np.uint8(select_pattern)).convert('RGB')
    
    color_image = [[color_code for _ in range(200)] for _ in range(200)]
    color_image = np.array(color_image)
    color_image = Image.fromarray(np.uint8(color_image)).convert('RGB')    
    
    image = Image.open(image_path).convert('RGB')
    image.paste(select_pattern.resize((200, 200)))
    image.paste(color_image, (200, 0))
    
    save_path = path.join(output_dir, image_path.split("/")[-1])
    image.save(save_path)
    
    
async def train(model, setting: Setting):
    model_config = setting.model
    pattern_config = {k: v for k, v in model_config.items() if k not in ['net', 'version']}
    
    model = PatternModel(model=model,
                         **pattern_config)
    
    
    data_module_config = setting.data_module
    data_module = PatternDataModule(**data_module_config)
    
    checkpoint_config = setting.checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best-checkpoint',
        verbose=True,
        mode='min',
        **checkpoint_config
    )
    
    logger = TensorBoardLogger('lightning_logs', name=setting.logger['name'])
    
    early_stopping_config = setting.early_stopping
    es_enable = early_stopping_config['enable']
    if es_enable is True:
        early_stopping_callback = EarlyStopping(**{k: v for k, v in early_stopping_config.items() if k != 'enable'})
    
    
    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback] if es_enable is True else None,
        progress_bar_refresh_rate=30,
        **setting.trainer
    )
    
    trainer.fit(model, data_module)
    
    return model.model

@app.post('/train')
async def train_route(path: str):
    try:
        global model
        lpips_model = model.get_model()
        
        setting = Setting()
        setting.data_module['folder'] = path
        
        if setting.run_train is True:
            lpips_model = await train(lpips_model, setting)
        else:
            lpips_model = lpips.LPIPS()
        
        model = None
        torch.cuda.empty_cache()
        
        setting.sample_dir = path
        model = PatternClassifier(sample_dir=setting.sample_dir, model=lpips_model, device=setting.device, max_image_size=setting.max_image_size)
    
    except Exception as ex:
        return JSONResponse(jsonable_encoder({'error': f"Error in training model {ex}"}), status_code=500)
    
    return JSONResponse(jsonable_encoder({'status': 'successfull'}), status_code=200)