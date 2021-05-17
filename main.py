import json
from typing import Optional
import os
from io import BytesIO
import sys
from os import path
import asyncio

sys.setrecursionlimit(10**6)

from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from PIL import Image
import numpy as np

from src.model import PatternClassifier
from src.setting import Setting

model = PatternClassifier(sample_dir=Setting.sample_dir, model_path=Setting.model, device=Setting.device, max_image_size=Setting.max_image_size)

app = FastAPI()

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


@app.post("/demo-path")
async def demo_pattern(image: str): 
    if not path.exists(image):
        return JSONResponse(jsonable_encoder({'error': f"{image} is not exists"}), status_code=400)
    
    img = Image.open(image).convert('RGB')
    img = img.crop((3200, 1700, 3712, 2212))
    
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
    