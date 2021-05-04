from typing import Optional
import os
import sys
from os import path
import asyncio

sys.setrecursionlimit(10**6)

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from PIL import Image
import numpy as np

from src.model import PatternClassifier

model = PatternClassifier(sample_dir='image_120_wider_sub/', model_path='model/60_net_.pth')

app = FastAPI()

class Data(BaseModel):
    image: str
    image_mask: str
    output_dir: str


@app.post("/")
async def pattern(data: Data):
    image = data.image
    image_mask = data.image_mask
    output_dir = data.output_dir
    
    if not path.exists(image):
        return JSONResponse(jsonable_encoder({'error': f"{image} is not exists"}), status_code=400)
    
    if not path.exists(image_mask):
        return JSONResponse(jsonable_encoder({'error': f"{image_mask} is not exists"}), status_code=400)
    
    if len(output_dir) > 0 and not path.exists(output_dir):
        return JSONResponse(jsonable_encoder({'error': f"{output_dir} is not exists"}), status_code=400)
    
    
    img = Image.open(image).convert('RGB')
    mask = Image.open(image_mask)
    
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
    