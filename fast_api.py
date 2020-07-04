from fastapi import FastAPI, File, UploadFile
app = FastAPI()

import os, random, time, io
from PIL import Image
import numpy as np

from img_color import get_im, im_kmeans

def GetImage(image_url = None, file_obj = None, image_store_dir = '/tmp/posterize/images/'):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    hash_str = random.getrandbits(128)
    im_fname = f'{image_store_dir}{timestr}_{hash_str}.jpg'

    if image_url:
        print(f'{timestr}: Downloading image from {image_url}...')
        im_array_rgb = np.array(get_im(im = image_url))
    elif file_obj:
        print(f'{timestr}: Downloading image with size {len(file_obj)}...')
        pil_im = Image.open(io.BytesIO(file_obj))
        im_array_rgb = np.array(pil_im)
    else:
        return None
    return im_array_rgb

def predict(im_array_rgb, k, b_fast_kmeans, color_name_space,
            kmeans_kargs = {'normalize': True}):
    result = im_kmeans(im_array_rgb, k= k, get_json = True, verbose = False, fast_kmeans = b_fast_kmeans,
                        kmeans_kargs = kmeans_kargs, color_name_space = color_name_space)
    return result

@app.get("/")
def read_root():
    return {"Color Extractor": "see 'docs/' endpoint for usage"}

@app.post("/predict")
def predict_upload(file_obj: bytes = File(...) ,
                    k : int = 3, b_fast_kmeans : bool = True, color_name_space: str = 'xkcd'):
    im = GetImage(file_obj = file_obj)
    return predict(im_array_rgb = im, k = k, b_fast_kmeans = b_fast_kmeans, color_name_space = color_name_space)

@app.get("/predict")
def predict_url(image_url: str = None,
            k : int = 3, b_fast_kmeans : bool = True, color_name_space: str = 'xkcd'):
    im = GetImage(image_url = image_url)
    return predict(im_array_rgb = im, k = k, b_fast_kmeans = b_fast_kmeans, color_name_space = color_name_space)
