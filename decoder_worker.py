import json
import base64
import io

import redis
import torch
from PIL import Image
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

class DecoderWorker():
    def __init__(self,):
        self.vae =  AutoencoderKL.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", torch_dtype=torch.float32).to("cuda")
        self.r = redis.Redis(host='localhost', port=8866)
        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    

    def run(self,):
        while True:
            _, task = r.blpop('decode_tasks')
            data = json.loads(task)
            unet_output = torch.tensor(data['unet_output']).to('cuda')
            image = self.vae.decode(unet_output / self.vae.config.scaling_factor, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type='pil', do_denormalize=True)
            #image to base64
            byte_io = io.BytesIO()
            image.save(byte_io, format='PNG')
            img_str = base64.b64encode(buffered.getvalue())
            img_str_utf8 = img_str.decode('utf-8')

            #send to producer
            self.r.lpush('finished_images', json.dumps({'image': img_str_utf8}))
