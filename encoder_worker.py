import base64
import json
from io import BytesIO

import redis
import torch
from PIL import Image
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))



class EncoderWorker():
    def __init__(self,):
        pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", torch_dtype=torch.float32).to("cuda")
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler
        self.r = redis.Redis(host='localhost', port=8866)
        self.scaling_factor = pipe.vae.config.scaling_factor
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        self.device = torch.device('cuda')
        self.dtype = torch.float32
        self.strength = 0.5
        self.num_inference_steps = 2

    
    
    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start
    
    def prepare_latents(self, image, timestep):
        init_latents = self.vae.encode(image).latent_dist.sample()
        init_latents = init_latents * self.scaling_factor
        shape = init_latents.shape
        noise = randn_tensor(shape, device=self.device, dtype=self.dtype)
        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        return init_latents

    def run(self,):
        while True:
            _, task = self.r.blpop('encode_tasks')
            data = json.loads(task)
            prompt = data['prompt']
            # base64 to image
            image = base64_to_image(data['image'])

            # preprocess image
            image = self.image_processor.preprocess(image).to(device='cuda', dtype=torch.float32)

            # set timesteps
            self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, self.strength)
            
            # Prepare latent
            latents = self.prepare_latents(image, timestep)

            task = {'latents' : latents.numpy().tolist(), 'timesteps' : timesteps, 'prompt': prompt}
            self.r.lpush('unet_tasks', json.dump(task))

