import json

import redis
import torch
from diffusers import DiffusionPipeline
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils.torch_utils import randn_tensor


class UnetWorker():
    def __init__(self,):
        self.r = redis.Redis(host='localhost', port=8866)
        pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
        pipe = pipe.to(torch_device="cuda", torch_dtype=torch.float32)
        self.unet = pipe.unet
        self.tokenizer = pipe.tokenizer

    def encode_prompt(self, prompt):
        text_inputs = self.tokenizer(prompt,
                                    padding="max_length",
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors="pt",)
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder(text_input_ids.to('cuda'))
        prompt_embeds = prompt_embeds[0]
        return prompt_embeds

    def run(self,):
        while True:
            _, task = r.blpop('unet_tasks')
            data = json.loads(task)
            timesteps = data['timesteps']
            prompt = data['prompt']
            latents = data['latents']        
            latents = torch.tensor(latents).to("cuda")
            # encode prompt
            prompt_embeds = self.encode_prompt(prompt)

            #Denoising loop
            for i, t in enumerate(timesteps):
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]
                latents = self.scheduler.step(noise_pred, t, latents)[0]

            task = {'unet_output': latents.cpu().numpy().tolist()}
            r.lpush('decode_tasks', json.dumps(task))
