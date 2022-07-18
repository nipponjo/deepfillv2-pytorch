import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from utils.misc import infer_deepfill
from model import load_model

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def _load_models(config_path, device='cuda'):
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader)

    config = {name:cfg for name, cfg in config.items() 
                       if os.path.exists(cfg['path'])}

    loaded_models = {}

    for name, cfg in config.items():
        is_loaded = False
        if cfg['load_at_startup']:
            model = load_model(cfg['path'], device)
            loaded_models[name] = model
            is_loaded = True
        config[name]['is_loaded'] = is_loaded

    return config, loaded_models


class Inpainter:
    def __init__(self, device=None):
        self.available_models = None
        self.loaded_models = None
        self.host = ""
        self.device = torch.device('cuda'
                      if torch.cuda.is_available() else 'cpu') \
                      if device is None else device

    def load_models(self, config_path):
        available_models, loaded_models = _load_models(config_path, self.device)
        self.available_models = available_models
        self.loaded_models = loaded_models

    def get_model_info(self):
        model_data = []
        for name, cfg in self.available_models.items():
            model_dict = cfg.copy()
            model_dict['name'] = name
            model_dict['type'] = 'df'
            model_data.append(model_dict)

        return model_data

    def check_requested_models(self, models):
        for name in models:
            if not name in self.loaded_models:
                path = self.available_models[name]['path']
                model = load_model(path, self.device)
                if model is None:
                    print(f"model @ {path} not found!")
                    continue
                self.loaded_models[name] = model
                self.available_models['is_loaded'] = True
                print(f'Loaded model: {name}')

    def inpaint(self, image, mask, models, max_size=512):

        req_models = models.split(',')
        self.check_requested_models(req_models)

        image_pil = Image.open(image.file).convert('RGB')
        mask_pil = Image.open(mask.file)

        mw, mh = mask_pil.size
        scale = max_size / max(mw, mh)

        mask_pil = mask_pil.resize(
            (max_size, int(scale*mh)) if mw > mh else (int(scale*mw), max_size))
        image_pil = image_pil.resize(mask_pil.size)

        image, mask = ToTensor()(image_pil), ToTensor()(mask_pil)

        response_data = []
        for idx_model, model_name in enumerate(req_models):
            return_vals = self.available_models[model_name]['return_vals']
            model_output_list = []
            outputs = infer_deepfill(
                self.loaded_models[model_name],
                image.to(self.device), 
                mask.to(self.device),
                return_vals=return_vals
            )
            for idx_out, output in enumerate(outputs):
                Image.fromarray(output) \
                    .save(f'app/files/out_{idx_model}_{idx_out}.png')

                model_output_list.append({
                    'name': return_vals[idx_out],
                    'file': f'{self.host}/files/out_{idx_model}_{idx_out}.png'
                })

            model_output_dict = {
                'name': model_name,
                'output': model_output_list
            }
            response_data.append(model_output_dict)

        return response_data
