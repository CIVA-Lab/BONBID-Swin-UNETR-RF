# from:
# https://gitlab.missouri.edu/CIVALab/3d-segmentation-monai/-/blob/gani-update/model.py
# hash: 2c45f6ed

from typing import Sequence

import torch
import yaml
from monai.networks import nets


def get_config(model_config_path: str) -> dict:
    with open(model_config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def edit_keys(d: dict, replace_strs: Sequence[str],
              with_strs: Sequence[str]) -> dict:
    new_d = {}
    for key in d:
        new_key = key
        for replace_str, with_str in zip(replace_strs, with_strs):
            new_key = new_key.replace(replace_str, with_str)
        new_d[new_key] = d[key]
    return new_d


def get_model_from_config(model_config: dict) -> torch.nn.Module:
    model_name = model_config['model']
    model_args = model_config['args']
    model = getattr(nets, model_name)(**model_args)
    if 'weights' in model_config:
        weights = torch.load(model_config['weights']['path'])
        if 'remap_keys' in model_config['weights']:
            replace_str, with_str = model_config['weights']['remap_keys']
            weights['state_dict'] = edit_keys(
                weights['state_dict'], replace_str, with_str)
        model.load_from(weights=weights)
    return model
