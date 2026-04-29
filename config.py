import json
import os
import numpy as np


def load_camera_config(config_path):
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    K = np.array(cfg['K'], dtype=np.float64)
    dist = np.array(cfg.get('distortion', [0., 0., 0., 0., 0.]), dtype=np.float64)
    width = cfg.get('width', None)
    height = cfg.get('height', None)

    return {
        'K': K,
        'distortion': dist,
        'width': width,
        'height': height,
    }


def save_camera_config(config_path, K, distortion=None, width=None, height=None):
    if distortion is None:
        distortion = [0., 0., 0., 0., 0.]

    cfg = {
        'K': K.tolist(),
        'distortion': list(distortion),
    }
    if width is not None:
        cfg['width'] = width
    if height is not None:
        cfg['height'] = height

    os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)
