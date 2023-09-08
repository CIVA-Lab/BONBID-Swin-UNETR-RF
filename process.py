from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from monai.data import image_writer
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Spacing

from model import edit_keys, get_model_from_config
from transforms import get_transform_from_config

INPUT_PREFIX = Path('/input')
INPUT_ZADC_DIR = 'images/z-score-adc'
INPUT_ADC_SS_DIR = 'images/skull-stripped-adc-brain-mri'

OUTPUT_PREFIX = Path('/output')
OUTPUT_LESION_DIR = 'images/hie-lesion-segmentation'
OUTPUT_FILENAME = 'overlay.mha'

# Change with 2-channel 2-output MONAI model.
NET_CFG = 'configs/bonbid_swinunetr.yml'
NET_CKPT = 'model/bonbid_swin_unetr_48_data-50-50-split-dice-clip-5.0.ckpt'

INPUT_TRANSFORM = 'configs/transform.yml'
OUTPUT_TRANSFORM = 'configs/output_transform.yml'

FORCE_CUDA = True


class Net(object):
  def __init__(self, config_path, checkpoint):
    with open(config_path) as f:
      config = yaml.safe_load(f)
    if FORCE_CUDA or torch.cuda.is_available():
      self.device = 'cuda:0'
    else:
      self.device = 'cpu'
    self.model = get_model_from_config(config)
    self.model.to(self.device)
    state_dict = torch.load(checkpoint, map_location=self.device)
    state_dict = edit_keys(state_dict['state_dict'], ['_model.'], [''])
    self.model.load_state_dict(state_dict)
    self.model.eval()

    self.transform = get_transform_from_config(INPUT_TRANSFORM)

  def predict(self, inputs: dict[str, Any]) -> torch.Tensor:
    '''Predicts using RF and a moving window.'''
    # Load and pre-process
    inputs = self.transform(inputs)
    x = inputs['image'].unsqueeze(0)
    # Inference
    with torch.no_grad():
      x = x.to(self.device)
      out = sliding_window_inference(
          x, (128, 128, 128), 1, self.model, overlap=0,
          device=self.device, sw_device=self.device)[0]
    out = out.detach().cpu()

    # Post-process
    output_transform = Compose([
      get_transform_from_config(OUTPUT_TRANSFORM),
      Spacing(
        inputs['z_adc_meta_dict']['spacing'],
        mode='nearest')
    ])
    out = output_transform(out)[0]
    out = out.numpy().astype('uint8')

    return {'data': out, 'meta': inputs['z_adc_meta_dict']}


def main() -> int:

  # Trailing comma insures there is exactly one volume.
  zadc, = list((INPUT_PREFIX / INPUT_ZADC_DIR).glob('*.mha'))
  adc_ss, = list((INPUT_PREFIX / INPUT_ADC_SS_DIR).glob('*.mha'))

  # Prediction.
  model = Net(NET_CFG, NET_CKPT)
  output = model.predict({'z_adc': zadc, 'adc_ss': adc_ss})

  output_directory = OUTPUT_PREFIX / OUTPUT_LESION_DIR
  output_directory.mkdir(exist_ok=True, parents=True)
  file_save_name = output_directory / OUTPUT_FILENAME

  writer = image_writer.ITKWriter(output_dtype=np.uint8)
  writer.set_data_array(output['data'], channel_dim=None)
  writer.set_metadata(output['meta'])
  writer.write(file_save_name)

  return 0


if __name__ == '__main__':
  raise SystemExit(main())
