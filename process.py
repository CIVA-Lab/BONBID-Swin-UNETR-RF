import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import SimpleITK
import torch
import yaml
from evalutils.io import SimpleITKLoader
from monai.data import image_writer
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Spacing
from skimage.util.shape import view_as_windows

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
RF_CKPT = 'model/RandomForestClassifier.pkl'

INPUT_TRANSFORM = 'configs/transform.yml'
OUTPUT_TRANSFORM = 'configs/output_transform.yml'

FORCE_CUDA = True


class Net(object):
  def __init__(self, config_path, checkpoint, rf_checkpoint):
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
    self.rf = joblib.load(rf_checkpoint)

  def _load_mha(self, path):
    loader = SimpleITKLoader()
    im = loader.load_image(path)
    spacing = im.GetSpacing()
    im = SimpleITK.GetArrayFromImage(im)
    return im, spacing

  def _deep_predict(self, inputs: dict[str, Any]) -> torch.Tensor:
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
      Spacing(
        inputs['z_adc_meta_dict']['spacing'],
        mode='bilinear')
    ])
    out = torch.nn.functional.softmax(output_transform(out), dim=0)[1].numpy()
    out = out.transpose(2, 1, 0)
    warnings.warn(str(out.shape))
    return out, inputs['z_adc_meta_dict']

  def predict(self, inputs, window=5, th=0.5):
    pr, meta = self._deep_predict(inputs)
    zadc = inputs['z_adc']
    adc = inputs['adc_ss']

    x, _ = self._load_mha(zadc)
    ad, _ = self._load_mha(adc)

    x = np.pad(x, ((0, 0), (window//2, window//2), (window//2, window//2)),
               mode='reflect')
    ad = np.pad(ad, ((0, 0), (window//2, window//2), (window//2, window//2)),
                mode='reflect')
    pr = np.pad(pr, ((0, 0), (window//2, window//2), (window//2, window//2)),
                mode='reflect')

    # Create a list of patches
    patches = view_as_windows(x, (1, window, window))
    patches_ad = view_as_windows(ad, (1, window, window))
    patches_pr = view_as_windows(pr, (1, window, window))

    patches = np.concatenate([patches.reshape(-1, window**2),
                              patches_ad.reshape(-1, window**2),
                              patches_pr.reshape(-1, window**2)], axis=1)

    patches = patches.reshape(-1, 3 * window**2)
    # Predict
    y = self.rf.predict_proba(patches)[:, 1] > th

    # Reconstruct the image
    y = np.array(y).reshape(x.shape[0],
                            x.shape[1] - window + 1,
                            x.shape[2] - window + 1)
    return y.transpose(2, 1, 0), meta


def main() -> int:

  # Trailing comma insures there is exactly one volume.
  zadc, = list((INPUT_PREFIX / INPUT_ZADC_DIR).glob('*.mha'))
  adc_ss, = list((INPUT_PREFIX / INPUT_ADC_SS_DIR).glob('*.mha'))

  # Prediction.
  model = Net(NET_CFG, NET_CKPT, RF_CKPT)
  output, meta = model.predict({'z_adc': zadc, 'adc_ss': adc_ss})
  output = output.astype(np.uint8)

  output_directory = OUTPUT_PREFIX / OUTPUT_LESION_DIR
  output_directory.mkdir(exist_ok=True, parents=True)
  file_save_name = output_directory / OUTPUT_FILENAME

  writer = image_writer.ITKWriter(output_dtype=np.uint8)
  writer.set_data_array(output, channel_dim=None)
  writer.set_metadata(meta)
  writer.write(file_save_name)

  return 0


if __name__ == '__main__':
  raise SystemExit(main())
