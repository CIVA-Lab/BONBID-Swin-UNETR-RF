import argparse
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
from skimage.morphology import remove_small_objects
from skimage.util.shape import view_as_windows

from model import edit_keys, get_model_from_config
from transforms import get_transform_from_config


class Net(object):
  def __init__(
      self,
      config_path,
      checkpoint,
      rf_checkpoint,
      input_transform,
      force_cuda=False
  ) -> None:
    with open(config_path) as f:
      config = yaml.safe_load(f)
    if force_cuda or torch.cuda.is_available():
      self.device = 'cuda:0'
    else:
      self.device = 'cpu'
    self.model = get_model_from_config(config)
    self.model.to(self.device)
    state_dict = torch.load(checkpoint, map_location=self.device)
    state_dict = edit_keys(state_dict['state_dict'], ['_model.'], [''])
    self.model.load_state_dict(state_dict)
    self.model.eval()

    self.transform = get_transform_from_config(input_transform)
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

    return out, inputs['z_adc_meta_dict']

  def predict(self, inputs, window=5, th=0.5):
    zadc = inputs['z_adc']
    adc = inputs['adc_ss']
    pr, meta = self._deep_predict(inputs)

    x, _ = self._load_mha(zadc)
    ad, _ = self._load_mha(adc)
    padded_pr = np.zeros(x.shape, pr.dtype)

    padded_pr[:pr.shape[0], :pr.shape[1], :pr.shape[2]] = pr
    pr = padded_pr

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
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input_zadc_dir', type=str,
      default='/input/images/z-score-adc')
  parser.add_argument(
      '--input_adc_ss_dir', type=str,
      default='/input/images/skull-stripped-adc-brain-mri')
  parser.add_argument(
      '--output_dir', type=str,
      default='/output/images/hie-lesion-segmentation')
  parser.add_argument(
      '--output_filename', type=str,
      default='overlay.mha')
  parser.add_argument(
      '--net_cfg', type=str,
      default='configs/bonbid_swinunetr.yml')
  parser.add_argument(
      '--net_ckpt', type=str,
      default='model/bonbid_swin_unetr_48_no-empty-overfit-48-dice+loghaus-lr=0.0001.ckpt')
  parser.add_argument(
      '--rf_ckpt', type=str,
      default='model/RandomForestClassifier.pkl')
  parser.add_argument(
      '--input_transform', type=str,
      default='configs/transform.yml')
  parser.add_argument(
      '--many', action='store_true', help='Whether to process many files')
  parser.add_argument(
      '--force_cuda', type=str, default=True)
  args = parser.parse_args()

  # Trailing comma insures there is exactly one volume.
  zadcs: list[Path] = list(Path(args.input_zadc_dir).glob('*.mha'))
  adc_sss: list[Path] = list(Path(args.input_adc_ss_dir).glob('*.mha'))

  assert len(zadcs) == len(adc_sss), f"found {len(zadcs)=} != {len(adc_sss)=}"
  if not args.many and len(zadcs) != 1:
    raise ValueError('Expecting a single image. If you need to process many '
                     'images, please pass the `--many` flag')
  zadcs.sort()
  adc_sss.sort()

  for zadc, adc_ss in zip(zadcs, adc_sss):
    # Prediction.
    model = Net(
        args.net_cfg,
        args.net_ckpt,
        args.rf_ckpt,
        args.input_transform,
        args.force_cuda
    )
    output, meta = model.predict({'z_adc': zadc, 'adc_ss': adc_ss})
    pp_output = []
    for slice in output:
      pp_slice = remove_small_objects(slice, min_size=5)
      pp_output.append(pp_slice)
    output = np.stack(pp_output)
    output = output.astype(np.uint8)

    output_directory = Path(args.output_dir)
    output_directory.mkdir(exist_ok=True, parents=True)
    if args.many:
      # Ex: MGHNICU_010-VISIT_01-ADC_ss.mha -> MGHNICU_010-VISIT_01_lesion.mha
      output_filename = adc_ss.name.replace('-ADC_ss', '_lesion')
    else:
      output_filename = args.output_filename

    file_save_name = output_directory / output_filename

    writer = image_writer.ITKWriter(output_dtype=np.uint8)
    writer.set_data_array(output, channel_dim=None)
    writer.set_metadata(meta)
    writer.write(file_save_name)

  return 0


if __name__ == '__main__':
  raise SystemExit(main())
