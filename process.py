# Copyright 2023 Radboud University Medical Center
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# codes written by Rina Bao (rina.bao@childrens.harvard.edu) for BONBID-HIE
# MICCAI Challenge 2023 (https://bonbid-hie2023.grand-challenge.org/).
# Demo of Algorithm docker

from dataclasses import dataclass, make_dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.data import image_writer
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Spacing

from model import edit_keys, get_model_from_config
from transforms import get_transform_from_config

INPUT_PREFIX = Path('/input')
OUTPUT_PREFIX = Path('/output')

FORCE_CUDA = True


@dataclass
class Interface:
  slug: str
  relative_path: str

  def load(self):
    input_directory = INPUT_PREFIX / self.relative_path
    mha_files = {f for f in input_directory.glob("*.mha") if f.is_file()}

    if len(mha_files) == 1:
      mha_file = mha_files.pop()
      return mha_file
    elif len(mha_files) > 1:
      raise RuntimeError(
          f'More than one mha file was found in {input_directory!r}'
      )
    else:
      raise NotImplementedError

  def save(self, *, data):
    output_directory = OUTPUT_PREFIX / self.relative_path
    output_directory.mkdir(exist_ok=True, parents=True)
    file_save_name = output_directory / 'overlay.mha'

    writer = image_writer.ITKWriter(output_dtype=np.uint8)
    writer.set_data_array(data['data'], channel_dim=None)
    writer.set_metadata(data['meta'])
    writer.write(file_save_name)


INPUT_INTERFACES = [
    Interface(
        slug="z_score_apparent_diffusion_coefficient_map",
        relative_path="images/z-score-adc"),
    Interface(
        slug="skull_stripped_adc",
        relative_path="images/skull-stripped-adc-brain-mri"),
]

OUTPUT_INTERFACES = [
    Interface(
        slug='hypoxic_ischemic_encephalopathy_lesion_segmentation',
        relative_path='images/hie-lesion-segmentation'),
]

Inputs = make_dataclass(
    cls_name='Inputs',
    fields=[(inpt.slug) for inpt in INPUT_INTERFACES])

Outputs = make_dataclass(
    cls_name='Outputs',
    fields=[(output.slug) for output in OUTPUT_INTERFACES])


def load() -> Inputs:
  return Inputs(
      **{interface.slug: interface.load() for interface in INPUT_INTERFACES}
  )


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

  def predict(self, x: torch.Tensor) -> torch.Tensor:
    '''Predicts using RF and a moving window.'''
    # Add Batch + channel
    with torch.no_grad():
      x = x.to(self.device)
      out = sliding_window_inference(
          x, (128, 128, 128), 1, self.model, overlap=0,
          device=self.device, sw_device=self.device)[0]
    return out


def predict(*, inputs: Inputs) -> Outputs:
  transform = get_transform_from_config('configs/transform.yml')

  input = transform({
      'z_adc': inputs.z_score_apparent_diffusion_coefficient_map,
      'adc_ss': inputs.skull_stripped_adc
  })

  model = Net(
      './configs/bonbid_swinunetr.yml',
      './model/bonbid_swin_unetr_48_data-50-50-split-dice-clip-5.0.ckpt')
  x = input['image'].unsqueeze(0)

  out = model.predict(x)
  out = out.detach().cpu()
  output_transform = Compose([
    get_transform_from_config('configs/output_transform.yml'),
    Spacing(
      input['z_adc_meta_dict']['spacing'],
      mode='nearest')
  ])

  out = output_transform(out)[0]
  out = out.numpy().astype('uint8')
  hie_segmentation = {
    'data': out,
    'meta': input['z_adc_meta_dict']
  }

  outputs = Outputs(
      hypoxic_ischemic_encephalopathy_lesion_segmentation=hie_segmentation
  )
  return outputs


def save(*, outputs: Outputs) -> None:
  for interface in OUTPUT_INTERFACES:
    interface.save(data=getattr(outputs, interface.slug))


def main() -> int:
  inputs = load()
  outputs = predict(inputs=inputs)
  save(outputs=outputs)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
