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
from typing import Optional

import numpy as np
import SimpleITK
import torch
import yaml
from monai.inferers import sliding_window_inference

from model import edit_keys, get_model_from_config

INPUT_PREFIX = Path('/input')
OUTPUT_PREFIX = Path('/output')


@dataclass
class Interface:
  slug: str
  relative_path: str

  def load(self):
    input_directory = INPUT_PREFIX / self.relative_path
    mha_files = {f for f in input_directory.glob("*.mha") if f.is_file()}

    if len(mha_files) == 1:
      mha_file = mha_files.pop()
      return SimpleITK.ReadImage(mha_file)
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
    SimpleITK.WriteImage(data, file_save_name)


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


class UNet(object):
  def __init__(self, config_path, checkpoint):
    with open(config_path) as f:
      config = yaml.safe_load(f)
    self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    self.model = get_model_from_config(config)
    self.model.to(self.device)
    state_dict = torch.load(checkpoint, map_location=self.device)
    # print(state_dict.keys())
    state_dict = edit_keys(state_dict['state_dict'], ['_model.'], [''])
    self.model.load_state_dict(state_dict)
    self.model.eval()

  def predict(
      self,
      zadc: np.ndarray,
      adc: Optional[np.ndarray] = None,
      window: int = 5,
      th: float = 0.5
  ) -> np.ndarray:
    '''Predicts using RF and a moving window.'''
    # Add Batch + channel
    x = torch.tensor(zadc).unsqueeze(0).unsqueeze(0).to(self.device)
    out = sliding_window_inference(
      x, (128, 128, 32), 4, self.model, overlap=0.5)[0]
    out = out.detach().cpu().argmax(dim=0).numpy()
    return out.astype('uint8')


def predict(*, inputs: Inputs) -> Outputs:
  z_adc = inputs.z_score_apparent_diffusion_coefficient_map
  adc_ss = inputs.skull_stripped_adc
  z_adc = SimpleITK.GetArrayFromImage(z_adc)
  adc_ss = SimpleITK.GetArrayFromImage(adc_ss)

  model = UNet(
    './configs/bonbid_unet.yml',
    './model/best_bonbid_unet-epoch=737-val_dice=0.77.ckpt')
  out = model.predict(z_adc)

  hie_segmentation = SimpleITK.GetImageFromArray(out)

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
