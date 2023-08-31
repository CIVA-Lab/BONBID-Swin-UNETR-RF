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

import joblib
import numpy as np
import SimpleITK
from skimage.util.shape import view_as_windows

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


class BaseNet(object):
  def __init__(self, save_model_path):
    with open(save_model_path, 'rb') as f:
      self.model = joblib.load(f)

  def predict(
      self,
      zadc: np.ndarray,
      adc: Optional[np.ndarray] = None,
      window: int = 5,
      th: float = 0.5
  ) -> np.ndarray:
    '''Predicts using RF and a moving window

    Args:
        x (np.ndarray): input image (H, W, D)

    Returns:
        np.ndarray: predicted image (H, W, D)
    '''
    # Pad the image
    zadc = np.pad(
        zadc, ((0, 0), (window//2, window//2), (window//2, window//2)),
        mode='reflect')
    if adc is not None:
      adc = np.pad(
          adc, ((0, 0), (window//2, window//2), (window//2, window//2)),
          mode='reflect')

    # Create a list of patches
    patches_zadc = view_as_windows(zadc, (1, window, window))
    if adc is not None:
      ptches_adc = view_as_windows(adc, (1, window, window))

    if adc is not None:
      patches = np.concatenate(
          [patches_zadc.reshape(-1, window**2),
           ptches_adc.reshape(-1, window**2)], axis=1)
    else:
      patches = patches_zadc.reshape(-1, window**2)
    # Predict
    y = self.model.predict_proba(patches)[:, 1] > th

    # Reconstruct the image
    y = np.array(y).reshape(zadc.shape[0],
                            zadc.shape[1] - window + 1,
                            zadc.shape[2] - window + 1)
    return y.astype('uint8')


def predict(*, inputs: Inputs) -> Outputs:
  z_adc = inputs.z_score_apparent_diffusion_coefficient_map
  adc_ss = inputs.skull_stripped_adc
  z_adc = SimpleITK.GetArrayFromImage(z_adc)
  adc_ss = SimpleITK.GetArrayFromImage(adc_ss)

  model = BaseNet('./model/RandomForestClassifier-good-one.pkl')
  out = model.predict(z_adc, adc_ss)

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
