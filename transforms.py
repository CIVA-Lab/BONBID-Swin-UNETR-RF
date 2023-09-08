from typing import Any, Union

import monai.transforms
import numpy as np
import yaml


def _preprocess_args(args: dict[str, Any]) -> dict[str, Any]:
  ret = {}
  for k, v in args.items():
    if isinstance(v, list):
      ret[k] = np.array(v)
    else:
      ret[k] = v
  return ret


def get_transform_from_config(
    transform_config: Union[list[dict[str, Any]], dict[str, Any], str]
) -> monai.transforms.Compose:
  if isinstance(transform_config, str):
    with open(transform_config, "r") as stream:
      transform_config = yaml.safe_load(stream)
  transform_list = []
  if isinstance(transform_config, dict):
    for transform_name, transform_args in transform_config.items():
      transform = getattr(monai.transforms, transform_name)(
          **_preprocess_args(transform_args))
      transform_list.append(transform)
  elif isinstance(transform_config, list):
    for transform_dict in transform_config:
      assert len(transform_dict) == 1, "Only one transform per item is allowed"
      transform_name, transform_args = list(transform_dict.items())[0]
      transform = getattr(monai.transforms, transform_name)(
          **_preprocess_args(transform_args))
      transform_list.append(transform)
  return monai.transforms.Compose(transform_list)
