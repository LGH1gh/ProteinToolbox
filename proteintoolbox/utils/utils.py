import math
from typing import List, Callable
import torch
import torch.nn.functional as F


def get_activation_name() -> List:
  return [
    "relu", "gelu"
  ]

def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def get_activation_func(name: str) -> Callable:
  if name == "relu":
    return F.relu
  elif name == "gelu":
    return gelu
  else:
    raise ValueError(f'Unrecognized activation function name: {name}.'
                     f'Must be one of ["relu", "gelu"]')
