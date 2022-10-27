import torch
from torch import nn

from ..utils import get_activation_func
from ..utils import get_activation_name

class ProteinResnetEmbedding(nn.Module):
  def __init__(self, args, dictionary) -> None:
    super().__init__()
    self.hidden_dim = args.hidden_dim
    self.padding_idx = dictionary.padding_idx
    self.token_embedding = nn.Embedding(len(dictionary), args.hidden_dim)
    inverse_frequency = 1 / (10000 ** (torch.arange(0.0, args.hidden_dim, 2.0) / args.hidden_dim))
    self.register_buffer('inverse_frequency', inverse_frequency)

    self.layer_norm = nn.LayerNorm(args.hidden_dim, eps=1e-12)
    self.dropout = nn.Dropout(args.dropout)
  
  def forward(self, tokens):
    assert tokens.ndim == 2
    batch_size, sequence_length = tokens.size()
    padding_mask = tokens.eq(self.padding_idx) 
    x = self.token_embedding(tokens)
    position_ids = torch.arange(
      sequence_length-1, -1, -1.0, 
      dtype=x.dtype,
      device=x.device)
    sinusoidal_input = torch.ger(position_ids, self.inverse_frequency)
    position_embedding = torch.cat([sinusoidal_input.sin(), sinusoidal_input.cos(), -1])
    position_embedding = position_embedding.unsqueeze(0)
    x = x + position_embedding

    x = self.dropout(self.layer_norm(x))
    x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
    return x


class MaskedConv1d(nn.Conv1d):
  def forward(self, x, mask=None):
    if mask is not None:
      x = x * mask
    return super().forward(x)


class ProteinResnetBlock(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()
    self.conv1 = MaskedConv1d(args.hidden_dim, args.hidden_dim, 3, padding=1, bias=False)
    self.norm1 = nn.LayerNorm(args.hidden_dim)
    self.conv2 = MaskedConv1d(args.hidden_dim, args.hidden_dim, 3, padding=1, bias=False)
    self.norm2 = nn.LayerNorm(args.hidden_dim)
    self.activation_fn = get_activation_func(args.activation)

  def forward(self, x, mask=None):
    residual = x
    x = self.conv1(x, mask)
    x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
    x = self.activation_fn(x)
    x = self.conv2(x, mask)
    x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
    x = x + residual
    return x


class ProteinResnetEncoder(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()
    self.layers = nn.ModuleList([
      ProteinResnetBlock(args) for _ in range(args.hidden_layer_num)
    ])

  def forward(self, x, mask=None):
    for layer in self.layers:
      x = layer(x, mask)
    return x

class ProteinResnetRepresentation(nn.Module):
  @classmethod
  def add_args(parser):
    parser.add_argument("--hidden-layer-num", type=int, metavar="L", help="hidden layer num")
    parser.add_argument("--hidden-dim", type=int, metavar="H", help="hidden embedding dimension")
    parser.add_argument("--activation", choices=get_activation_name(), help="activation function to use")
    parser.add_argument("--dropout", type=float, metavar="D", help="dropout probability")

  def _init_weights(self, module):
    if isinstance(module, nn.Embedding):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
      if module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.Conv1d):
      nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
      if module.bias is not None:
        module.bias.data.zero_()

  def __init__(self, args, dictionary) -> None:
    super().__init__()
    self.args = args
    self.dictionary = dictionary
    self.embedding = ProteinResnetEmbedding(args, dictionary)
    self.encoder = ProteinResnetEncoder(args)

    self._init_weights()

  def forward(self, tokens, mask=None):
    if mask is not None and torch.any(mask != 1):
      extended_mask = mask.unsqueeze(2)
      extended_mask = extended_mask.to(dtype=next(self.parameters()).dtype)
    else:
      extended_mask = None

    x = self.embedding(tokens)
    x = self.encoder(x.transpose(1, 2), extended_mask.transpose(1, 2)).transpose(1, 2)

    return x

