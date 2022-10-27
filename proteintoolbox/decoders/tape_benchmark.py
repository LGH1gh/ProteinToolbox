from torch import nn
from torch.nn.utils.weight_norm import weight_norm

class TAPESecondaryStructureDecoder(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()
    self.classify = nn.Sequential(
      weight_norm(nn.Linear(args.hidden_dim, args.hidden_dim // 2), dim=None),
      nn.GELU(),
      nn.Dropout(args.dropout, inplace=True),
      weight_norm(nn.Linear(args.hidden_dim // 2, args.output_dim), dim=None)
    )

  def forward(self, x):
    return self.classify(x)