import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F

import pytorch_lightning as pl


class TAPESecondaryStructureModel(pl.LightningModule):
  def __init__(self, args, tokenizer, encoder) -> None:
    super().__init__()
    self.args = args
    self.tokenizer = tokenizer
    self.encoder = encoder
    self.decoder = nn.Sequential(
      weight_norm(nn.Linear(args.hidden_dim, args.hidden_dim // 2), dim=None),
      nn.GELU(),
      nn.Dropout(args.dropout, inplace=True),
      weight_norm(nn.Linear(args.hidden_dim // 2, args.label_num), dim=None)
    )
    self.ignore_index = -100

  def forward(self, tokens):
    import IPython; IPython.embed(); exit()
    return 0

  def training_step(self, batch, batch_idx):
    sequences, labels = zip(*batch)
    tokens = self.encoder.converter(sequences).cuda()
    embedding = self.encoder(tokens)
    logits = self.decoder(embedding)

    refined_labels = torch.empty(
      (tokens.size(0), tokens.size(1)) , dtype=torch.int64, device=tokens.device
    )
    refined_labels.fill_(self.ignore_index)
    for sequence_idx, label in enumerate(labels):
      refined_labels[sequence_idx, :len(label)] = torch.tensor(label)

    lprobs = F.log_softmax(logits.reshape(-1, logits.size(-1)), dim=-1, dtype=torch.float32)
    loss = F.nll_loss(lprobs, refined_labels.reshape(-1), ignore_index=self.ignore_index, reduction="mean")
    self.log('train_loss', loss, logger=True, batch_size=self.args.train_batch_size)
    return loss

  def validation_step(self, batch, batch_idx):
    sequences, labels = zip(*batch)
    tokens = self.encoder.converter(sequences).cuda()
    embedding = self.encoder(tokens)
    logits = self.decoder(embedding)

    refined_labels = torch.empty(
      (tokens.size(0), tokens.size(1)) , dtype=torch.int64, device=tokens.device
    )
    refined_labels.fill_(self.ignore_index)
    for sequence_idx, label in enumerate(labels):
      refined_labels[sequence_idx, :len(label)] = torch.tensor(label)
    lprobs = F.log_softmax(logits.reshape(-1, logits.size(-1)), dim=-1, dtype=torch.float32)
    outputs = {}
    outputs['correct'] = sum(refined_labels.view(-1) == torch.argmax(lprobs.view(-1, lprobs.size(-1)), 1))
    outputs['total'] = sum(refined_labels.ne(self.ignore_index).view(-1))
    return outputs


  def validation_epoch_end(self, outputs) -> None:
    acc = sum([output['correct'] for output in outputs]) / sum([output['total'] for output in outputs])
    self.log("val_acc", acc.data, prog_bar=True, on_epoch=True)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    return [optimizer], [torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)]


