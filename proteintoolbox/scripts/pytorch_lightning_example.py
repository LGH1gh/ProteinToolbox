import torch
from torch import embedding, nn
import pytorch_lightning as pl
import torch.nn.functional as F
from ..tokenizers import OneHotTokenizer
from ..encoders import ProteinResnetRepresentation
from ..decoders import TAPESecondaryStructureDecoder
from ..datasets import TAPESecondaryStructureDataset


class LitAutoEncoder(pl.LightningModule):
  def __init__(self, args):
    super().__init__()
    self.dictionary = OneHotTokenizer.build_dictionary()
    self.encoder = ProteinResnetRepresentation(args, self.dictionary)
    self.decoder = TAPESecondaryStructureDecoder(args)

  def forward(self, tokens):
    embedding = self.encoder(tokens)
    return embedding

  def training_step(self, batch, batch_idx):
    tokens, labels = batch
    embedding = self.encoder(tokens)
    logits = self.decoder(embedding)
    loss = F.mse_loss(logits, labels)
    self.log("train_loss", loss)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer

