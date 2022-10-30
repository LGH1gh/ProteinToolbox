import argparse
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from proteintoolbox import DATA_MODULES, ENCODERS, MODELS, TOKENIZERS



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--tokenizer-name', type=str, default='OneHot')
  parser.add_argument('--encoder-name', type=str, default='Resnet')
  parser.add_argument('--dataset-name', type=str, default='TAPE_secondary_structure')
  parser.add_argument('--data-dir', type=str, default='../benchmark/')
  parser = pl.Trainer.add_argparse_args(parser)

  args, _ = parser.parse_known_args(['tokenizer_name', 'encoder_name', 'dataset_name'])
  if hasattr(args, 'encoder_name'):
    ENCODERS[args.encoder_name].add_args(parser)
  if hasattr(args, 'dataset_name'):
    DATA_MODULES[args.dataset_name].add_args(parser)
  args = parser.parse_args()

  data_module = DATA_MODULES[args.dataset_name](args)
  tokenizer = TOKENIZERS[args.tokenizer_name].build_tokenizer()
  encoder = ENCODERS[args.encoder_name](args, tokenizer)
  model = MODELS[args.dataset_name](args, tokenizer, encoder)
  print(args)
  trainer = pl.Trainer.from_argparse_args(args)
  trainer.fit(model, datamodule=data_module)
  
  

  import IPython; IPython.embed(); exit()




if __name__ == '__main__':
  main()
