import lmdb
import pickle as pkl
import numpy as np
from typing import Union, Optional
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class LMDBDataset(Dataset):
  def __init__(self, data_path: Union[str, Path], in_memory=False) -> None:
    data_path = Path(data_path)
    if not data_path.exists():
      raise FileNotFoundError(data_path)
    
    self._env = lmdb.open(str(data_path), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    with self._env.begin(write=False) as txn:
      self._data_size = pkl.loads(txn.get(b'num_examples'))
    
    if in_memory:
      cache = [None] * self._data_size
      self._cache = cache
    self._in_memory = in_memory

  def __len__(self) -> int:
    return self._data_size

  def __getitem__(self, index):
    if self._in_memory and self._cache[index] is not None:
      item = self._cache[index]
    else:
      with self._env.begin(write=False) as txn:
        item = pkl.loads(txn.get(str(index).encode()))
        if 'id' not in item:
          item['id'] = str(index)
        if self._in_memory:
          self._cache[index] = item
    return item


class TAPESecondaryStructureDataset(Dataset):
  def __init__(self, args, split) -> None:
    if split not in ('train', 'valid', 'casp12', 'ts115', 'cb513'):
      raise ValueError(f'Unrecognized split: {split}. Must be one of '
                       f'["train", "valid", "casp12", "ts115", "cb513"]')
    data_dir = Path(args.data_dir)
    data_file = f'TAPE/data/secondary_structure/secondary_structure_{split}.lmdb'
    self.data = LMDBDataset(data_dir / data_file, args.in_memory)
    self.task_type = args.task_type

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    item = self.data[index]
    sequence = item['primary']
    label = np.asarray(item[self.task_type], np.int64)
    return sequence, label

  def size(self, index):
    item = self.data[index]
    return len(item['primary'])


class TAPESecondaryStructureDataModule(pl.LightningDataModule):
  @classmethod
  def add_args(cls, parser):
    parser.add_argument("--task-test-name", type=str, choices=['casp12', 'ts115', 'cb513'])
    parser.add_argument("--task-type", type=str, choices=['ss3', 'ss8'])
    parser.add_argument("--in-memory", type=bool, default=False)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--inference-batch-size", type=int)

  def __init__(self, args) -> None:
    super().__init__()
    self.args = args
    self.task_test_name = args.task_test_name
    self.train_batch_size = args.train_batch_size
    self.inference_batch_size = args.inference_batch_size
    args.label_num = 3 if args.task_type == 'ss3' else 8
  
  def prepare_data(self) -> None:
    pass

  def setup(self, stage: Optional[str]=None) -> None:
    self.train_dataset = TAPESecondaryStructureDataset(self.args, 'train')
    self.valid_dataset = TAPESecondaryStructureDataset(self.args, 'valid')
    self.test_dataset = TAPESecondaryStructureDataset(self.args, self.task_test_name)

  def train_dataloader(self):
    train_dataloader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, collate_fn=lambda x: x)
    return train_dataloader

  def val_dataloader(self):
    valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.inference_batch_size, collate_fn=lambda x: x)
    return valid_dataloader

  def test_dataloader(self):
    test_dataloader = DataLoader(self.test_dataset, batch_size=self.inference_batch_size, collate_fn=lambda x: x)
    return test_dataloader