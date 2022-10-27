import lmdb
import pickle as pkl
import numpy as np
from typing import Union
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

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
  def __init__(self, data_dir, split, in_memory=False) -> None:
    if split not in ('train', 'valid', 'casp12', 'ts115', 'cb513'):
      raise ValueError(f'Unrecognized split: {split}. Must be one of '
                       f'["train", "valid", "casp12", "ts115", "cb513"]')
    data_dir = Path(data_dir)
    data_file = f'secondary_structure/secondary_structure_{split}.lmdb'
    self.data = LMDBDataset(data_dir / data_file, in_memory)
    self.label_num = 3

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    item = self.data[index]
    sequence = item['primary']
    label = np.asarray(item['ss3'], np.int64)
    return sequence, label


if __name__ == '__main__':
  dataset = TAPESecondaryStructureDataset('/Users/wangzeyuan/Code/OpenProtein/ProteinToolbox/benchmark/TAPE/data', 'valid')
  dataloader = DataLoader(dataset, batch_size=4, collate_fn=lambda x: x)
  for sample in dataloader:
    sequences, labels = zip(*sample)
    import IPython; IPython.embed()