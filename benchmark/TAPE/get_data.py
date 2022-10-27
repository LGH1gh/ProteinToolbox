import os
import wget, tarfile

path = './data/'.strip()
if not os.path.exists(path):
  os.makedirs(path)

datasets = {
  'secondary_structure': {
    'fname': 'secondary_structure.tar.gz',
    'address': 'http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/secondary_structure.tar.gz',
  },
  'contact_map': {
    'fname': 'proteinnet.tar.gz',
    'address': 'http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/proteinnet.tar.gz',
  },
  'remote_homology': {
    'fname': 'remote_homology.tar.gz',
    'address': 'http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/remote_homology.tar.gz'
  },
  'fluorescence': {
    'fname': 'fluorescence.tar.gz',
    'address': 'http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz'
  },
  'stability': {
    'fname': 'stability.tar.gz',
    'address': 'http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/stability.tar.gz'
  }
}

for name in datasets.keys():
  out_fname = path + datasets[name]['fname']
  wget.download(datasets[name]['address'], out=out_fname)
  with tarfile.open(out_fname) as tar:
    tar.extractall(path)
  os.remove(out_fname)