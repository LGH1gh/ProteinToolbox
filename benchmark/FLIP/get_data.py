import os
import wget, tarfile

path = './data/'.strip()
if not os.path.exists(path):
  os.makedirs(path)


datasets = {
  'aav': {
    'fname': 'secondary_structure.tar.gz',
    'address': 'http://data.bioembeddings.com/public/FLIP/aav/',
  },
  'contact_map': {
    'fname': 'contact_map.tar.gz',
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