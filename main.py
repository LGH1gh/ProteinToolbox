import pytorch_lightning as pl
from pytorch_lightning import seed_everything


def main():
  a = ((1, 2, 3), ('seq1', 'seq2', 'seq3'))
  b = zip(a[0], a[1])
  print(list(b))


if __name__ == '__main__':
  main()