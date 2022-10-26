"""Process raw files as objects in python"""

def fasta2dict(data_path):
  """Return {name: sequence} dict from a fasta format file."""
  name2sequence = {}
  with open(f'{data_path}', 'r', encoding='utf-8') as f:
    for line in f:
      if line.startswith('>'):
        name = line.replace('>', '').strip()
        name2sequence[name] = ''
      else:
        name2sequence[name] += line.replace('\n', '').strip()
  return name2sequence

