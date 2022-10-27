from torch import nn


def resize_embedding(old_embedding, new_token_num=None, init_weight_func=None):
  """ Build a resized Embedding Module from a provided token Embedding Modules.
  """
  if new_token_num == None:
    return old_embedding
  
  old_token_num, old_embedding_dim = old_embedding.weight.size()
  if old_token_num == new_token_num:
    return old_embedding

  new_embedding = nn.Embedding(new_token_num, old_embedding_dim)
  new_embedding.to(old_embedding.weight.device)

  if init_weight_func is not None:
    init_weight_func(new_embedding)
  else:
    new_embedding.weight.data.normal_(mean=0.0, std=0.02)

  copy_token_num = min(old_token_num, new_token_num)
  new_embedding.weight.data[:copy_token_num, :] = \
    old_embedding.weight.data[:copy_token_num, :]
  
  return new_embedding
