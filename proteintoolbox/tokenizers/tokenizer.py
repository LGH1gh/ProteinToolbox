import itertools
from typing import Sequence, List
import constant


class OneHotTokenizer(object):
  """Tokenizer"""

  def __init__(
    self, 
    standard_toks: Sequence[str], 
    special_toks: Sequence[str] = ( "<eos>", "<pad>", "<unk>")
  ):
    self.standard_toks = list(standard_toks)
    self.special_toks = list(special_toks)

    self.all_toks = list(self.standard_toks)
    self.all_toks.extend(self.special_toks)
    for i in range((8 - (len(self.all_toks) % 8)) % 8):
      self.all_toks.append(f"<null_{i + 1}>")

    self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}
    self.unk_idx = self.tok_to_idx["<unk>"]
    self.padding_idx = self.get_idx("<pad>")
    self.eos_idx = self.get_idx("<eos>")
    self.unique_no_split_tokens = self.all_toks

  def __len__(self):
    return len(self.all_toks)

  def get_idx(self, tok):
    return self.tok_to_idx.get(tok, self.unk_idx)

  def get_tok(self, idx):
    return self.all_toks[idx]

  def to_dict(self):
    return self.tok_to_idx.copy()

  def pad(self):
    return self.padding_idx

  @classmethod
  def build_dictionary(cls):
    standard_toks = list(constant.PROTEIN_TOKS.keys())
    special_toks = ("<eos>", "<pad>", "<unk>")
    return cls(standard_toks, special_toks)

  def _tokenize(self, text) -> str:
    return text.split()

  def tokenize(self, text, **kwargs) -> List[str]:
    def split_on_token(tok, text):
      result = []
      split_text = text.split(tok)
      for i, sub_text in enumerate(split_text):
        if i < len(split_text) - 1:
          sub_text = sub_text.rstrip()
        if i > 0:
          sub_text = sub_text.lstrip()
              
        if i == 0 and not sub_text:
          result.append(tok)
        elif i == len(split_text) - 1:
          if sub_text:
            result.append(sub_text)
          else:
            pass
        else:
          if sub_text:
            result.append(sub_text)
          result.append(tok)
      return result

    def split_on_tokens(tok_list, text):
      if not text.strip():
        return []

      tokenized_text = []
      text_list = [text]
      for tok in tok_list:
        tokenized_text = []
        for sub_text in text_list:
          if sub_text not in self.unique_no_split_tokens:
            tokenized_text.extend(split_on_token(tok, sub_text))
          else:
            tokenized_text.append(sub_text)
        text_list = tokenized_text

      return list(
        itertools.chain.from_iterable(
          (
            self._tokenize(token)
            if token not in self.unique_no_split_tokens
            else [token]
            for token in tokenized_text
          )
        )
      )

    no_split_token = self.unique_no_split_tokens
    tokenized_text = split_on_tokens(no_split_token, text)
    return tokenized_text

  def encode(self, text):
    return [self.tok_to_idx[tok] for tok in self.tokenize(text)]


