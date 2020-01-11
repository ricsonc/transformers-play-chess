#!/usr/bin/env python
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
#from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
from ipdb import set_trace as st
import random
import glob

@registry.register_problem
class Chess(text_problems.Text2SelfProblem):
# You can also choose to use a character-level encoder or a token encoder where you provide the vocab file yourself. See Text2TextProblem.vocab_type.

  @property
  def has_inputs(self):
    return False
  
  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  @property
  def is_generate_per_split(self):
    return False

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 9,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]
  
  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate samples of text.
    Args:
      data_dir: final data directory. Typically only used in this method to copy
        over user-supplied vocab files (for example, if vocab_type ==
        VocabType.TOKEN).
      tmp_dir: temporary directory that you can use for downloading and scratch.
      dataset_split: problem.DatasetSplit, which data split to generate samples
        for (for example, training and evaluation).
    Yields:
      Sample: dict<str feature_name, str text>: for language modeling problems
        (i.e. Text2SelfProblems), this generator should yield dicts with only
        the "targets" key.
    """
    #The vocab file should be stored in
    #    `data_dir/` with the name specified by `self.vocab_filename`.

    months = sorted(glob.glob('dump/*'))

    for month in months:
      print(month)
      with open(month, 'r') as f:
        while 1:
          line = f.readline()
          if not line:
            break
          yield {'targets': line.strip()}


# @registry.register_hparams
# def transformer_tiny2():
#   hparams = transformer_base()
#   hparams.num_hidden_layers = 2
#   hparams.hidden_size = 128
#   hparams.filter_size = 512
#   hparams.num_heads = 4
#   hparams.max_length=64
#   return hparams          
