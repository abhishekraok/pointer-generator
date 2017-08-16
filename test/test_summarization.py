import unittest
from model import SummarizationModel
from run_summarization import get_hyper_params_from_flag, set_tf_flags_defalts
import tensorflow as tf
from data import Vocab

vocab_path = 'vocab_9999'
vocab_size = 9999
FLAGS = tf.app.flags.FLAGS


class SummarizationTest(unittest.TestCase):
  def test_model_loads(self):
    set_tf_flags_defalts(tf.app.flags)
    hps = get_hyper_params_from_flag(FLAGS)
    vocab = Vocab(vocab_path, vocab_size)  # create a vocabulary
    SummarizationModel(hps, vocab, FLAGS.device)
