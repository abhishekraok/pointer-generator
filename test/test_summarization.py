import unittest
from model import SummarizationModel
from run_summarization import get_hyper_params_from_flag
import tensorflow as tf
import default_params
from data import Vocab

vocab_path = 'vocab_9999'
vocab_size = 9999
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_root', '','')


class SummarizationTest(unittest.TestCase):
  def test_model_builds_graph(self):
    hps = get_hyper_params_from_flag(default_params.__dict__)
    vocab = Vocab(vocab_path, vocab_size)  # create a vocabulary
    device = '/cpu:0'
    model = SummarizationModel(hps, vocab, device)
    with tf.device(device):
      model.build_graph()  # build the graph
