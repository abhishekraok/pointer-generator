# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This is the top-level file to train, evaluate or test your summarization model"""
from __future__ import absolute_import
from __future__ import print_function

import sys

from server.summarizer import Summarizer

sys.path.append('..')
import tensorflow as tf
from collections import namedtuple
from data import Vocab
from model import SummarizationModel
from decode import BeamSearchDecoder
from flask import Flask
import optparse

FLAGS = tf.app.flags.FLAGS
from flask import render_template, request

# Console article input
tf.app.flags.DEFINE_string('input_article', '', 'To summarize a single article given in command line, '
                                                'this is the input article.')

# Where to find data
tf.app.flags.DEFINE_string('data_path', '',
                           'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', False,
                            'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')

# Where to save output
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '',
                           'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35,
                            'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000,
                            'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False,
                            'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0,
                          'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False,
                            'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')

with open('server/templates/fish_article.txt') as f:
  default_article = f.read()


def setup_summarizer(settings):
  tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want
  tf.logging.info('Starting seq2seq_attention ')

  # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
  vocab = Vocab(settings.vocab_path, settings.vocab_size)  # create a vocabulary

  # If in decode mode, set batch_size = beam_size
  # Reason: in decode mode, we decode one example at a time.
  # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
  FLAGS.batch_size = FLAGS.beam_size

  # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
  hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt',
                 'pointer_gen']
  hps_dict = {}
  for key, val in FLAGS.__flags.items():  # for each flag
    if key in hparam_list:  # if it's in the list
      hps_dict[key] = val  # add it to the dict
  hps = namedtuple("HParams", list(hps_dict.keys()))(**hps_dict)

  tf.set_random_seed(111)  # a seed value for randomness

  if hps.mode != 'decode':
    raise ValueError("The 'mode' flag must be decode for serving")
  decode_model_hps = hps  # This will be the hyperparameters for the decoder model
  decode_model_hps = hps._replace(
    max_dec_steps=1)  # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
  serving_device = '/cpu:0'
  model = SummarizationModel(decode_model_hps, vocab, default_device=serving_device)
  decoder = BeamSearchDecoder(model, None, vocab)
  return Summarizer(decoder, vocab=vocab, hps=hps)


class Settings:
  def __init__(self, vocab_path, vocab_size):
    self.vocab_path = vocab_path
    self.vocab_size = vocab_size


parser = optparse.OptionParser()
parser.add_option("-L", "--log_root", help="log root")
parser.add_option("-V", "--vocab_path", help="vocab file")
options, _ = parser.parse_args()

FLAGS.log_root = options.log_root
vocab_path = options.vocab_path
FLAGS.mode = 'decode'
settings = Settings(vocab_path=vocab_path, vocab_size=50000)
summarizer = setup_summarizer(settings)

app = Flask(__name__)


@app.route('/')
def index():
  return render_template('index.html', summary='N/A', article=default_article)


@app.route('/', methods=['POST'])
def index_post():
  article = request.form['article']
  summarized_text = summarizer.summarize(article)
  return render_template('index.html', summary=summarized_text, article=article)


@app.route('/summarize/<text>')
def summarize(text):
  return summarizer.summarize(text)


app.run(host='0.0.0.0')
