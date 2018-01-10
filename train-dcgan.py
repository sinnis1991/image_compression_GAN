#!/usr/bin/env python3

# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/main.py
#   + License: MIT
# [2016-08-05] Modifications for Inpainting: Brandon Amos (http://bamos.github.io)
#   + License: MIT

import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use")
flags.DEFINE_string("dataset", "lfw-aligned-64", "Dataset directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_integer("z_dim", 100, "Define the z dim[z_dim]")
flags.DEFINE_integer("gf_dim", 64, "Define the gf dim[gf_dim]")
flags.DEFINE_integer("df_dim", 64, "Define the df dim[df_dim]")
flags.DEFINE_integer("c_bits", 8, "Define the bits for every entry")

FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

FLAGS.checkpoint_dir = './checkpoint/SVHN_full'
FLAGS.dataset = "./image_data/SVHN_data/full_image_test"
FLAGS.z_dim = 100
FLAGS.gf_dim = 64
FLAGS.df_dim = 64
FLAGS.epoch = 10
FLAGS.image_size = 32
FLAGS.batch_size = 64
# FLAGS.simple_size =128

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                  is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir, z_dim=FLAGS.z_dim,
                  gf_dim=FLAGS.gf_dim,df_dim=FLAGS.df_dim,c_bits=FLAGS.c_bits)

    dcgan.train(FLAGS)
