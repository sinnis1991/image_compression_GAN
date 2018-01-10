#!/usr/bin/env python3
#
# Brandon Amos (http://bamos.github.io)
# License: MIT
# 2016-08-05

import argparse
import os
import tensorflow as tf

from model import DCGAN
from model import dataset_files

parser = argparse.ArgumentParser()
parser.add_argument('--approach', type=str,
                    choices=['adam', 'hmc'],
                    default='adam')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--hmcBeta', type=float, default=0.2)
parser.add_argument('--hmcEps', type=float, default=0.001)
parser.add_argument('--hmcL', type=int, default=100)
parser.add_argument('--hmcAnneal', type=float, default=1)
parser.add_argument('--nIter', type=int, default=1000)
parser.add_argument('--imgSize', type=int, default=64)
parser.add_argument('--lam', type=float, default=0.1)
parser.add_argument('--lam_c', type=float, default=0.1)
parser.add_argument('--checkpointDir', type=str, default='checkpoint')
parser.add_argument('--classifier_checkpointDir', type=str, default='checkpoint')
parser.add_argument('--outDir', type=str, default='completions')
parser.add_argument('--outInterval', type=int, default=250)
parser.add_argument('--maskType', type=str,
                    choices=['random', 'center', 'left', 'full', 'grid', 'lowres'],
                    default='center')
parser.add_argument('--centerScale', type=float, default=0.25)
parser.add_argument('--z_dim', type=int, default=100)
parser.add_argument('imgs', type=str, default = 'dataset')
parser.add_argument('--c_bits', type=int, default = 32)


args = parser.parse_args()

args.imgs = './image_data/SVHN_data/SVHN_image_test'
args.imgs = dataset_files(args.imgs)
args.imgs.sort()
# args.imgs = args.imgs[:64*(len(args.imgs)//64)]
args.imgs = args.imgs[:64*1]
args.checkpointDir = './checkpoint/checkpoint_SVHN_z100_d64'
args.classifier_checkpointDir = '../SVHN_classifier/checkpoint'
args.outInterval = args.nIter/10
args.z_dim = 100
args.lam = 0
args.lam_c = 100
args.nIter =10000
args.imgSize =32
args.c_bits=8
args.outDir = './result_set/SVHN_with_Classifier/classifier_test/g_{}_c_{}_con_y'.format(args.lam,args.lam_c)


assert(os.path.exists(args.checkpointDir))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess, image_size=args.imgSize,
                  batch_size=min(64, len(args.imgs)),
                  checkpoint_dir=args.checkpointDir,
                  classifier_checkpoint_dir=args.classifier_checkpointDir,
                  lam=args.lam,lam_c=args.lam_c,
                  z_dim=args.z_dim,
                  gf_dim=64, df_dim=64,
                  c_bits=args.c_bits)
    dcgan.complete(args)

print 'stop'