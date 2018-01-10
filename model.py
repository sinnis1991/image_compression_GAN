# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
#   + License: MIT
# [2016-08-05] Modifications for Completion: Brandon Amos (http://bamos.github.io)
#   + License: MIT

from __future__ import division
import os
import time
import math
import itertools
import pickle
from glob import glob
import tensorflow as tf
from six.moves import xrange

from ops import *
from utils import *

from extract_data import *
# from svhn import deepnn, conv2d_S, max_pool_2x2, weight_variable, bias_variable
# import svhn
from svhn import *

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]

def dataset_files(root):
    """Returns a list of all image files in the given directory"""
    return list(itertools.chain.from_iterable(
        glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))


class DCGAN(object):
    def __init__(self, sess, image_size=64, is_crop=False,
                 batch_size=64, sample_size=64,
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3,
                 checkpoint_dir=None, classifier_checkpoint_dir=None,
                 lam=0.1, lam_c =0.1,
                 c_bits=32):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            lowres: (optional) Low resolution image/mask shrink factor. [8]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        # Currently, image size must be a (power of 2) and (8 or higher).
        assert(image_size & (image_size - 1) == 0 and image_size >= 8)

        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, c_dim]

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.lam = lam
        self.lam_c = lam_c

        self.c_dim = c_dim
        self.c_bits = c_bits

        # batch normalization : deals with poor initialization helps gradient flow !batch norm is not an initializer!
        self.d_bns = [
            batch_norm(name='d_bn{}'.format(i, )) for i in range(1, 4)]      #batch_norm(name='d_bn{}'.format(i,)) for i in range(4)]

        # for the transpose conv layer, it should range from the imagesize(64X4) to 8X8 by half its length by each layer
        log_size = int(math.log(image_size) / math.log(2))
        self.g_bns = [
            batch_norm(name='g_bn{}'.format(i, )) for i in range(log_size - 1)]        #batch_norm(name='g_bn{}'.format(i,)) for i in range(log_size)]

        self.checkpoint_dir = checkpoint_dir
        self.classifier_checkpoint_dir = classifier_checkpoint_dir  #classifier_checkpoint_dir
        self.build_model()

        self.model_name = "DCGAN.model"

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        #here the [None] means the tf have no idea the bach_size is
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        self.G = self.generator(self.z)

        self.D, self.D_logits = self.discriminator(self.images)

        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                    labels=tf.ones_like(self.D)))  #change lables to targets
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.zeros_like(self.D_))) #change lables to targets
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.ones_like(self.D_))) #change lables to targets

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=1)


        # Completion.
        self.contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(self.G - self.images)), 1)

        self.perceptual_loss = self.g_loss
        self.complete_loss = self.contextual_loss  + self.lam*self.perceptual_loss
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

        # Classifier
        # Classifier
        # # Create the model
        # self.x = tf.placeholder(tf.float32, [None, 32,32,3])
        #
        # # Define loss and optimizer
        # self.y_ = tf.placeholder(tf.float32, [None, 10])
        #
        # # Build the graph for the deep net
        # self.y_conv, self.keep_prob = deepnn(self.x)
        #
        # with tf.name_scope('loss'):
        #     self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
        #                                                             logits=self.y_conv)
        # self.cross_entropy = tf.reduce_mean(self.cross_entropy)
        #
        # with tf.name_scope('adam_optimizer'):
        #     self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        #
        # with tf.name_scope('accuracy'):
        #     self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        #     self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
        # self.accuracy = tf.reduce_mean(self.correct_prediction)
        #
        # classifier_V = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='svhn_classifier')
        # self.saver2 = tf.train.Saver(classifier_V)
        # classifier_V2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='BatchNorm')
        # self.saver3 = tf.train.Saver(classifier_V2)
        # classifier_V3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='BatchNorm_1')
        # self.saver4 = tf.train.Saver(classifier_V3)
        # classifier_V4 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='BatchNorm_2')
        # self.saver5 = tf.train.Saver(classifier_V4)


        # self.x = tf.placeholder(tf.float32, [None, 32,32,3])
        self.y_ = tf.placeholder(tf.float32, [None, 10])

        self.y_conv_g,self.keep_prob_g = deepnn(self.G*127.5+127.5)
        self.y_conv,self.keep_prob = deepnn(self.images,reuse=True)

        # self.y_conv,self.keep_prob_g = deepnn(self.x,reuse=True)



        with tf.name_scope('loss'):
            self.cross_entropy_g = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                                         logits=self.y_conv_g)
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                                         logits=self.y_conv)
        with tf.name_scope('accuracy'):
            self.y_conv_label = tf.argmax(self.y_conv, 1)
            self.y_g_label = tf.argmax(self.y_conv_g, 1)
            self.y_label = tf.argmax(self.y_, 1)
            self.correct_prediction = tf.equal(self.y_conv_label, self.y_label)
            self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_prediction)

        # self.cross_entropy_g = tf.reduce_mean(self.cross_entropy_g)
        # self.cross_entropy = tf.reduce_mean(self.cross_entropy)
        # self.cross_entropy_S =tf.reduce_mean(tf.abs(self.y_conv - self.y_conv_g),axis=1)

        # self.classifier_loss = tf.abs(self.cross_entropy_g-self.cross_entropy)
        self.classifier_loss = self.cross_entropy_g

        self.complete_loss_Special = self.contextual_loss +self.lam * self.perceptual_loss+self.lam_c*self.classifier_loss
        self.grad_complete_loss_Special = tf.gradients(self.complete_loss_Special, self.z)



        classifier_V = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='svhn_classifier')
        self.saver2 = tf.train.Saver(classifier_V)

        tvars = tf.trainable_variables()
        for var in tvars:
            print(var.name)



    def train(self, config): #change configpu to config
        data = dataset_files(config.dataset)
        np.random.shuffle(data)
        assert(len(data) > 0)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = tf.summary.merge(
            [self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        sample_files = data[0:self.sample_size]

        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print("""

======
An existing model was found in the checkpoint directory.
If you just cloned this repository, it's a model for faces
trained on the CelebA dataset for 20 epochs.
If you want to train a new model from scratch,
delete the checkpoint directory or specify a different
--checkpoint_dir argument.
======

""")
        else:
            print("""

======
An existing model was not found in the checkpoint directory.
Initializing a new one.
======

""")

        for epoch in xrange(config.epoch):
            data = dataset_files(config.dataset)
            batch_idxs = min(len(data), config.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                         for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                    feed_dict={ self.images: batch_images, self.z: batch_z, self.is_training: True })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z, self.is_training: True })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z, self.is_training: True })
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.is_training: False})
                errD_real = self.d_loss_real.eval({self.images: batch_images, self.is_training: False})
                errG = self.g_loss.eval({self.z: batch_z, self.is_training: False})

                counter += 1
                print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                    epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, batch_idxs//2) == 0 or (epoch == config.epoch-1 and idx == batch_idxs-1) :
                    samples, d_loss, g_loss = self.sess.run(
                        [self.G, self.d_loss, self.g_loss],
                        feed_dict={self.z: sample_z, self.images: sample_images, self.is_training: False}
                    )
                    save_images(samples, [8, 8],
                                config.sample_dir+'/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

                if np.mod(counter, batch_idxs//2) == 0 or (epoch == config.epoch-1 and idx == batch_idxs-1):
                    self.save(config.checkpoint_dir, counter)


    def complete(self, config):
        def make_dir(name):
            # Works on python 2.7, where exist_ok arg to makedirs isn't available.
            p = os.path.join(config.outDir, name)
            if not os.path.exists(p):
                os.makedirs(p)
        make_dir('decompressed_imgs')
        make_dir('logs')

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)

        self.saver2.restore(self.sess,'../SVHN_classifier/checkpoint_SuperSuper/model.ckpt')
        # self.saver3.restore(self.sess,'../SVHN_classifier/checkpoint/model.ckpt')
        # self.saver4.restore(self.sess,'../SVHN_classifier/checkpoint/model.ckpt')
        # self.saver5.restore(self.sess,'../SVHN_classifier/checkpoint/model.ckpt')


        # isClassifierLoad = self.load(self.classifier_checkpoint_dir)
        # assert(isLoaded)

        # self.saver2 = tf.train.Saver({"svhn_classifier/conv1"})
        # self.saver2.restore(self.sess,'../SVHN_classifier/checkpoint/model.ckpt')
        # tvars = tf.trainable_variables()
        # tvars_vals = self.sess.run(tvars)
        # for var, val in zip(tvars, tvars_vals):
        #     print(var.name)

        nImgs = len(config.imgs)

        batch_idxs = int(np.ceil(nImgs/self.batch_size))

        for idx in xrange(0, batch_idxs):

            l = idx*self.batch_size
            u = min((idx+1)*self.batch_size, nImgs)
            batchSz = u-l
            batch_files = config.imgs[l:u]
            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                     for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            if batchSz < self.batch_size:
                padSz = ((0, int(self.batch_size-batchSz)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)

            nRows = np.ceil(batchSz / 8)
            nCols = min(8, batchSz)
            save_images(batch_images[:batchSz, :, :, :], [nRows, nCols],
                        os.path.join(config.outDir, 'ori_img_{}-{}.png'.format(l+1,u)))
            """
                is the test for 
            """

            im2, label2 = extract('../SVHN_classifier', 'test_32x32.mat')

            accuracy_sum = 0
            for k in range(im2.shape[0] // batch_size):
                accuracy_sum += self.accuracy.eval(feed_dict={
                    self.images: im2[self.batch_size * k:self.batch_size * (k + 1)],
                    self.y_: label2[self.batch_size * k:self.batch_size * (k + 1)],
                    self.keep_prob: 1.0})

            print('epoch %d test accuracy %g' % (idx, accuracy_sum / (im2.shape[0] // batch_size)))
            #
            # feed_im = im2[:64] #[l+64*hh:u+64*hh]
            # feed_label = label2[:64] #[l+64*hh:u+64*hh]
            # # print im2[0]
            # train_accuracy = self.accuracy.eval(feed_dict={
            #     self.x: feed_im, self.y_: feed_label, self.keep_prob: 1.0})

            # print ("classifier accuracy:%g" % (accuracy_sum/(im2.shape[0]//batch_size)))

            loss = [0]*self.batch_size
            zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            m = 0
            v = 0
            former_iter = [0]*self.batch_size

            file_names =[]


            for i in range(self.batch_size):
                img_name = os.path.basename(batch_files[i])
                file_names.append(os.path.join(config.outDir,'logs/'+img_name[:-3]+'pkl'))
                if os.path.exists(file_names[i]):
                    with open(file_names[i],'rb')as f:
                        dict = pickle.load(f)
                    loss[i] = dict['loss'][-1]
                    zhats[i] = dict['zhat'][-1]
                    former_iter[i] = dict['iter'][-1]
                    m = dict['m'][-1]
                    v = dict['v'][-1]


                else:
                    print file_names[i]
                    dict = {
                        'loss': [0],
                        'con_loss':[0],
                        'cla_loss':[0],
                        'per_loss':[0],
                        'y_g_label':[0],
                        'iter': [0],
                        'zhat': [np.random.uniform(-1, 1, size=(self.z_dim))],
                        'm': [0],
                        'v': [0]

                    }
                    loss[i] = dict['loss'][-1]
                    zhats[i] = dict['zhat'][-1]
                    former_iter[i] = dict['iter'][-1]
                    f = open(file_names[i],"wb")
                    pickle.dump(dict,f)
                    f.close()

            for i in xrange(config.nIter):
                # fd = {
                #     self.z: zhats,
                #     self.images: batch_images,
                #     self.is_training: False #,
                #     # self.y_: label2[:64],
                #     # self.keep_prob: 1
                # }
                # run = [self.complete_loss, self.grad_complete_loss, self.G]
                # loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

                fd = {
                    self.z: zhats,
                    self.images: batch_images,
                    self.is_training: False,
                    self.y_: label2[:64],
                    self.keep_prob_g: 1,
                    self.keep_prob: 1
                }
                run = [self.contextual_loss,self.classifier_loss,self.perceptual_loss,
                       self.y_g_label,self.y_label,
                       self.complete_loss_Special, self.grad_complete_loss_Special, self.G]

                con_loss,cla_loss,per_loss,\
                y_g_label,y_label,\
                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

                if i == config.nIter - 1:
                  for img in range(batchSz):
                      with open(file_names[img], 'rb')as f:
                          dict = pickle.load(f)
                      dict['loss'].append(loss[img])
                      dict['con_loss'].append(con_loss[img])
                      dict['cla_loss'].append(cla_loss[img])
                      dict['per_loss'].append(per_loss)
                      dict['y_g_label'].append(y_g_label[img])
                      dict['iter'].append(former_iter[img]+config.nIter)
                      dict['zhat'].append(zhats[img])
                      iteration = dict['iter'][-1]
                      print(iteration, "img:{} loss:{} con_loss:{} cro_loss:{} per_loss:{} y_g:{} y:{}".format(img,
                                                                                            loss[img],
                                                                                            con_loss[img],
                                                                                            cla_loss[img],
                                                                                            per_loss,
                                                                                            y_g_label[img],
                                                                                            y_label[img]))
                      with open(file_names[img], 'wb')as f:
                          pickle.dump(dict,f)

                      if np.mod(img, batchSz)==1:
                        imgName = os.path.join(config.outDir,
                                             'decompressed_imgs/d_imgs_{}-{}_i{}_z{}_lam{}_lamc{}_{}bit.png'.format(l+1,u,dict['iter'][-1],
                                                                                                      self.z_dim,
                                                                                                      self.lam,self.lam_c,
                                                                                                        self.c_bits))
                        nRows = np.ceil(batchSz / 8)
                        nCols = min(8, batchSz)
                        save_images(G_imgs[:batchSz, :, :, :], [nRows,nCols], imgName)

                  # feed_im = G_imgs[:64]*127.5+127.5
                  # feed_label = label2[:64]
                  # train_accuracy = self.accuracy.eval(feed_dict={
                  #     self.x: feed_im, self.y_: feed_label, self.keep_prob: 1.0})
                  #
                  # print ("classifier accuracy:%g" % train_accuracy)

                if i % config.outInterval == 0 and i!=config.nIter-1:
                    with open(file_names[0], 'rb')as f:
                        dict = pickle.load(f)
                    iteration = dict['iter'][-1]+i
                    print(iteration, "loss:{} con_loss:{} cro_loss:{} per_loss:{}".format(np.mean(loss[0:batchSz]),
                                                                                          np.mean(con_loss[0:batchSz]),
                                                                                          np.mean(cla_loss[0:batchSz]),
                                                                                          np.mean(per_loss)))

                if config.approach == 'adam':
                    # Optimize single completion with Adam
                    m_prev = np.copy(m)
                    v_prev = np.copy(v)
                    m = config.beta1 * m_prev + (1 - config.beta1) * g[0]
                    v = config.beta2 * v_prev + (1 - config.beta2) * np.multiply(g[0], g[0])
                    if i == config.nIter - 1:
                        for img in range(self.batch_size):
                           with open(file_names[img], 'rb')as f:
                               dict = pickle.load(f)
                           dict['m'] = m
                           dict['v'] = v
                           with open(file_names[img], 'wb')as f:
                             pickle.dump(dict, f)

                    m_hat = m / (1 - config.beta1 ** (i + 1))
                    v_hat = v / (1 - config.beta2 ** (i + 1))
                    zhats += - np.true_divide(config.lr * m_hat, (np.sqrt(v_hat) + config.eps))
                    zhats = np.clip(zhats, -1, 1)
                    c_bits = self.c_bits
                    k = 2**c_bits-1
                    zhats = np.round(((zhats+1.0)/2.0)*k,0)/(k/2)-1

                elif config.approach == 'hmc':
                    # Sample example completions with HMC (not in paper)
                    zhats_old = np.copy(zhats)
                    loss_old = np.copy(loss)
                    v = np.random.randn(self.batch_size, self.z_dim)
                    v_old = np.copy(v)

                    for steps in range(config.hmcL):
                        v -= config.hmcEps/2 * config.hmcBeta * g[0]
                        zhats += config.hmcEps * v
                        np.copyto(zhats, np.clip(zhats, -1, 1))
                        loss, g, _, _ = self.sess.run(run, feed_dict=fd)
                        v -= config.hmcEps/2 * config.hmcBeta * g[0]

                    for img in range(batchSz):
                        logprob_old = config.hmcBeta * loss_old[img] + np.sum(v_old[img]**2)/2
                        logprob = config.hmcBeta * loss[img] + np.sum(v[img]**2)/2
                        accept = np.exp(logprob_old - logprob)
                        if accept < 1 and np.random.uniform() > accept:
                            np.copyto(zhats[img], zhats_old[img])

                    config.hmcBeta *= config.hmcAnneal

                else:
                    assert(False)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # TODO: Investigate how to parameterise discriminator based off image size.
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bns[0](conv2d(h0, self.df_dim*2, name='d_h1_conv'), self.is_training))
            h2 = lrelu(self.d_bns[1](conv2d(h1, self.df_dim*4, name='d_h2_conv'), self.is_training))
            h3 = lrelu(self.d_bns[2](conv2d(h2, self.df_dim*8, name='d_h3_conv'), self.is_training))
            h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h3_lin')            #h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')
    
            return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*4*4, 'g_h0_lin', with_w=True)
    
            # TODO: Nicer iteration pattern here. #readability
            hs = [None]
            hs[0] = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
            hs[0] = tf.nn.relu(self.g_bns[0](hs[0], self.is_training))

            i = 1 # Iteration number.
            depth_mul = 4  #8   # Depth decreases as spatial component increases.
            size = 8  # Size increases as depth decreases.

            while size < self.image_size:
                hs.append(None)
                name = 'g_h{}'.format(i)
                hs[i], _, _ = conv2d_transpose(hs[i-1],
                    [self.batch_size, size, size, self.gf_dim*depth_mul], name=name, with_w=True)
                hs[i] = tf.nn.relu(self.g_bns[i](hs[i], self.is_training))

                i += 1
                depth_mul //= 2
                size *= 2

            hs.append(None)
            name = 'g_h{}'.format(i)
            hs[i], _, _ = conv2d_transpose(hs[i - 1],
                [self.batch_size, size, size, 3], name=name, with_w=True)
    
            return tf.nn.tanh(hs[i])

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # print ckpt.model_checkpoint_path
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            # print ckpt.model_checkpoint_path
            self.true = True
            return self.true
        else:
            return False

