from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np

from ops import *
from utils import *

class pix2pix(object):
    def __init__(self, sess, batch_size, load_size, fine_size, dataset_name, which_direction, checkpoint_dir, load_model,
                 gf_dim, df_dim, L1_lambda, input_c_dim, output_c_dim, flips, rotations, keep_aspect, pad_to_white, gcn, interp, acc_threshold):

        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.load_model = load_model
        self.is_grayscale_A = (input_c_dim == 1 and which_direction=='AtoB') or (output_c_dim == 1 and which_direction=='BtoA')
        self.is_grayscale_B = (input_c_dim == 1 and which_direction=='BtoA') or (output_c_dim == 1 and which_direction=='AtoB')
        self.acc_threshold = acc_threshold

        self.batch_size = batch_size
        self.load_size = load_size
        self.image_size = fine_size
        self.output_size = fine_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda
        self.flips = flips
        self.rotations = rotations
        self.keep_aspect = keep_aspect
        self.pad_to_white = pad_to_white
        self.gcn = gcn
        self.interp = interp
        self.which_direction = which_direction

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):

        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        if self.which_direction=='AtoB':
            self.real_A = self.real_data[..., :self.input_c_dim]
            self.real_B = self.real_data[..., self.input_c_dim:]
        elif self.which_direction=='BtoA':
            self.real_A = self.real_data[..., self.output_c_dim:]
            self.real_B = self.real_data[..., :self.output_c_dim]
        else:
            raise ValueError('Bad direction: ' + self.which_direction)

        self.fake_B = self.generator(self.real_A)

        self.real_AB = tf.concat(3, [self.real_A, self.real_B])
        self.fake_AB = tf.concat(3, [self.real_A, self.fake_B])
        self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        self.fake_B_sample = self.sampler(self.real_A)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, 0.9*tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))

        fake_argmin = tf.select(tf.reduce_min(self.fake_B_sample, axis=3) < self.acc_threshold*2-1, tf.argmin(self.fake_B_sample, axis=3), -tf.ones(shape=tf.shape(self.real_B)[:-1], dtype=tf.int64))
        real_argmin = tf.select(tf.reduce_min(self.real_B, axis=3) < self.acc_threshold*2-1, tf.argmin(self.real_B, axis=3), -tf.ones(shape=tf.shape(self.real_B)[:-1], dtype=tf.int64))
        self.fake_threshold = tf.one_hot(fake_argmin, 3, on_value=-1, off_value=1)
        self.pixel_acc = tf.reduce_mean(tf.cast(tf.equal(fake_argmin, real_argmin), tf.float32))
        self.l1_loss = tf.reduce_mean(tf.abs(self.real_B - self.fake_B_sample))

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_))) \
                      + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def evaluate(self, input, output, threshold, which_direction, input_c_dim, output_c_dim, path):
        if which_direction == 'AtoB':
            real_A = input[..., :input_c_dim]
            real_B = input[..., input_c_dim:]
        elif which_direction == 'BtoA':
            real_A = input[..., output_c_dim:]
            real_B = input[..., output_c_dim]
        images = [real_A, real_B, threshold, output]
        for i in range(len(images)):
            images[i] = images[i][0]
            if images[i].shape[2] == 1:
                images[i] = np.repeat(images[i], 3, 2)

        save_images(images[2][None,...], [1, 1],
                    os.path.join(os.path.dirname(path), '_thresh_' + os.path.basename(path)))
        if which_direction == 'BtoA':
            images.reverse()
        save_images(np.asarray(images), [1, len(images)], os.path.join(os.path.dirname(path), '_eval_' + os.path.basename(path)))


    def load_random_samples(self):
        if not os.path.exists('./datasets/{}/val'.format(self.dataset_name)):
            return None
        val_files = glob('./datasets/{}/val/*.jpg'.format(self.dataset_name))+glob('./datasets/{}/val/*.png'.format(self.dataset_name))
        if not val_files:
            return None
        data = np.random.choice(val_files, self.batch_size)
        sample = [load_data(sample_file, load_size=self.load_size, fine_size=self.image_size, aspect=self.keep_aspect, pad_to_white=self.pad_to_white, which_direction=self.which_direction, gcn=self.gcn, interp=self.interp, flip=self.flips, rot=self.rotations, is_grayscale_A=self.is_grayscale_A, is_grayscale_B=self.is_grayscale_B) for sample_file in data]

        sample_images = np.array(sample).astype(np.float32)
        return sample_images

    def sample_model(self, sample_dir, epoch, idx):
        sample_images = self.load_random_samples()
        if sample_images is None:
            return
        samples, d_loss, g_loss, sample_threshold = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss, self.fake_threshold],
            feed_dict={self.real_data: sample_images}
        )
        path = './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx)
        self.evaluate(sample_images, samples, sample_threshold, self.which_direction, self.input_c_dim, self.output_c_dim, path)
        save_images(samples, [self.batch_size, 1], path)

        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self, args):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d__sum,
            self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load_model:
            if self.load(self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epochs):
            data = glob('./datasets/{}/train/*.jpg'.format(self.dataset_name))+glob('./datasets/{}/train/*.png'.format(self.dataset_name))
            if not args.serial_batches:
                np.random.shuffle(data)
            batch_idxs = min(len(data), args.train_size) // self.batch_size

            for idx in range(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_data(batch_file, load_size=self.load_size, fine_size=self.image_size, aspect=self.keep_aspect, pad_to_white=self.pad_to_white, which_direction=self.which_direction, gcn=self.gcn, interp=self.interp, flip=self.flips, rot=self.rotations, is_grayscale_A=self.is_grayscale_A, is_grayscale_B=self.is_grayscale_B) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                counter += 1

                if (args.print_freq>0 and np.mod(counter, args.print_freq) == 0) or (epoch==args.epochs-1 and counter==batch_idxs+1):
                    errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
                    errD_real = self.d_loss_real.eval({self.real_data: batch_images})
                    errG = self.g_loss.eval({self.real_data: batch_images})

                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, idx, batch_idxs,
                             time.time() - start_time, errD_fake + errD_real, errG))

                if (args.sample_freq>0 and np.mod(counter, args.sample_freq) == 0) or (epoch==args.epochs-1 and counter==batch_idxs+1):
                    self.sample_model(args.sample_dir, epoch, idx)

                if (args.save_latest_freq>0 and np.mod(counter, args.save_latest_freq) == 0) or (epoch==args.epochs-1 and counter==batch_idxs+1):
                    self.save(args.checkpoint_dir)

    def discriminator(self, image, y=None, reuse=False):
        # image is 256 x 256 x (input_c_dim + output_c_dim)
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
        # h3 is (16 x 16 x self.df_dim*8)
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4

    def generator(self, image, y=None):
        s = self.output_size

        s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
        # e1 is (128 x 128 x self.gf_dim)
        e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
        # e8 is (1 x 1 x self.gf_dim*8)

        self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
            [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
        d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
        d1 = tf.concat(3, [d1, e7])
        # d1 is (2 x 2 x self.gf_dim*8*2)

        self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
            [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
        d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
        d2 = tf.concat(3, [d2, e6])
        # d2 is (4 x 4 x self.gf_dim*8*2)

        self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
            [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
        d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
        d3 = tf.concat(3, [d3, e5])
        # d3 is (8 x 8 x self.gf_dim*8*2)

        self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
            [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
        d4 = self.g_bn_d4(self.d4)
        d4 = tf.concat(3, [d4, e4])
        # d4 is (16 x 16 x self.gf_dim*8*2)

        self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
            [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
        d5 = self.g_bn_d5(self.d5)
        d5 = tf.concat(3, [d5, e3])
        # d5 is (32 x 32 x self.gf_dim*4*2)

        self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
            [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
        d6 = self.g_bn_d6(self.d6)
        d6 = tf.concat(3, [d6, e2])
        # d6 is (64 x 64 x self.gf_dim*2*2)

        self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
            [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
        d7 = self.g_bn_d7(self.d7)
        d7 = tf.concat(3, [d7, e1])
        # d7 is (128 x 128 x self.gf_dim*1*2)

        self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
            [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(self.d8)

    def sampler(self, image, y=None):
        tf.get_variable_scope().reuse_variables()

        s = self.output_size

        s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
        # e1 is (128 x 128 x self.gf_dim)
        e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
        # e8 is (1 x 1 x self.gf_dim*8)

        self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
            [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
        d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
        d1 = tf.concat(3, [d1, e7])
        # d1 is (2 x 2 x self.gf_dim*8*2)

        self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
            [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
        d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
        d2 = tf.concat(3, [d2, e6])
        # d2 is (4 x 4 x self.gf_dim*8*2)

        self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
            [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
        d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
        d3 = tf.concat(3, [d3, e5])
        # d3 is (8 x 8 x self.gf_dim*8*2)

        self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
            [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
        d4 = self.g_bn_d4(self.d4)
        d4 = tf.concat(3, [d4, e4])
        # d4 is (16 x 16 x self.gf_dim*8*2)

        self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
            [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
        d5 = self.g_bn_d5(self.d5)
        d5 = tf.concat(3, [d5, e3])
        # d5 is (32 x 32 x self.gf_dim*4*2)

        self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
            [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
        d6 = self.g_bn_d6(self.d6)
        d6 = tf.concat(3, [d6, e2])
        # d6 is (64 x 64 x self.gf_dim*2*2)

        self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
            [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
        d7 = self.g_bn_d7(self.d7)
        d7 = tf.concat(3, [d7, e1])
        # d7 is (128 x 128 x self.gf_dim*1*2)

        self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
            [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(self.d8)

    def save(self, checkpoint_dir, step=None):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        sample_files = sorted(glob('./datasets/{}/test/*.jpg'.format(self.dataset_name))+glob('./datasets/{}/test/*.png'.format(self.dataset_name)))

        # load testing input
        print("Loading testing images ...")
        sample = [load_data(sample_file, load_size=self.load_size, fine_size=self.image_size, aspect=self.keep_aspect, pad_to_white=self.pad_to_white, which_direction=self.which_direction, gcn=self.gcn, interp=self.interp, flip=self.flips, rot=self.rotations, is_test=True, is_grayscale_A=self.is_grayscale_A, is_grayscale_B=self.is_grayscale_B) for sample_file in sample_files]

        sample_images = np.array(sample).astype(np.float32)

        sample_images = [sample_images[i:i+self.batch_size]
                         for i in range(0, len(sample_images), self.batch_size)]
        sample_images = np.array(sample_images)

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        l1_loss_sum = 0
        pixel_acc_sum = 0
        for i, sample_image in enumerate(sample_images):
            idx = i+1
            print("sampling image ", idx)
            samples, l1_loss, pixel_acc, sample_threshold = self.sess.run(
                [self.fake_B_sample, self.l1_loss, self.pixel_acc, self.fake_threshold],
                feed_dict={self.real_data: sample_image}
            )
            l1_loss_sum += l1_loss
            pixel_acc_sum += pixel_acc
            if self.batch_size==1:
                path = './{}/{}'.format(args.test_dir, os.path.basename(sample_files[i].replace('.jpg','.png')))
                self.evaluate(sample_image, samples, sample_threshold, self.which_direction, self.input_c_dim, self.output_c_dim, path)
            else:
                path = './{}/test_{:04d}.png'.format(args.test_dir, idx)
            save_images(samples, [self.batch_size, 1], path)

        print(l1_loss_sum/len(sample_images), pixel_acc_sum/len(sample_images))