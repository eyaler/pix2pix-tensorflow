import argparse
import os
import shutil

from model import pix2pix
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='facades', help='name of the dataset')
parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='# of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
#parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flips', dest='flips', type=int, default=True, help='use flips for data argumentation: 1: true, 0: false')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
#parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=100, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--sample_freq', dest='sample_freq', type=int, default=100, help='save the current sample results every sample_freq_iterations)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=1, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=int, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=int, default=False, help='1: takes images in order to make batches, 0: takes them randomly')
#parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=int, default=True, help='iter into serial image list: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='validation samples are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test samples are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')
parser.add_argument('--rotations', dest='rotations', type=int, default=False, help='use rotations for data augmentation: 1: true, 0: false')
parser.add_argument('--keep_aspect_ratio', dest='keep_aspect', type=int, default=False, help='keep aspect ratio when scaling image: 1: true, 0: false')
parser.add_argument('--pad_to_white', dest='pad_to_white', type=int, default=False, help='when keeping aspect ratio should we pad to white? 1: true, 0: false')
parser.add_argument('--gcn', dest='gcn', type=int, default=False, help='global contrast normalization preprocessing for source image: 1: true, 0: false')
parser.add_argument('--interp', dest='interp', type=int, default=True, help='use interpolation when resizing target image: 1: true, 0: false')
parser.add_argument('--acc_threshold', dest='acc_threshold', type=float, default=0.5, help='background probability threshold for accuracy calculation: 1: true, 0: false')

args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    with tf.Session() as sess:
        model = pix2pix(sess, batch_size=args.batch_size, load_size=args.load_size, fine_size=args.fine_size,
                        dataset_name=args.dataset_name, which_direction=args.which_direction, checkpoint_dir=args.checkpoint_dir,
                        load_model=args.continue_train, gf_dim=args.ngf, df_dim=args.ndf, L1_lambda=args.L1_lambda,
                        input_c_dim=args.input_nc, output_c_dim=args.output_nc, flips=args.flips,
                        rotations=args.rotations, keep_aspect=args.keep_aspect, pad_to_white=args.pad_to_white, gcn=args.gcn, interp=args.interp, acc_threshold=args.acc_threshold)

        if args.phase == 'train':
            shutil.rmtree(args.sample_dir, ignore_errors=True)
            os.makedirs(args.sample_dir)
            model.train(args)

        shutil.rmtree(args.test_dir, ignore_errors=True)
        os.makedirs(args.test_dir)
        model.test(args)

if __name__ == '__main__':
    tf.app.run()
