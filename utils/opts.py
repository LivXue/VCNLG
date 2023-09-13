from __future__ import print_function
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Data settings
    parser.add_argument('--gpu', type=str, default='4', help='id of GPU')
    parser.add_argument('--dataset', type=str, default='VIST-E',
                        help='dataset: VIST-E / LSMDC-E')
    parser.add_argument('--data_in_memory', action='store_true',
                        help='True if we want to save the features in memory')
    parser.add_argument('--src_vocab_dim', type=int, default=768,
                        help="dimension for source word embeddings")
    parser.add_argument('--tgt_vocab_dim', type=int, default=768,
                        help="dimension for target word embeddings")
    parser.add_argument('--input_txt_len', type=int, default=120,
                        help="max length of input text")
    parser.add_argument('--know_txt_len', type=int, default=50,
                        help="max length of knowledge text")
    parser.add_argument('--output_txt_len', type=int, default=40,
                        help="max length of output text")

    # Model settings
    parser.add_argument('--common_size', type=int, default=1024,
                        help='Multimodal common feature size')
    parser.add_argument('--img_fea_size', type=int, default=1024,
                        help='Last dimensionality of image convolutional feature')
    parser.add_argument('--num_head', type=int, default=8,
                        help='Number of heads in multi-head attention')

    # feature manipulation
    parser.add_argument('--use_box', type=bool, default=False,
                        help='If use box features')
    parser.add_argument('--use_know', type=bool, default=True,
                        help='If use knowledge features')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--grad_clip_mode', type=str, default='value',
                        help='value or norm')
    parser.add_argument('--grad_clip_value', type=float, default=0.1,
                        help='clip gradients at this value/max_norm, 0 means no clipping')
    parser.add_argument('--drop_prob_lm', type=float, default=0.1,
                        help='strength of dropout in the Language Model RNN')

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                        help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam|adamw')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=5,
                        help='at what epoch to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=1,
                        help='every how many epoches thereafter to drop learning rate?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.5,
                        help='rate of weight decaying')
    parser.add_argument('--optim_alpha', type=float, default=0.8,#0.5,
                        help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.99,#0.999,
                        help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight_decay')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='early stop number')

    # BERT settings
    parser.add_argument('--use_bert', type=bool, default=True,
                        help='If use BERT as the encoder.')
    parser.add_argument('--bert_architecture', type=str, default="bert-base-uncased",
                        help='BERT architecture.')
    parser.add_argument('--finetuning', type=int, default=5,
                        help='Beginning epoch of finetuning, -1 means never.')

    parser.add_argument('--reinforce_st_epoch', type=int, default=5,
                        help='Beginning epoch of reinforce learning, -1 means never.')

    # misc
    parser.add_argument('--id', type=str, default='5',
                        help='an id identifying this run/job. used in cross-val and appended when writing progress files')

    args = parser.parse_args()

    # Paths for the dataset
    args.input_dir = './data/' + args.dataset
    args.img_dir = './data/' + args.dataset + '/ViT_features'
    args.clip_dir = './data/' + args.dataset + '/clip_features'
    args.conv_ext = '.npy'
    args.input_box_dir = ''
    args.img_len = 197 if args.dataset == "VIST-E" else 1379

    # Check if args are valid
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert 1 > args.drop_prob_lm >= 0, "drop_prob_lm should be between 0 and 1"

    # default value for checkpoint_path
    args.checkpoint_path = 'log/' + args.dataset + '/log_%s' % args.id

    return args
