'''
MIT License

Copyright (c) 2016 Ankesh Anand

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import tensorflow as tf

from argparse import ArgumentParser
from src.model import CryptoNet
from src.config import *


def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--msg-len', type=int,
                        dest='msg_len', help='message length',
                        metavar='MSG_LEN', default=MSG_LEN)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='Number of Epochs in Adversarial Training',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    with tf.Session() as sess:
        crypto_net = CryptoNet(sess, msg_len=options.msg_len, epochs=options.epochs,
                               batch_size=options.batch_size, learning_rate=options.learning_rate)

        crypto_net.train()

if __name__ == '__main__':
    main()
