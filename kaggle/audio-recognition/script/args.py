# encoding=utf8

import argparse


FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str, default='0', help='which fold')
    FLAGS, _ = parser.parse_known_args()

    print(FLAGS.fold)
