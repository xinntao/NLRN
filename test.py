from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import importlib


from hashlib import sha256
import numpy as np

from PIL import Image


import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', help='GCS location to load exported model', required=True)
parser.add_argument('--input-dir', help='GCS location to load input images', required=True)
parser.add_argument('--output-dir', help='GCS location to load output images', required=True)
parser.add_argument(
    '--noise-sigma', help='Scale for image super-resolution', default=25, type=float)
parser.add_argument(
    '--patch-size', help='Number of pixels in height or width of patches', default=43, type=int)
args = parser.parse_args()

with tf.Session(graph=tf.Graph()) as sess:
    metagraph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                                args.model_dir)
    signature_def = metagraph_def.signature_def[
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    input_tensor = sess.graph.get_tensor_by_name(signature_def.inputs['inputs'].name)
    output_tensor = sess.graph.get_tensor_by_name(signature_def.outputs['output'].name)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    psnr_list = []
    for input_file in os.listdir(args.input_dir):
        print(input_file)
        sha = sha256(input_file.encode('utf-8'))
        seed = np.frombuffer(sha.digest(), dtype='uint32')
        rstate = np.random.RandomState(seed)

        output_file = os.path.join(args.output_dir, input_file)
        input_file = os.path.join(args.input_dir, input_file)
        input_image = np.asarray(Image.open(input_file))
        input_image = input_image.astype(np.float32) / 255.0

        def forward_images(images):
            images = output_tensor.eval(feed_dict={input_tensor: images})
            return images

        stride = 7
        h_idx_list = list(range(0, input_image.shape[0] - args.patch_size,
                                stride)) + [input_image.shape[0] - args.patch_size]
        w_idx_list = list(range(0, input_image.shape[1] - args.patch_size,
                                stride)) + [input_image.shape[1] - args.patch_size]
        output_image = np.zeros(input_image.shape)
        overlap = np.zeros(input_image.shape)
        noise_image = input_image + rstate.normal(0, args.noise_sigma / 255.0,
                                                    input_image.shape)
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                # print(h_idx, w_idx)
                input_patch = noise_image[h_idx:h_idx + args.patch_size, w_idx:
                                            w_idx + args.patch_size]
                input_patch = np.expand_dims(input_patch, axis=-1)
                input_patch = np.expand_dims(input_patch, axis=0)
                output_patch = forward_images(input_patch)
                output_patch = output_patch[0, :, :, 0]
                output_image[h_idx:h_idx + args.patch_size, w_idx:
                                w_idx + args.patch_size] += output_patch
                overlap[h_idx:h_idx + args.patch_size, w_idx:w_idx + args.patch_size] += 1
        output_image /= overlap

        def psnr(im1, im2):
            im1_uint8 = np.rint(np.clip(im1 * 255, 0, 255))
            im2_uint8 = np.rint(np.clip(im2 * 255, 0, 255))
            diff = np.abs(im1_uint8 - im2_uint8).flatten()
            rmse = np.sqrt(np.mean(np.square(diff)))
            psnr = 20 * np.log10(255.0 / rmse)
            print(psnr)
            return psnr

        psnr_list.append(psnr(output_image, input_image))
        output_image = np.around(output_image * 255.0).astype(np.uint8)
        output_image = Image.fromarray(output_image)
        output_image.save(output_file)
    print('PSNR: ', np.average(np.array(psnr_list)))
