# Created by pengsheng.huang at 8/14/2020

import os
import numpy as np
import config
import cv2
import tensorflow as tf
from glob import glob


def load_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [128, 128])
    img /= 255.0
    return img


def load_preprocess_image_with_label(path, label):
    return load_preprocess_image(path), label


def build_dataset(data_path, positive_only=True):
    if positive_only:
        all_image_paths = glob(os.path.join(data_path, "posi*/*"))
        data_set = tf.data.Dataset.from_tensor_slices(all_image_paths)
        data_set = data_set.map(load_preprocess_image)
        data_set = data_set.shuffle(300)
        return data_set
    else:
        all_image_paths = glob(os.path.join(data_path, '*/*'))
        all_label_names = ['positive' if 'positive' in x else 'negative' for x in all_image_paths]
        label_to_index = {
            'positive': 0,
            'negative': 1
        }
        all_labels = [label_to_index[x] for x in all_label_names]
        data_set = tf.data.Dataset.from_tensor_slices((all_image_paths, all_labels))
        data_set = data_set.map(load_preprocess_image_with_label)
        data_set = data_set.shuffle(300)
        return data_set


if __name__ == '__main__':
    ds = build_dataset("/data2/huangps/data_freakie/gx/data/gonet_original/data_train_annotation", positive_only=False)
    ds = ds.batch(config.BATCH_SIZE)
    for sample in iter(ds):
        image = sample[0].numpy()[0]
        print(image)
        print(sample)
        print(image.shape)
        cv2.imshow("image", image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
