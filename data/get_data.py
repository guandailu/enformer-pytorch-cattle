import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import os
import json
import functools
import torch

def deserialize(serialized_example, metadata):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)

    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (metadata['target_length'], metadata['num_targets']))
    target = tf.cast(target, tf.float32)

    return {'sequence': sequence,
            'target': target}

###### load metadata
data_path="/group/zhougrp4/dguan/BovineFAANG/23_deep_learning/basenji/BovineFAANG_data/ATAC/data/"
path = os.path.join(data_path, 'statistics.json')
with tf.io.gfile.GFile(path, 'r') as f:
    metadata=json.load(f)
print(metadata)

###### load data
subset='train'
num_threads=4
def tfrecord_files(subset):
    # Sort the values by int(*).
    return sorted(tf.io.gfile.glob(os.path.join(
        data_path, 'tfrecords', f'{subset}-*.tfr'
    )), key=lambda x: int(x.split('-')[-1].split('.')[0]))

dataset = tf.data.TFRecordDataset(tfrecord_files( subset),
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_threads)
loaded_dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                          num_parallel_calls=num_threads)
# Load dataset
cattle_dataset = loaded_dataset.batch(1).repeat()


n_examples = 10

for name, dataset in [("cattle", cattle_dataset)]:
    it = iter(dataset)

    sequence = []
    target = []
    for _ in tqdm(range(n_examples)):
        example = next(it)
        sequence.append(torch.from_numpy(example["sequence"].numpy()))
        target.append(torch.from_numpy(example["target"].numpy()))

    sequence = torch.cat(sequence, dim=0)
    target = torch.cat(target, dim=0)

    assert sequence.shape == torch.Size((n_examples, 131072, 4))
    
    out = {"sequence": sequence, "target": target}
    torch.save(out, f"data/example_data_{name}.pt")
