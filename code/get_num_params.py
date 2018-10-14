#!/usr/bin/env python

# Returns parameter counts EXCLUDING optimizer parameters!
import sys

if len(sys.argv) == 2:
    ckpt_fpath = sys.argv[1]
else:
    print('Usage: python count_ckpt_param.py path-to-ckpt')
    sys.exit(1)

import tensorflow as tf
import numpy as np

# Open TensorFlow ckpt
reader = tf.train.NewCheckpointReader(ckpt_fpath)

print('\nCount the number of parameters in ckpt file (%s)' % ckpt_fpath)
param_map = reader.get_variable_to_shape_map()
total_count = 0
total_size_map = dict()
for k, v in param_map.items():
    if 'optimizer' not in k and 'global_step' not in k:
        temp = np.prod(v)
        total_count += temp
        print('%s: %s => %d' % (k, str(v), temp))
        total_size_map[k] = (temp)

print('Total Param Count: %d' % total_count)

import operator
top_5 = sorted(total_size_map.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]
print('Top 5 Parameter Sizes:')
for item in top_5:
    print item[0], param_map[item[0]], item[1]
