# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold, decomposition
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import gaussian_kde

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(data,limit = np.inf):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """


  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')

    data_features = np.zeros((min(limit, data.shape[0]),2048))
    for index, row in data.iterrows():
      if index==limit:
            break
      image = './images/' + str(row['image_id']) + '.jpg'
      if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
      image_data = tf.gfile.FastGFile(image, 'rb').read()
      features = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
      data_features[index] = np.squeeze(features)
    #image = (FLAGS.image_file if FLAGS.image_file else
     #      os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))


    return data_features

    # Creates node ID --> English string lookup.
    #node_lookup = NodeLookup()

    #top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
   # for node_id in top_k:
     # human_string = node_lookup.id_to_string(node_id)
      #score = predictions[node_id]
      #print('%s (score = %.5f)' % (human_string, score))


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def imscatter(x, y, image, ax=None, label=False):
    label=label==True
    im = OffsetImage(image)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=label)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def main(_):
  maybe_download_and_extract()

  data = pd.read_csv('final_train.csv')
  n=7000
  X = run_inference_on_image(data,n)
  y = data['image_label'].values[:X.shape[0]].ravel()

  X = decomposition.PCA(n_components=50).fit_transform(X)
  tsne = manifold.TSNE()
  X = tsne.fit_transform(X)

  fig, ax = plt.subplots()
  scale_factor=15
  fig.set_size_inches(16*scale_factor, 9*scale_factor, forward=True)

  for index, row in data.iterrows():
      if index==n:
          break
      image_path = './images/' + str(row['image_id']) + '.jpg'
      image=cv2.imread(image_path)
      b,g,r = cv2.split(image)       # get b,g,r
      image = cv2.merge([r,g,b])     # switch it to rgb
      mx=min(80/image.shape[0],80/image.shape[1])
      image=cv2.resize(image, (0, 0), image, mx, mx)
      x1=X[index, 0]
      x2=X[index, 1]
      imscatter(x1, x2, image, ax)
      ax.plot(x1, x2)

  x1=X[y == 1][:, 0]
  x2=X[y == 1][:, 1]
  xy = np.vstack([x1,x2])
  kde = gaussian_kde(xy)#simple density estimation
  z = kde(xy)

  xmin, xmax = ax.get_xlim()
  ymin, ymax = ax.get_ylim()

  xedges = np.linspace(xmin, xmax, 700)
  yedges = np.linspace(ymin, ymax, 700)
  xx, yy = np.meshgrid(xedges, yedges)
  gridpoints = np.array([xx.ravel(), yy.ravel()])

  zz = np.reshape(kde(gridpoints), xx.shape)
  im = ax.imshow(zz, cmap='jet', interpolation='nearest', origin='lower', extent=[xmin, xmax, ymin, ymax])

  fig.colorbar(im)
  ax.grid()
  fig.savefig('7000.png', dpi=100)

  print("feature extracted")


if __name__ == '__main__':
  tf.app.run()
