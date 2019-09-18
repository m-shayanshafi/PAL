# Copyright 2018, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements DPQuery interface for Gaussian average queries.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.version import LooseVersion
import tensorflow as tf

from privacy.dp_query import dp_query
from privacy.dp_query import normalized_query

if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
  nest = tf.contrib.framework.nest
else:
  nest = tf.nest


class GaussianSumQuery(dp_query.SumAggregationDPQuery):
  """Implements DPQuery interface for Gaussian sum queries.

  Accumulates clipped vectors, then adds Gaussian noise to the sum.
  """

  def __init__(self, l2_norm_clip, stddev, ledger=None):
    """Initializes the GaussianSumQuery.

    Args:
      l2_norm_clip: The clipping norm to apply to the global norm of each
        record.
      stddev: The stddev of the noise added to the sum.
      ledger: The privacy ledger to which queries should be recorded.
    """
    self._l2_norm_clip = tf.cast(l2_norm_clip, tf.float32)
    self._stddev = tf.cast(stddev, tf.float32)
    self._ledger = ledger

  def derive_sample_params(self, global_state):
    return self._l2_norm_clip

  def initial_sample_state(self, global_state, template):
    if self._ledger:
      dependencies = [
          self._ledger.record_sum_query(self._l2_norm_clip, self._stddev)
      ]
    else:
      dependencies = []
    with tf.control_dependencies(dependencies):
      return nest.map_structure(
          dp_query.zeros_like, template)

  def preprocess_record_impl(self, params, record):
    """Clips the l2 norm, returning the clipped record and the l2 norm.

    Args:
      params: The parameters for the sample.
      record: The record to be processed.

    Returns:
      A tuple (preprocessed_records, l2_norm) where `preprocessed_records` is
        the structure of preprocessed tensors, and l2_norm is the total l2 norm
        before clipping.
    """
    l2_norm_clip = params
    record_as_list = nest.flatten(record)
    clipped_as_list, norm = tf.clip_by_global_norm(record_as_list, l2_norm_clip)
    return nest.pack_sequence_as(record, clipped_as_list), norm

  def preprocess_record(self, params, record):
    preprocessed_record, _ = self.preprocess_record_impl(params, record)
    return preprocessed_record

  def get_noised_result(self, sample_state, global_state):
    """See base class."""
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
      def add_noise(v):
        return v + tf.random_normal(tf.shape(v), stddev=self._stddev)
    else:
      random_normal = tf.random_normal_initializer(stddev=self._stddev)
      def add_noise(v):
        return v + random_normal(tf.shape(v))

    return nest.map_structure(add_noise, sample_state), global_state


class GaussianAverageQuery(normalized_query.NormalizedQuery):
  """Implements DPQuery interface for Gaussian average queries.

  Accumulates clipped vectors, adds Gaussian noise, and normalizes.

  Note that we use "fixed-denominator" estimation: the denominator should be
  specified as the expected number of records per sample. Accumulating the
  denominator separately would also be possible but would be produce a higher
  variance estimator.
  """

  def __init__(self,
               l2_norm_clip,
               sum_stddev,
               denominator,
               ledger=None):
    """Initializes the GaussianAverageQuery.

    Args:
      l2_norm_clip: The clipping norm to apply to the global norm of each
        record.
      sum_stddev: The stddev of the noise added to the sum (before
        normalization).
      denominator: The normalization constant (applied after noise is added to
        the sum).
      ledger: The privacy ledger to which queries should be recorded.
    """
    super(GaussianAverageQuery, self).__init__(
        numerator_query=GaussianSumQuery(l2_norm_clip, sum_stddev, ledger),
        denominator=denominator)
