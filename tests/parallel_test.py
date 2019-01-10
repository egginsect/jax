# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp
from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as np
from jax import test_util as jtu
from jax.api import spmd
from jax.api import papply
from jax.interpreters.parallel import collective_sum

from jax.config import config
config.parse_flags_with_absl()


class SpmdTest(jtu.JaxTestCase):

  def testConstantFunction(self):
    f = lambda x: 3
    ans = spmd(f, axis_name='batch')(onp.ones(4))
    expected = 3 * onp.ones(4)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testReduceSum(self):
    f = lambda x: collective_sum(x, 'batch')
    ans = spmd(f, axis_name='batch')(onp.ones(4))
    expected = 4 * onp.ones(4)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testLogSoftmax(self):

    def f(x):
      return x - np.log(collective_sum(np.exp(x), 'batch'))

    x = onp.log(onp.arange(10., dtype=onp.float32))

    ans = spmd(f, axis_name='batch')(x)
    expected = x - onp.log(onp.sum(onp.exp(x)))
    self.assertAllClose(ans, expected, check_dtypes=False)


class PapplyTest(jtu.JaxTestCase):

  def testIdentity(self):
    ans = papply(lambda x: x)(onp.arange(3))
    expected = onp.arange(3)
    self.assertAllClose(ans, expected, check_dtypes=False)


if __name__ == '__main__':
  absltest.main()
