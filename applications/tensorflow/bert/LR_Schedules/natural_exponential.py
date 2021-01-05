# coding=utf-8
# Copyright (c) 2020 Graphcore Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been added by Graphcore Ltd.

import tensorflow as tf
import math


class LearningRate:
    def __init__(self, opts, total_iterations):
        self.init_lr = opts["base_learning_rate"]
        self.decay_step = opts['decay_steps']
        self.decay_rate = opts['decay_rate']
        self.warmup_steps = 0
        self.warmup_steps = opts['warmup_steps']

    def feed_dict_lr(self, step, sess):
        with sess.graph.as_default():
            learing_rate = tf.train.natural_exp_decay(learning_rate=self.init_lr,
                                                      global_step=step-self.warmup_steps,
                                                      decay_steps=self.decay_step,
                                                      decay_rate=self.decay_rate,
                                                      staircase=True)
        lr = sess.run(learing_rate)

        if step < self.warmup_steps:
            return (step * self.init_lr) / self.warmup_steps
        else:
            return lr


def add_arguments(parser):
    lr_group = parser.add_argument_group('Natural Exponential Decay Learning Rate. Use with --lr-schedule exponential.')
    lr_group.add_argument('--warmup-steps', type=float, default=0,
                          help="Warmup length in steps (Default=0, set to 0 for no warmup)")
    lr_group.add_argument('--decay-steps', type=int, default=512,
                          help="Define the attenuation period, which can be matched with the \
                          parameter staircase in decay_. Keep the learning rate unchanged in step training rounds")
    lr_group.add_argument('--decay-rate', type=float, default=0.051,
                          help="The rate to decay (Default=0.051)")
    return parser


def set_defaults(opts):
    opts['summary_str'] += "Natural Exponential LR schedule\n"
    if opts["warmup_steps"] > 0:
        opts['summary_str'] += " Warmup: {warmup_steps} steps\n"
    else:
        opts['summary_str'] += " No warmup\n"
    opts['summary_str'] += "Natural exponential decay applied to learning rate with exponent {decay_rate}. "
    opts['summary_str'] += "Initial rate of {base_learning_rate}, decay learning rate after {decay_steps} steps.\n"
    return opts
