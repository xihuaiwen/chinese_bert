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
import argparse


class ScheduleArgumentParser(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(ScheduleArgumentParser, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs)
        self.default_input = kwargs['default']

    def __call__(self, parser, namespace, values, option_string=None):
        schedule = {}
        if len(values) == 0:
            schedule = self.default_input

        for kv in values:
            training_proportion, lr = kv.split(":")
            try:
                schedule[int(training_proportion)] = float(lr)
            except:
                raise Exception("Invalid Learning Rate Schedule provided. "
                                "It should be a set of <int>:<float> pairs."
                                "The first item is the step at which to update and the second is "
                                "the learning rate at that step.")

        setattr(namespace, self.dest, schedule)


class LearningRate:
    def __init__(self, opts, total_iterations):
        self.init_lr = opts["base_learning_rate"]
        if opts["lr_schedule_by_step"] is not None:
            self.lr_schedule_by_step = opts["lr_schedule_by_step"]
        else:
            raise Exception(
                "Not found `lr_schedule_by_step`. You must set to dictionary type which have `(step)<int>:(lr)<float>` pairs.")

    def feed_dict_lr(self, step, sess):
        diffs = {step-int(k): int(k)
                 for k in self.lr_schedule_by_step.keys() if int(k) <= step}
        nearest = str(diffs[min(diffs)])
        lr = self.lr_schedule_by_step[nearest]
        return lr


def add_arguments(parser):
    lr_group = parser.add_argument_group(
        'Customized Decay Learning Rate. Use with --lr-schedule custom.')
    lr_group.add_argument("--lr-schedule-by-step", action=ScheduleArgumentParser, nargs="*", default=None,
                          help="A schedule for learning rate warmup and decay, provided as space-separated "
                          "<int>:<float> pairs. The first item is the step at which to update and the second is "
                          "the learning rate at that step. \n"
                          "E.g.: --lr-schedule-by-step 0:0.00001 2500:0.0001 10000:0.0008 50000:0.00004 100000:0.00002")
    return parser


def set_defaults(opts):
    opts['summary_str'] += "Customized Step Learning Rate Schedule\n"
    return opts
