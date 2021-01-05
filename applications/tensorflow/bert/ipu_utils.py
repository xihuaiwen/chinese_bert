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

import os
import math
import sys
import re
import numpy as np

import tensorflow as tf
from tensorflow import pywrap_tensorflow
from tensorflow.python.ipu import utils
from collections import OrderedDict, defaultdict


def get_config(fp_exceptions=True,
               xla_recompute=False,
               availableMemoryProportion=None,
               disable_graph_outlining=False,
               num_required_ipus=1,
               esr=True,
               compile_only=False):

    # Builds ipu_options
    config = utils.create_ipu_config(
        merge_infeed_io_copies=True,
        max_cross_replica_sum_buffer_size=10*1024**5,
        always_rearrange_copies_on_the_host=False,
        disable_graph_convolution_caching=True,
        disable_graph_outlining=disable_graph_outlining,
        selection_order=utils.SelectionOrder.AUTO,
        scheduler_selection="Clustering")

    config = utils.auto_select_ipus(config, num_required_ipus)

    if availableMemoryProportion is not None:
        config = utils.set_matmul_options(config, {
            "availableMemoryProportion": str(availableMemoryProportion)
        })

    config = utils.set_recomputation_options(
        config, allow_recompute=xla_recompute)
    # simple way to skip the big `Transpose` operation due to bad allocation
    config = utils.set_matmul_options(config, clear_pass_type=True)
    config = utils.set_norm_options(config, use_stable_statistics=True)

    if compile_only:
        config = utils.set_ipu_connection_type(
            config, utils.DeviceConnectionType.NEVER, ipu_version=2, enable_remote_buffers=True)

    config = utils.set_floating_point_behaviour_options(config, inv=fp_exceptions, div0=fp_exceptions,
                                                        oflo=fp_exceptions, esr=esr, nanoo=fp_exceptions)
    return config


def get_var_list(func):
    """
        get variable names of func, exclude "self" if there is
    """
    func_code = func.__code__
    var_list = func_code.co_varnames[:func_code.co_argcount]
    var_list = [var for var in var_list if var != 'self']
    return var_list


def stage_wrapper(layer_list, needed_vars, stage_input_names):
    """a wrapper that generate stage function dynamically

    Args:
        layer_list: a list of model layer functions,
            layer's output must be a dictionary so that stage_function will know which param is needed by rest layers
        needed_values: list of string, values name that will be useful for rest stages
        stage_input_names: stage function need to output a list of tensors,
            so we need this additional list to keep the name for each tensor.
            stage_input_names will be updated at the end of each stage.
            stage_input_names will be in same order with needed_vars.

    Returns:
        a function that takes needed_vars concatenated and some key_word_args as it's inputs,
        sequentially call functions in "layer_list",
        and return a list of tensors that occur in "needed_vars" collected from each layer's output.
    """

    def stage_func(*args, **kwargs):
        """
        Args:
            args: can be from "inputs" of pipeline function or previous stage,
                if dataset is a list (instead of a dictionary), then it's values is also passed input args,
                that way,stage_input_names need to contain names for each value in dataset
            kwargs: can be from dataset, if dataset is a dictionary.
        """
        result = kwargs

        args = list(args)
        # args come from "inputs" argument of "pipeline" function
        result.update(zip(stage_input_names, args))

        for func in layer_list:
            var_list = get_var_list(func)
            outputs = func(**{name: result[name]
                              for name in var_list if name in result})
            # assume outputs to be a bectionary
            assert isinstance(outputs, dict)
            result.update(outputs)
        # only return needed vlaues
        result = OrderedDict([(key, result[key])
                              for key in needed_vars if key in result.keys()])
        # stage function can't return dictionary, so keep key in additional list
        # clear this list for use by next stage
        stage_input_names.clear()
        # now "stage_input_names" contains output name for current stage
        # and  at the same time, it's the input_name for next stage
        stage_input_names.extend(result.keys())
        return [result[key] for key in stage_input_names]
    return stage_func


def stages_constructor(stages_list, input_names, output_vars):
    """construct compuational_stages for pipeline

    Args:
        stages_list: list of list of layer functions.
            each list in stages_list represent a stage,
            function layers must output a dictionary so that this funciton can know name of it's output values
        input_names: appended inputs name list,
            if values are passed by "inputs" argument of pipeline function,
            this list will contain name of each value of it in the sequence of "inputs" value.
            if dataset is a list(instead of a dictionary), name for each value of it need to be
            appended after name of "inputs"
        output_vars: values output by last stage (used by optimizer)

    Returns:
        a list of stage functions
    """
    needed_vars = output_vars
    computational_stages = []
    # input names for args of a stage
    stage_input_names = list(input_names)
    # reverse the stage so that we start constructing from backward stages
    # that way, we can know which variables are needed by rest stages, and put it into "needed_vars"
    # in stage function, we will dynamically discard unsed variables to reduce transmission between stages
    for function_list in stages_list[::-1]:
        # for the last stage, output will be in same sequence with output_vars
        stage = stage_wrapper(function_list, needed_vars, stage_input_names)
        computational_stages.append(stage)
        needed_vars = set(needed_vars)
        for func in function_list:
            needed_vars = needed_vars.union(set(get_var_list(func)))
        # sort needed_varss so that it won't need to recompile because of output order changing
        needed_vars = sorted(needed_vars)
    # reverse computational_stages, because it's generated in reverse sequence
    return computational_stages[::-1]
