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
import re
import time
import argparse
import datetime
import random
from socket import gethostname
from collections import deque, OrderedDict, namedtuple, Counter
from functools import partial
from shutil import copytree
import numpy as np
import sys
import math
import importlib
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from tensorflow.python.training import optimizer
from tensorflow.python.ipu import pipelining_ops
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow import pywrap_tensorflow
from tensorboardX import SummaryWriter

from ipu_optimizer import get_optimizer
from ipu_utils import get_config, stages_constructor
import log as logger
import Datasets.data_loader as dataset
import modeling as bert_ipu
from multi_stage_wrapper import get_split_embedding_stages, get_split_matmul_stages, MultiStageEmbedding


GraphOps = namedtuple('graphOps',
                      ['graph', 'session', 'init', 'ops', 'placeholders',
                       'iterator', 'outfeed', 'saver', 'restore', 'tvars'])

pipeline_schedule_options = [str(p).split(".")[-1]
                             for p in list(ipu.ops.pipelining_ops.PipelineSchedule)]


def build_pretrain_pipeline_stages(model, bert_config, opts):
    """
    build pipeline stages according to "pipeline_stages" in config file
    """

    # flatten stages config into list of layers
    flattened_layers = []
    for stage in opts['pipeline_stages']:
        flattened_layers.extend(stage)
    layer_counter = Counter(flattened_layers)
    assert layer_counter['hid'] == opts['num_hidden_layers']
    assert layer_counter['emb'] == layer_counter['mlm']
    # pipeline_depth need to be times of stage_number*2
    # this is constrained by sdk
    assert opts['pipeline_depth'] % (len(opts['pipeline_stages'])*2) == 0

    computational_stages = []
    if layer_counter['emb'] > 1:
        # support distribute embedding to multiple IPUs
        embedding = MultiStageEmbedding(embedding_size=bert_config.hidden_size,
                                        vocab_size=bert_config.vocab_size,
                                        initializer_range=bert_config.initializer_range,
                                        n_stages=layer_counter['emb'],
                                        matmul_serialize_factor=opts["matmul_serialize_factor"],
                                        dtype=bert_config.dtype)
        embedding_stages = get_split_embedding_stages(
            embedding=embedding, split_count=layer_counter['emb'], bert_config=bert_config, batch_size=opts["batch_size"], seq_length=opts['seq_length'])
        # masked lm better be on same ipu with embedding layer for saving storage
        masked_lm_output_post_stages = get_split_matmul_stages(
            embedding=embedding, split_count=layer_counter['emb'], bert_config=bert_config)
    else:
        embedding_stages = [model.embedding_lookup_layer]
        masked_lm_output_post_stages = [model.mlm_head]

    layers = {
        'emb': embedding_stages,
        'pos': model.embedding_postprocessor_layer,
        'hid': model.encoder,
        'mlm': masked_lm_output_post_stages,
        'nsp': model.get_next_sentence_output_layer
    }
    stage_layer_list = []
    for stage in opts['pipeline_stages']:
        func_list = []
        for layer in stage:
            # embedding layer and mlm layer can be splited to mutliple IPUs, so need to be dealt with separately
            if layer == 'emb':
                func_list.append(embedding_stages[0])
                embedding_stages = embedding_stages[1:]
            elif layer == 'mlm':
                func_list.append(masked_lm_output_post_stages[0])
                masked_lm_output_post_stages = masked_lm_output_post_stages[1:]
            else:
                func_list.append(layers[layer])
        stage_layer_list.append(func_list)
    computational_stages = stages_constructor(
        stage_layer_list, ['learning_rate'], ['learning_rate', 'mlm_loss', 'nsp_loss'])

    return computational_stages


def build_network(infeed,
                  outfeed,
                  bert_config=None,
                  opts=None,
                  learning_rate=None,
                  is_training=True):

    # build model
    pipeline_model = bert_ipu.BertModel(bert_config, is_training=is_training)

    # build stages & device mapping
    computational_stages = build_pretrain_pipeline_stages(
        pipeline_model, bert_config, opts,)
    device_mapping = opts['device_mapping']

    logger.print_to_file_and_screen(
        f"************* computational stages: *************\n{computational_stages}", opts)
    logger.print_to_file_and_screen(
        f"************* device mapping: *************\n{device_mapping}", opts)

    # define optimizer
    def optimizer_function(learning_rate, mlm_loss, nsp_loss):
        total_loss = mlm_loss + nsp_loss
        optim = get_optimizer(learning_rate, opts)
        if opts["replicas"] > 1:
            optim = ipu.optimizers.cross_replica_optimizer.CrossReplicaOptimizer(optim)
        return ipu.ops.pipelining_ops.OptimizerFunctionOutput(optim, total_loss*opts["loss_scaling"])

    # define pipeline schedule
    pipeline_schedule = pipelining_ops.PipelineSchedule.Grouped
    if opts["pipeline_schedule"] == "Interleaved":
        pipeline_schedule = pipelining_ops.PipelineSchedule.Interleaved

    logger.print_to_file_and_screen(
        "Pipeline schdule use {}".format(opts["pipeline_schedule"]), opts)
    options = [ipu.pipelining_ops.PipelineStageOptions(
        matmul_options={
            "availableMemoryProportion": str(opts["available_memory_proportion"]),
            "partialsType": opts["half_partial"]
        },
        convolution_options={
            "partialsType": opts["half_partial"]}
    )] * len(device_mapping)
    # config pipeline
    if is_training:
        pipeline_ops = ipu.ops.pipelining_ops.pipeline(computational_stages=computational_stages,
                                                       pipeline_depth=int(
                                                           opts['pipeline_depth']),
                                                       repeat_count=opts['batches_per_step'],
                                                       inputs=[learning_rate],
                                                       infeed_queue=infeed,
                                                       outfeed_queue=outfeed,
                                                       optimizer_function=optimizer_function,
                                                       device_mapping=device_mapping,
                                                       forward_propagation_stages_poplar_options=options,
                                                       backward_propagation_stages_poplar_options=options,
                                                       offload_weight_update_variables=opts["variable_offloading"],
                                                       pipeline_schedule=pipeline_schedule,
                                                       name="Pipeline")
    else:
        pipeline_ops = ipu.ops.pipelining_ops.pipeline(computational_stages=computational_stages,
                                                       pipeline_depth=int(
                                                           opts['pipeline_depth']),
                                                       repeat_count=opts['batches_per_step'],
                                                       inputs=[learning_rate],
                                                       infeed_queue=infeed,
                                                       outfeed_queue=outfeed,
                                                       device_mapping=device_mapping,
                                                       forward_propagation_stages_poplar_options=options,
                                                       backward_propagation_stages_poplar_options=options,
                                                       offload_weight_update_variables=opts["variable_offloading"],
                                                       pipeline_schedule=pipeline_schedule,
                                                       name="Pipeline")

    return pipeline_ops


def build_graph(opts, is_training=True, feed_name=None):
    train_graph = tf.Graph()
    with train_graph.as_default():
        bert_config = bert_ipu.BertConfig.from_dict(opts)
        bert_config.dtype = tf.float32 if opts["precision"] == '32' else tf.float16

        # define placeholder
        placeholders = dict()
        placeholders['learning_rate'] = tf.placeholder(
            bert_config.dtype, shape=[])
        learning_rate = placeholders['learning_rate']

        # define input, datasets must be defined outside the ipu device scope.
        train_iterator = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset.load(opts, is_training=is_training),
                                                             feed_name=feed_name+"_in", replication_factor=opts['replicas'])
        # define output
        outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
            feed_name=feed_name+"_out", replication_factor=opts['replicas'])

        # building networks with pipeline
        def bert_net():
            return build_network(train_iterator,
                                 outfeed_queue,
                                 bert_config,
                                 opts,
                                 learning_rate,
                                 is_training)

        with ipu.scopes.ipu_scope('/device:IPU:0'):
            train = ipu.ipu_compiler.compile(bert_net, [])

        # get result from outfeed queue
        outfeed = outfeed_queue.dequeue()

        logger.print_trainable_variables(opts)

        restore = tf.train.Saver(var_list=tf.global_variables())
        savers = {
            "train_saver": tf.train.Saver(var_list=tf.global_variables(), max_to_keep=20),
            "best_saver": tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
        }

        ipu.utils.move_variable_initialization_to_cpu()
        train_init = tf.global_variables_initializer()
        tvars = tf.trainable_variables()

    # create session
    """calculate the number of required IPU"""
    num_ipus = (max(opts['device_mapping'])+1) * int(opts['replicas'])
    # The number of acquired IPUs must be the power of 2.
    if num_ipus & (num_ipus - 1) != 0:
        num_ipus = 2**int(math.ceil(math.log(num_ipus) / math.log(2)))
    ipu_options = get_config(fp_exceptions=opts["fp_exceptions"],
                             xla_recompute=opts["xla_recompute"],
                             availableMemoryProportion=opts["available_memory_proportion"],
                             disable_graph_outlining=opts["no_outlining"],
                             num_required_ipus=num_ipus,
                             esr=opts['stochastic_rounding'],
                             compile_only=opts['compile_only'])
    ipu.utils.configure_ipu_system(ipu_options)
    sess_cfg = tf.ConfigProto()
    sess_cfg.graph_options.rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.OFF
    train_sess = tf.Session(graph=train_graph, config=sess_cfg)

    return GraphOps(train_graph, train_sess, train_init, [train], placeholders, train_iterator, outfeed, savers, restore, tvars)


def training_step(train, learning_rate):
    _ = train.session.run(train.ops, feed_dict={
                          train.placeholders['learning_rate']: learning_rate})
    if not os.environ.get('TF_POPLAR_FLAGS') or '--use_synthetic_data' not in os.environ.get('TF_POPLAR_FLAGS'):
        _learning_rate, _mlm_loss, _nsp_loss = train.session.run(train.outfeed)
        mlm_loss = np.mean(_mlm_loss)
        nsp_loss = np.mean(_nsp_loss)
    else:
        mlm_loss, nsp_loss = 0, 0
    return mlm_loss, nsp_loss


def train(opts):
    # --------------- OPTIONS ---------------------
    total_samples = dataset.get_dataset_files_count(opts, is_training=True)
    opts["dataset_repeat"] = math.ceil(
        (opts["num_train_steps"]*opts["total_batch_size"])/total_samples)

    total_samples_per_epoch = total_samples/opts["duplicate_factor"]
    logger.print_to_file_and_screen(
        f"Total samples for each epoch {total_samples_per_epoch}", opts)
    steps_per_epoch = total_samples_per_epoch//opts["total_batch_size"]
    logger.print_to_file_and_screen(
        f"Total steps for each epoch {steps_per_epoch}", opts)

    steps_per_logs = math.ceil(
        opts["steps_per_logs"] / opts['batches_per_step'])*opts['batches_per_step']
    steps_per_tensorboard = math.ceil(
        opts["steps_per_tensorboard"] / opts['batches_per_step'])*opts['batches_per_step']
    steps_per_ckpts = math.ceil(
        opts["steps_per_ckpts"] / opts['batches_per_step'])*opts['batches_per_step']
    logger.print_to_file_and_screen(
        f"Checkpoint will be saved every {steps_per_ckpts} steps.", opts)

    total_steps = (opts["num_train_steps"] //
                   opts['batches_per_step'])*opts['batches_per_step']
    logger.print_to_file_and_screen(
        f"{opts['batches_per_step']} steps will be run for ipu to host synchronization once, it should be divided by num_train_steps, so num_train_steps will limit to {total_steps}.", opts)
    # learning rate stratege
    LR = lr_schedule.LearningRate(opts, total_steps)

    # -------------- BUILD TRAINING GRAPH ----------------
    train = build_graph(opts,
                        is_training=True, feed_name="trainfeed")
    train.session.run(train.init)
    train.session.run(train.iterator.initializer)

    step = 0
    # -------------- SAVE AND RESTORE --------------
    if opts["restore_dir"]:
        latest_checkpoint = tf.train.latest_checkpoint(opts['restore_dir'])
        logger.print_to_file_and_screen(
            f"Restoring training from latest checkpoint: {latest_checkpoint}", opts)
        step_pattern = re.compile(".*ckpt-([0-9]+)$")
        step = int(step_pattern.match(latest_checkpoint).groups()[0])
        train.restore.restore(train.session, latest_checkpoint)
        epoch = step / steps_per_epoch

        # restore event files
        source_path = opts["restore_dir"]+'/event'
        target_path = opts["save_path"]+'/event'
        copytree(source_path, target_path)
    else:
        if opts["init_checkpoint"]:
            train.restore.restore(train.session, opts["init_checkpoint"])
            logger.print_to_file_and_screen("Init Model from checkpoint {}".format(
                opts["init_checkpoint"]), opts)

    if opts['save_path']:
        file_path = train.restore.save(
            train.session, opts["save_path"]+"/ckpt", global_step=0)
        logger.print_to_file_and_screen(
            f"Saved checkpoint to {file_path}", opts)

    # tensorboard file path
    log_path = opts["save_path"]+'/event'
    logger.print_to_file_and_screen(
        f"Tensorboard event file path {log_path}", opts)
    writer = SummaryWriter(log_path)
    train_saver = train.saver["train_saver"]
    best_saver = train.saver["best_saver"]
    best_total_loss = 1.8
    best_step = 0
    # best_ckpt_path = os.path.join(opts["save_path"], "best_ckpt")
    # if not os.path.exists(best_ckpt_path):
    #     os.makedirs(best_ckpt_path)

    # ------------- TRAINING LOOP ----------------
    print_format = ("step: {step:6d}, epoch: {epoch:6.2f}, lr: {lr:6.6f}, mlm_loss: {mlm_loss:6.3f}, nsp_loss: {nsp_loss:6.3f},\
        samples/sec: {samples_per_sec:6.2f}, time: {iter_time:8.6f}, total_time: {total_time:8.1f}")

    learning_rate = mlm_loss = nsp_loss = 0
    batch_time = 1e+5
    start_all = time.time()

    try:
        while step < total_steps:
            start = time.time()
            learning_rate = LR.feed_dict_lr(step, train.session)
            try:
                mlm_loss, nsp_loss = training_step(
                    train, learning_rate)
            except tf.errors.OpError as e:
                raise tf.errors.ResourceExhaustedError(
                    e.node_def, e.op, e.message)

            batch_time /= opts['batches_per_step']

            is_log_step = (step % steps_per_logs == 0)
            is_save_tensorboard_step = (steps_per_tensorboard != 0 and (
                step % steps_per_tensorboard == 0))
            is_save_ckpt_step = (step and (
                step % steps_per_ckpts == 0 or step == total_steps - opts['batches_per_step']))

            if step == opts['batches_per_step']:
                poplar_compile_time = time.time() - start_all
                logger.print_to_file_and_screen(
                    f"Poplar compile time: {poplar_compile_time:.2f}s", opts)

            if is_log_step:
                total_time = time.time() - start_all
                epoch = step / steps_per_epoch
                stats = OrderedDict([
                    ('step', step),
                    ('epoch', epoch),
                    ('lr', learning_rate),
                    ('mlm_loss', mlm_loss),
                    ('nsp_loss', nsp_loss),
                    ('iter_time', batch_time),
                    ('samples_per_sec',
                        opts['total_batch_size']/batch_time),
                    ('total_time', total_time),
                ])

                logger.print_to_file_and_screen(
                    print_format.format(**stats), opts)

            samples = step*int(opts["total_batch_size"])
            writer.add_scalar("loss/MLM", mlm_loss, samples)
            writer.add_scalar("loss/NSP", nsp_loss, samples)
            writer.add_scalar("defaultLearningRate", learning_rate, samples)

            if is_save_tensorboard_step:
                save_model_tensorboard_infomation(file_path, writer, step)
            if is_save_ckpt_step:
                file_path = train_saver.save(
                    train.session, opts["save_path"] + '/ckpt', global_step=step)
                logger.print_to_file_and_screen(
                    f"Saved checkpoint to {file_path}", opts)
            
            if best_total_loss > mlm_loss + nsp_loss and step - best_step > 5:
                best_total_loss = mlm_loss + nsp_loss
                best_step = step
                filepath = best_saver.save(train.session, save_path=opts["save_path"]+'/best', global_step=step)
                logger.print_to_file_and_screen(
                    f"Saved checkpoint to {filepath}", opts)
            if mlm_loss>5 and step>30000:
                logger.print_to_file_and_screen(
                    f"Loss diverge at step={step}, mlm_loss={mlm_loss}", opts)
                break

            step += opts['batches_per_step']
            batch_time = (time.time() - start)
    finally:
        train.session.close()


def str_to_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise argparse.ArgumentTypeError(f'{value} is not a valid boolean value')


def add_arguments(parser, required=True):
    group = parser.add_argument_group('Main')

    '''***************************** training *****************************'''
    group.add_argument('--help', action='store_true',
                       help='Show help information')
    group.add_argument('--task', type=str,
                       help="which kind of NLP task: pretraining")
    group.add_argument('--config', type=str, default='configs/bert_tiny.json',
                       help='json file for BERT Base/Large config file.')
    group.add_argument('--restore-dir', type=str, default="")
    group.add_argument('--batch-size', type=int,
                       help="Set batch-size for training graph")
    group.add_argument('--base-learning-rate', type=float, default=2e-5,
                       help="Base learning rate exponent (2**N). blr = lr /  bs")
    group.add_argument('--epochs', type=float, default=5,
                       help="Number of training epochs")
    group.add_argument('--loss-scaling', type=float, default=1,
                       help="Loss scaling factor")
    group.add_argument('--steps-per-ckpts', type=int, default=256,
                       help="Steps per checkpoints")
    group.add_argument('--optimizer', type=str, default="momentum",
                       choices=['sgd', 'momentum', 'adam', 'adamw', 'lamb'],
                       help="Optimizer")
    group.add_argument('--momentum', type=float, default=0.984375,
                       help="Momentum coefficient.")
    group.add_argument('--beta1', type=float, default=0.9,
                       help="lamb/adam beta1 coefficient.")
    group.add_argument('--beta2', type=float, default=0.999,
                       help="lamb/adam beta2 coefficient.")
    group.add_argument('--epsilon', type=float,
                       default=1e-4, help="lamb/adam epsilon.")
    group.add_argument('--lr-schedule', default='exponential',
                       choices=["exponential", "custom",
                                "natural_exponential", "polynomial"],
                       help="Learning rate schedule function. Default: exponential")
    group.add_argument('--seed', default=None,
                       help="Seed for randomizing training")
    group.add_argument('--steps-per-logs', type=int, default=1,
                       help="Logs per epoch (if number of epochs specified)")

    '''***************************** model *****************************'''
    group.add_argument('--use-attention-projection-bias', type=str_to_bool, default=True,
                       help="Whether to use bias in linear projection behind attention layer. Default: True.")
    group.add_argument('--use-cls-layer', type=str_to_bool, default=True,
                       help="""Include the CLS layer in pretraining.
                       This layer comes after the encoders but before the projection for the MLM loss. Default: True.""")
    group.add_argument('--use-qkv-bias', type=str_to_bool, default=True,
                       help="""Whether to use bias in QKV calculation of attention layer. Default: True.""")

    '''***************************** ipu *****************************'''
    recomputation_mode_available = [
                    p.name for p in ipu.ops.pipelining_ops.RecomputationMode
                        ]
    group.add_argument('--recomputation-mode', type=str, default="RecomputeAndBackpropagateInterleaved",
                                   choices=recomputation_mode_available)
    group.add_argument('--pipeline-depth', type=int, default=1,
                       help="Depth of pipeline to use. Must also set --shards > 1.")
    group.add_argument('--pipeline-schedule', type=str, default="Interleaved",
                       choices=pipeline_schedule_options, help="Pipelining scheduler. Choose between 'Interleaved' and 'Grouped'")
    group.add_argument('--replicas', type=int, default=1,
                       help="Replicate graph over N workers to increase batch to batch-size*N")
    group.add_argument('--matmul-serialize-factor', type=int, default=4,
                       help="The factor to serialize matmul. This can decrease memory footprint.")
    group.add_argument('--precision', type=str, default="16", choices=["16", "32"],
                       help="Precision of Ops(weights/activations/gradients) data types: 16, 32.")
    group.add_argument('--batches-per-step', type=int, default=1,
                       help="Maximum number of batches to perform on the device before returning to the host.")
    group.add_argument('--available-memory-proportion', default=0.23, nargs='+',
                       help="Proportion of memory which is available for convolutions. Use a value of less than 0.6 "
                            "to reduce memory usage.")
    group.add_argument('--variable-offloading', type=str_to_bool, default=True,
                       help="Enable ofﬂoad variables into remote memory.")
    group.add_argument('--stochastic-rounding', type=str_to_bool, default=True,
                       help="Enable stochastic rounding. Set to False when run evaluation.")
    group.add_argument('--no-outlining', type=str_to_bool, nargs="?", const=True, default=False,
                       help="Disable TF outlining optimisations. This will increase memory for a small throughput improvement.")
    group.add_argument('--half-partial', type=str, default="float", choices=["half", "float"],
                       help="Mamul&Conv precision data type. Choose between 'half' and 'float'")
    group.add_argument('--compile-only', action="store_true", default=False,
                       help="Configure Poplar to only compile the graph. This will not acquire any IPUs and thus facilitate profiling without using hardware resources.")

    '''***************************** dataset *****************************'''
    group.add_argument('--train-file', type=str, required=False,
                       help="path to wiki/corpus training dataset tfrecord file.")
    group.add_argument("--seq-length", type=int, default=128,
                       help="the max sequence length, default 128.")
    group.add_argument("--max-predictions-per-seq", type=int, default=20,
                       help="the number of masked words per sentence, default 20.")
    group.add_argument('--parallell-io-threads', type=int, default=4,
                       help="Number of cpu threads used to do data prefetch.")
    group.add_argument('--synthetic-data', type=str_to_bool, default=False,
                       help="Use synthetic data.")
    group.add_argument('--dataset-repeat', type=int, default=1,
                       help="Number of times dataset to repeat.")

    return parser


def set_training_defaults(opts):
    opts['name'] = 'BERT_' + opts['task']
    opts['total_batch_size'] = opts['batch_size'] * \
        opts['pipeline_depth']*opts['replicas']
    opts['summary_str'] = "Training\n"
    opts['summary_str'] += " Batch Size: {total_batch_size}\n"
    if opts['pipeline_depth'] > 1:
        opts['summary_str'] += "  Pipeline depth {pipeline_depth} \n"
    opts['summary_str'] += (" Base Learning Rate: {base_learning_rate}\n"
                            " Loss Scaling: {loss_scaling}\n")

    if opts['optimizer'].lower() == 'sgd':
        opts['summary_str'] += "SGD\n"
    elif opts['optimizer'].lower() == 'momentum':
        opts['name'] += '_Mom'
        opts['summary_str'] += ("SGD with Momentum\n"
                                " Momentum Coefficient: {momentum}\n")
    elif opts['optimizer'].lower() == 'adam':
        opts['name'] += '_Adam'
        opts['summary_str'] += ("Adam\n"
                                " beta1: {beta1}, beta2: {beta2}, epsilon: {epsilon}\n")
    elif opts['optimizer'].lower() == 'adamw':
        opts['name'] += '_AdamW'
        opts['summary_str'] += ("Adam With Weight decay\n"
                                " beta1: {beta1}, beta2: {beta2}, epsilon: {epsilon}\n")
    elif opts['optimizer'].lower() == 'lamb':
        opts['name'] += '_LAMB'
        opts['summary_str'] += ("LAMB\n"
                                " beta1: {beta1}, beta2: {beta2}, epsilon: {epsilon}\n")


def set_ipu_defaults(opts):
    opts['summary_str'] += "Using Infeeds\n Max Batches Per Step: {batches_per_step}\n"
    opts['summary_str'] += 'Device\n'
    opts['summary_str'] += ' Precision: {}\n'.format(opts['precision'])
    opts['summary_str'] += ' IPU\n'
    opts['poplar_version'] = os.popen('popc --version').read()
    opts['summary_str'] += ' {poplar_version}'

    opts['hostname'] = gethostname()
    opts['datetime'] = str(datetime.datetime.now())

    if opts['seed']:
        seed = int(opts['seed'])
        random.seed(seed)
        # tensorflow seed
        tf.set_random_seed(random.randint(0, 2**32 - 1))
        # numpy seed
        np.random.seed(random.randint(0, 2**32 - 1))
        # ipu seed
        ipu.utils.reset_ipu_seed(random.randint(-2**16, 2**16 - 1))
    opts['summary_str'] += (' {hostname}\n'
                            ' {datetime}\n')


def set_defaults(opts, lr_schedule):
    dataset.set_defaults(opts)
    set_training_defaults(opts)
    lr_schedule.set_defaults(opts)
    set_ipu_defaults(opts)
    logger.set_defaults(opts)


def save_model_tensorboard_infomation(checkpoint_path, writer=None, i=0):
    initializers = {}
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_map = reader.get_variable_to_shape_map()
    for key, _ in var_to_map.items():
        if key == 'global_step':
            continue
        initializers[key] = reader.get_tensor(key)
    for name, np_weight in initializers.items():
        if "Momentum" in name:
            continue
        name = name.replace(":", "_")
        writer.add_histogram(name, np_weight.astype(np.float32), i)
        writer.add_scalar(
            f"L2/{name}", np.linalg.norm(np_weight.astype(np.float32)), i)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser(
        description='BERT  Pretraining in TensorFlow', add_help=False)
    parser = add_arguments(parser)

    opts = vars(parser.parse_args())

    if opts['help']:
        parser.print_help()
    else:
        opts.update(bert_ipu.BertConfig.from_json_file(opts['config']))
        try:
            lr_schedule = importlib.import_module(
                "LR_Schedules." + opts['lr_schedule'])
        except ImportError:
            raise ValueError(
                "LR_Schedules/{}.py not found".format(opts['lr_schedule']))
        set_defaults(opts, lr_schedule)
        opts["command"] = ' '.join(sys.argv)
        logger.print_to_file_and_screen(
            "Command line: " + opts["command"], opts)
        logger.print_to_file_and_screen(
            opts["summary_str"].format(**opts), opts)
        opts["summary_str"] = ""
        logger.print_to_file_and_screen(opts, opts)
        train(opts)
