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
import json
import os
import re
import six
import time
import argparse
import datetime
import random
import collections
from socket import gethostname
from collections import deque, OrderedDict, namedtuple, defaultdict
from functools import partial
import numpy as np
import sys
import math
import importlib
import log as logger
from tensorflow.python import ipu
from ipu_utils import get_config, stages_constructor
from tensorflow.python.ipu.autoshard import automatic_sharding
from tensorflow.python.ipu import loops, ipu_infeed_queue, ipu_outfeed_queue, ipu_compiler, utils, scopes
from tensorflow.python.ipu.optimizers import sharded_optimizer
from tensorflow.python.training import gradient_descent
from tensorflow.python.ipu.utils import reset_ipu_seed
from tensorflow.python.ipu.ops import pipelining_ops
from ipu_optimizer import StageMomentumOptimizer
from tensorflow.python.ipu.scopes import ipu_scope
import Datasets.data_loader as dataset
import modeling as bert_ipu
from tensorboardX import SummaryWriter
from ipu_optimizer import get_optimizer

GraphOps = namedtuple(
    'graphOps', ['graph',
                 'session',
                 'init',
                 'ops',
                 'placeholders',
                 'iterator',
                 'outfeed',
                 'saver',
                 'restore',
                 'tvars'])
RawResult = namedtuple("RawResult",
                       ["unique_id", "start_logits", "end_logits"])

pipeline_schedule_options = [str(p).split(".")[-1]
                             for p in list(pipelining_ops.PipelineSchedule)]


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = dataset.tokenization.BasicTokenizer(
        do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if opts["verbose_logging"]:
            tf.logging.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if opts["verbose_logging"]:
            tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if opts["verbose_logging"]:
            tf.logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if opts["verbose_logging"]:
            tf.logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file):
    """Write final predictions to the json file and log-odds of null if needed."""
    tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
    tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = namedtuple(
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = OrderedDict()
    all_nbest_json = OrderedDict()
    scores_diff_json = OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # If we could have irrelevant answers, get the min score of irrelevant.
            if opts["version_2_with_negative"]:
                feature_null_score = result.start_logits[0] + \
                    result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        if opts["version_2_with_negative"]:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = namedtuple(
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(
                    pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(
                    orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # If we didn't inlude the empty option in the n-best, inlcude it.
        if opts["version_2_with_negative"]:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="", start_logit=null_start_logit,
                        end_logit=null_end_logit))
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not opts["version_2_with_negative"]:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > opts["null_score_diff_threshold"]:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with tf.gfile.GFile(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with tf.gfile.GFile(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if opts["version_2_with_negative"]:
        with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def build_squad_pipeline_stages(model, bert_config, opts, is_training):
    """
    build pipeline stages according to "pipeline_stages" in config file
    """

    # flatten stages config into list of layers
    flattened_layers = []
    for stage in opts['pipeline_stages']:
        flattened_layers.extend(stage)
    layer_counter = collections.Counter(flattened_layers)
    assert layer_counter['hid'] == opts['num_hidden_layers']
    assert layer_counter['emb'] == 1
    # pipeline_depth need to be times of stage_number*2
    # this is constrained by sdk
    assert opts['pipeline_depth'] % (len(opts['pipeline_stages'])*2) == 0

    layers = {
        'emb': model.embedding_lookup_layer,
        'pos': model.embedding_postprocessor_layer,
        'hid': model.encoder,
        'loc': model.get_loc_logic_output_layer
    }
    stage_layer_list = []
    for stage in opts['pipeline_stages']:
        func_list = []
        for layer in stage:
            # embedding layer and mlm layer can be splited to mutliple IPUs, so need to be dealt with separately
            func_list.append(layers[layer])
        stage_layer_list.append(func_list)
    computational_stages = stages_constructor(
        stage_layer_list, ['learning_rate'], ['learning_rate', 'total_loss'])

    return computational_stages


def build_network(infeed,
                  outfeed,
                  iterations_per_step=1,
                  bert_config=None,
                  opts=None,
                  learning_rate=None,
                  is_training=True):
    # build model
    pipeline_model = bert_ipu.BertModel(bert_config,
                                        is_training=is_training)

    # build stages & device mapping
    computational_stages = build_squad_pipeline_stages(
        pipeline_model, bert_config, opts, is_training)
    device_mapping = opts['device_mapping']
    logger.print_to_file_and_screen(
        f"************* computational stages: *************\n{computational_stages}", opts)
    logger.print_to_file_and_screen(
        f"************* device mapping: *************\n{device_mapping}", opts)

    # define optimizer
    def optimizer_function(learning_rate, total_loss):
        optimizer = get_optimizer(learning_rate, opts)
        if opts["replicas"] > 1:
            optimizer = ipu.optimizers.cross_replica_optimizer.CrossReplicaOptimizer(
                optimizer)
        return pipelining_ops.OptimizerFunctionOutput(optimizer, total_loss)

    options = [ipu.pipelining_ops.PipelineStageOptions(
        matmul_options={"availableMemoryProportion": str(
            opts["available_memory_proportion"]), "partialsType": opts["half_partial"]},
        convolution_options={"partialsType": opts["half_partial"]})] * len(computational_stages)

    if is_training:
        return pipelining_ops.pipeline(computational_stages=computational_stages,
                                       pipeline_depth=int(
                                           opts['pipeline_depth']),
                                       repeat_count=iterations_per_step,
                                       inputs=[learning_rate],
                                       infeed_queue=infeed,
                                       outfeed_queue=outfeed,
                                       device_mapping=device_mapping,
                                       forward_propagation_stages_poplar_options=options,
                                       backward_propagation_stages_poplar_options=options,
                                       offload_weight_update_variables=opts['variable_offloading'],
                                       optimizer_function=optimizer_function,
                                       name="Pipeline")
    else:
        return pipelining_ops.pipeline(computational_stages=computational_stages,
                                       pipeline_depth=int(
                                           opts['pipeline_depth']),
                                       repeat_count=iterations_per_step,
                                       inputs=[learning_rate],
                                       infeed_queue=infeed,
                                       outfeed_queue=outfeed,
                                       device_mapping=device_mapping,
                                       forward_propagation_stages_poplar_options=options,
                                       backward_propagation_stages_poplar_options=options,
                                       offload_weight_update_variables=opts['variable_offloading'],
                                       name="Pipeline")


def build_graph(opts, iterations_per_step=1, is_training=True, feed_name=None):

    train_graph = tf.Graph()
    with train_graph.as_default():
        bert_config = bert_ipu.BertConfig.from_dict(opts)
        bert_config.dtype = tf.float32 if opts["precision"] == '32' else tf.float16
        placeholders = dict()
        placeholders['learning_rate'] = tf.placeholder(
            bert_config.dtype, shape=[])
        learning_rate = placeholders['learning_rate']

        train_iterator = ipu_infeed_queue.IPUInfeedQueue(dataset.load(opts, is_training=is_training),
                                                         feed_name=feed_name + "_in",
                                                         replication_factor=opts['replicas'])
        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name=feed_name + "_out",
                                                          replication_factor=opts['replicas'])

        # building networks with pipeline
        def bert_net():
            return build_network(train_iterator,
                                 outfeed_queue,
                                 iterations_per_step,
                                 bert_config,
                                 opts,
                                 learning_rate,
                                 is_training)

        with ipu_scope('/device:IPU:0'):
            train = ipu.ipu_compiler.compile(bert_net, [])

        outfeed = outfeed_queue.dequeue()

        logger.print_trainable_variables(opts)

        restore = tf.train.Saver(var_list=tf.global_variables())
        train_saver = tf.train.Saver(max_to_keep=5)

        ipu.utils.move_variable_initialization_to_cpu()
        train_init = tf.global_variables_initializer()
        tvars = tf.trainable_variables()

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
                             esr=True if is_training else False)
    ipu.utils.configure_ipu_system(ipu_options)
    train_sess = tf.Session(graph=train_graph, config=tf.ConfigProto())
    return GraphOps(train_graph, train_sess, train_init, [train], placeholders, train_iterator, outfeed,
                    train_saver, restore, tvars)


def training_step(train, learning_rate):
    start = time.time()
    _ = train.session.run(train.ops, feed_dict={
                          train.placeholders['learning_rate']: learning_rate})
    batch_time = (time.time() - start)
    if not os.environ.get('TF_POPLAR_FLAGS') or '--use_synthetic_data' not in os.environ.get(
            'TF_POPLAR_FLAGS'):
        _, _loss = train.session.run(train.outfeed)
        loss = np.mean(_loss)
    else:
        loss = 0
    return loss, batch_time


def predict_step(predict, learning_rate):
    _ = predict.session.run(predict.ops,
                            feed_dict={predict.placeholders['learning_rate']: learning_rate})
    _unique_ids, _start_logits, _end_logits = predict.session.run(
        predict.outfeed)
    return _unique_ids, _start_logits, _end_logits


def train(opts):
    consume_time = None
    # --------------- OPTIONS ---------------------
    tf.gfile.MakeDirs(opts["output_dir"])
    epochs = opts["epochs"]
    train_examples = dataset.read_squad_examples(
        opts['train_file'], opts, is_training=True)
    total_samples = len(train_examples)
    logger.print_to_file_and_screen(
        f"Total samples {total_samples}", opts)
    iterations_per_epoch = total_samples // opts["total_batch_size"]

    log_iterations = opts['batches_per_step'] * opts["steps_per_logs"]
    ckpt_iterations = opts['batches_per_step'] * opts["steps_per_ckpts"]
    # total iterations
    iterations = epochs * iterations_per_epoch
    # So many iterations will be run for one step.
    iterations_per_step = opts['batches_per_step']
    # Avoid nan issue caused by queue length is zero.
    queue_len = iterations_per_epoch // iterations_per_step
    if queue_len == 0:
        queue_len = 1
    batch_times = deque(maxlen=queue_len)
    # learning rate stratege
    LR = lr_schedule.LearningRate(opts, iterations)
    # -------------- BUILD LOG PATH ----------------
    log_time = datetime.datetime.now().strftime('%Y-%m-%d-%T')
    writer = SummaryWriter('./log/' + log_time)
    if opts['do_training']:
        # -------------- BUILD TRAINING GRAPH ----------------
        train = build_graph(opts, iterations_per_step,
                            is_training=True, feed_name="trainfeed")
        train.session.run(train.init)
        train.session.run(train.iterator.initializer)

        # -------------- SAVE AND RESTORE --------------
        if opts["init_checkpoint"]:
            (assignment_map, initialized_variable_names
             ) = bert_ipu.get_assignment_map_from_checkpoint(train.tvars, opts["init_checkpoint"])

            reader = tf.train.NewCheckpointReader(opts["init_checkpoint"])
            load_vars = reader.get_variable_to_shape_map()

            saver_restore = tf.train.Saver(assignment_map)
            saver_restore.restore(train.session, opts["init_checkpoint"])
        # assignment_map lenth = embbeding(5 tensors) + encorder layers(each layer 10 tensors)
        assert len(assignment_map) == 5 + opts['num_hidden_layers']*10

        if opts['steps_per_ckpts']:
            filepath = train.saver.save(
                train.session, opts["save_path"] + '/ckpt', global_step=0)
            logger.print_to_file_and_screen(
                f"Saved checkpoint to {filepath}", opts)

        if opts.get('restoring'):
            filename_pattern = re.compile(".*ckpt-[0-9]+$")
            ckpt_pattern = re.compile(".*ckpt-([0-9]+)$")
            filenames = sorted([os.path.join(opts['restore_ckpt'], f[:-len(".index")])
                                for f in os.listdir(opts['restore_ckpt']) if
                                filename_pattern.match(f[:-len(".index")]) and f[-len(
                                    ".index"):] == ".index"],
                               key=lambda x: int(ckpt_pattern.match(x).groups()[0]))
            latest_checkpoint = filenames[-1]
            logger.print_to_file_and_screen(
                f"Restoring training from latest checkpoint: {latest_checkpoint}", opts)
            ckpt_pattern = re.compile(".*ckpt-([0-9]+)$")
            i = int(ckpt_pattern.match(latest_checkpoint).groups()[0]) + 1
            train.saver.restore(train.session, latest_checkpoint)
            epoch = float(opts["total_batch_size"] *
                          (i + iterations_per_step)) / total_samples
        else:
            i = 0

        start_time = datetime.datetime.now()
        # ------------- TRAINING LOOP ----------------
        print_format = (
            "step: {step:6d}, iteration: {iteration:6d}, epoch: {epoch:6.2f}, lr: {lr:6.4g}, loss: {train_loss_avg:6.3f}, "
            " samples/sec: {samples_per_sec:6.2f}, time: {it_time:8.6f}, total_time: {total_time:8.1f}")
        all_results = []
        step = 0
        start_all = time.time()
        while i < iterations:

            step += 1
            epoch = float(opts["total_batch_size"] * i) / total_samples

            if opts['lr_schedule'] == 'custom':
                iteration_standard = i
            elif opts['lr_schedule'] == 'schedule_by_epoch':
                iteration_standard = epoch
            elif opts['lr_schedule'] == 'natural_exponential':
                iteration_standard = step
            else:
                iteration_standard = i
            learning_rate = LR.feed_dict_lr(iteration_standard, train.session)

            try:
                loss, batch_time = training_step(
                    train, learning_rate)
            except tf.errors.OpError as e:
                raise tf.errors.ResourceExhaustedError(
                    e.node_def, e.op, e.message)

            batch_time /= iterations_per_step

            if i != 0:
                batch_times.append([batch_time])

            if i % log_iterations == 0:

                if len(batch_times) != 0:
                    avg_batch_time = np.mean(batch_times)
                else:
                    avg_batch_time = batch_time

                # flush times every time it is reported
                batch_times.clear()

                total_time = time.time() - start_all

                stats = OrderedDict([
                    ('step', step),
                    ('iteration', i + iterations_per_step),
                    ('epoch', epoch),
                    ('lr', learning_rate),
                    ('train_loss_avg', loss),
                    ('it_time', avg_batch_time),
                    ('samples_per_sec',
                     opts['total_batch_size'] / avg_batch_time),
                    ('total_time', total_time),
                ])

                logger.print_to_file_and_screen(
                    print_format.format(**stats), opts)

            writer.add_scalar("Loss", loss, step)
            writer.add_scalar("LearningRate", learning_rate, step)

            if i % ckpt_iterations == 0:
                filepath = train.saver.save(train.session, opts["save_path"] + '/ckpt',
                                            global_step=i + iterations_per_step)
                logger.print_to_file_and_screen(
                    f"Saved checkpoint to {filepath}", opts)

            i += iterations_per_step
        train.session.close()
        end_time = datetime.datetime.now()
        consume_time = (end_time - start_time).seconds
    if opts["do_predict"]:
        if opts['optimizer'] == 'adam':
            # all stages 14 = num_hidden_layers/1 + embbeding + loss (1 layer each stage avoid OOM)
            opts["pipeline_depth"] = 14
        else:
            # all stages 8 = num_hidden_layers/2 + embbeding + loss (2 layers each stage)
            opts["pipeline_depth"] = 8

        i = 0
        eval_examples = dataset.read_squad_examples(
            opts["predict_file"], opts, is_training=False)

        eval_writer = dataset.FeatureWriter(
            filename=os.path.join(opts["tfrecord_dir"], "eval.tf_record"),
            is_training=False)
        eval_features = []

        tokenizer = dataset.tokenization.FullTokenizer(
            vocab_file=opts['vocab_file'], do_lower_case=opts['do_lower_case'])

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        dataset.convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=opts["seq_length"],
            doc_stride=opts["doc_stride"],
            max_query_length=opts["max_query_length"],
            is_training=False,
            output_fn=append_feature)

        eval_writer.close()
        # -------------- BUILD TRAINING GRAPH ----------------
        predict = build_graph(
            opts, iterations_per_step, is_training=False, feed_name="evalfeed")
        predict.session.run(predict.init)
        predict.session.run(predict.iterator.initializer)

        if opts['do_training']:
            # --------------- RESTORE ----------------
            filename_pattern = re.compile(".*ckpt-[0-9]+$")
            ckpt_pattern = re.compile(".*ckpt-([0-9]+)$")
            filenames = sorted([os.path.join(opts["logs_path"], f[:-len(".index")])
                                for f in os.listdir(opts["logs_path"]) if
                                filename_pattern.match(f[:-len(".index")]) and f[-len(
                                    ".index"):] == ".index"],
                               key=lambda x: int(ckpt_pattern.match(x).groups()[0]))
            latest_checkpoint = filenames[-1]
            logger.print_to_file_and_screen(
                f"Init from latest checkpoint: {latest_checkpoint}", opts)
            predict.saver.restore(predict.session, latest_checkpoint)
        else:
            if opts["init_checkpoint"]:
                (assignment_map, initialized_variable_names
                 ) = bert_ipu.get_assignment_map_from_checkpoint(predict.tvars, opts["init_checkpoint"])
                saver_restore = tf.train.Saver(assignment_map)
                saver_restore.restore(predict.session, opts["init_checkpoint"])
                assert len(assignment_map) >= 127
        all_results = []
        iterations = len(
            eval_features) // (opts['batch_size'] * opts['pipeline_depth'])
        while i < iterations:
            learning_rate = LR.feed_dict_lr(i, predict.session)
            try:
                unique_ids, start_logits, end_logits = predict_step(
                    predict, learning_rate)
            except tf.errors.OpError as e:
                raise tf.errors.ResourceExhaustedError(
                    e.node_def, e.op, e.message)

            i += iterations_per_step

            if len(all_results) % 1 == 0:
                logger.print_to_file_and_screen(
                    ("Processing example: %d" % (len(all_results))), opts)

            for j in range(len(unique_ids)):
                unique_id = int(unique_ids[j])
                start_logit = [float(x) for x in start_logits[j].flat]
                end_logit = [float(x) for x in end_logits[j].flat]
                all_results.append(
                    RawResult(
                        unique_id=unique_id,
                        start_logits=start_logit,
                        end_logits=end_logit))
        output_prediction_file = os.path.join(
            opts["tfrecord_dir"], "predictions.json")
        output_nbest_file = os.path.join(
            opts["tfrecord_dir"], "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(
            opts["tfrecord_dir"], "null_odds.json")
        eval_features = eval_features[:len(all_results)]
        write_predictions(eval_examples, eval_features, all_results,
                          opts["n_best_size"], opts["max_answer_length"],
                          opts["do_lower_case"], output_prediction_file,
                          output_nbest_file, output_null_log_odds_file)

        # --------------- CLEANUP ----------------
        predict.session.close()
    logger.print_to_file_and_screen(
        f"training times： {consume_time} s", opts)


def add_arguments(parser, required=True):
    group = parser.add_argument_group('Main')

    '''*****************************training*****************************'''
    group.add_argument('--help', action='store_true',
                       help='Show help information')
    group.add_argument('--task', type=str,
                       help="which kind of NLP task: pretraining/squad.")
    group.add_argument('--do-predict', action="store_true",
                       help="Whether to do valiation.")
    group.add_argument('--do-training', action="store_true",
                       help="Whether do training.")
    group.add_argument('--config', required=True, type=str,
                       help='json file for BERT Base/Large config file.')
    group.add_argument('--lr-schedule', default='stepped',
                       help="Learning rate schedule function. Default: stepped")
    group.add_argument('--vocab-file', type=str,
                       help="The vocabulary file that the BERT model was trained on.")
    group.add_argument('--init-checkpoint', type=str,
                       help="Initial checkpoint (usually from a pre-trained BERT model).")
    group.add_argument('--batch-size', type=int,
                       help="Set batch-size for training graph")
    group.add_argument('--base-learning-rate', type=float, default=5e-5,
                       help="Base learning rate exponent (2**N). blr = lr /  bs")
    group.add_argument('--epochs', type=float, default=1,
                       help="Number of training epochs")
    group.add_argument('--loss-scaling', type=float, default=128,
                       help="Loss scaling factor")
    group.add_argument('--ckpts-per-epoch', type=int, default=10,
                       help="Checkpoints per epoch")
    group.add_argument('--optimizer', type=str, default="SGD", choices=['SGD', 'momentum'],
                       help="Optimizer")
    group.add_argument('--momentum', type=float, default=0.9,
                       help="Momentum coefficient.")
    group.add_argument('--predict-file', type=str,
                       help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    group.add_argument('--output-dir', type=str,
                       help="The output directory where the model checkpoints will be written.")
    group.add_argument("--learning-rate", type=float, default=5e-5,
                       help="The initial learning rate for Adam.")
    group.add_argument("--doc-stride", type=int, default=128,
                       help="When splitting up a long document into chunks, how much stride to take between chunks.")
    group.add_argument("--do-lower-case", action="store_true",
                       help="Case sensitive or not")
    group.add_argument("--verbose-logging", action="store_true",
                       help="If true, all of the warnings related to data processing will be printed. "
                       "A number of warnings are expected for a normal SQuAD evaluation.")
    group.add_argument("--version-2-with-negative", action="store_true",
                       help="If true, the SQuAD examples contain some that do not have an answer.")
    group.add_argument("--null-score-diff-threshold", type=float, default=0.0,
                       help="If null_score - best_non_null is greater than the threshold predict null.")
    group.add_argument("--max-query-length", type=int, default=64,
                       help="The maximum number of tokens for the question. Questions longer than "
                       "this will be truncated to this length.")
    group.add_argument("--n-best-size", type=int, default=20,
                       help="The total number of n-best predictions to generate in the "
                       "nbest_predictions.json output file.")
    group.add_argument("--max-answer-length", type=int, default=30,
                       help="The maximum length of an answer that can be generated. This is needed "
                       "because the start and end predictions are not conditioned on one another.")
    group.add_argument('--adam-beta1', type=float, default=0.9,
                       help="Adam beta1 coefficient.")
    group.add_argument('--adam-beta2', type=float, default=0.999,
                       help="Adam beta2 coefficient.")
    group.add_argument('--adam-eps', type=float,
                       default=1e-6, help="Adam epsilon.")

    '''*****************************ipu*****************************'''
    group.add_argument('--pipeline-depth', type=int, default=1,
                       help="Depth of pipeline to use. Must also set --shards > 1.")
    group.add_argument('--pipeline-schedule', type=str, default="Interleaved",
                       choices=pipeline_schedule_options,
                       help="Pipelining scheduler. Choose between 'Interleaved' and 'Grouped'")
    group.add_argument('--replicas', type=int, default=1,
                       help="Replicate graph over N workers to increase batch to batch-size*N")
    group.add_argument('--max-cross-replica-buffer-size', type=int, default=10 * 1024 * 1024,
                       help="""The maximum number of bytes that can be waiting before a cross
                                        replica sum op is scheduled. [Default=10*1024*1024]""")
    group.add_argument('--hidden-layers-per-ipu', type=int, default=1,
                       help='number of hidden and ff layers per IPU.')
    group.add_argument('--precision', type=str, default="16", choices=["16", "32"],
                       help="Precision of Ops(weights/activations/gradients) data types: 16, 32.")
    group.add_argument('--no-stochastic-rounding', action="store_true",
                       help="Disable Stochastic Rounding")
    group.add_argument('--batches-per-step', type=int, default=1000,
                       help="Maximum number of batches to perform on the device before returning to the host.")
    group.add_argument('--fp-exceptions', action="store_true",
                       help="Turn on floating point exceptions")
    group.add_argument('--xla-recompute', action="store_true",
                       help="Allow recomputation of activations to reduce memory usage")
    group.add_argument('--seed', default=None,
                       help="Seed for randomizing training")
    group.add_argument('--no-outlining', type=bool, nargs="?", const=True, default=False,
                       help="Disable TF outlining optimisations. This will increase memory for a small throughput improvement.")
    group.add_argument('--available-memory-proportion', default=0.23, nargs='+',
                       help="Proportion of memory which is available for convolutions. Use a value of less than 0.6 "
                            "to reduce memory usage.")
    group.add_argument('--half-partial', type=str, default="float", choices=["half", "float"],
                       help="Mamul&Conv precision data type. Choose between 'half' and 'float'")

    return parser


def set_training_defaults(opts):
    opts['name'] = 'BERT_' + opts['task']
    opts['total_batch_size'] = opts['batch_size'] * opts['pipeline_depth']
    opts['summary_str'] = "Training\n"
    opts['summary_str'] += " Batch Size: {total_batch_size}\n"
    if opts['pipeline_depth'] > 1:
        opts['summary_str'] += "  Pipelined depth {pipeline_depth} \n"
    opts['summary_str'] += (" Base Learning Rate: 2**{base_learning_rate}\n"
                            " Loss Scaling: {loss_scaling}\n")
    opts['summary_str'] += " Epochs: {epochs}\n"

    if opts['optimizer'] == 'SGD':
        opts['summary_str'] += "SGD\n"
    elif opts['optimizer'] == 'momentum':
        opts['name'] += '_Mom'
        opts['summary_str'] += ("SGD with Momentum\n"
                                " Momentum Coefficient: {momentum}\n")


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
        tf.set_random_seed(random.randint(0, 2 ** 32 - 1))
        # numpy seed
        np.random.seed(random.randint(0, 2 ** 32 - 1))
        # ipu seed
        reset_ipu_seed(random.randint(-2 ** 16, 2 ** 16 - 1))
    opts['summary_str'] += (' {hostname}\n'
                            ' {datetime}\n')


def set_defaults(opts, lr_schedule):
    dataset.set_defaults(opts)
    set_training_defaults(opts)
    lr_schedule.set_defaults(opts)
    set_ipu_defaults(opts)
    logger.set_defaults(opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SQuAD Training/Inference in TensorFlow',
                                     add_help=False)
    parser = add_arguments(parser)
    tf.logging.set_verbosity(tf.logging.ERROR)

    opts = vars(parser.parse_args())
    if opts['help']:
        parser.print_help()
    else:
        opts.update(bert_ipu.BertConfig.from_json_file(
            opts['config']))
        try:
            lr_schedule = importlib.import_module(
                "LR_Schedules." + opts['lr_schedule'])
        except ImportError:
            raise ValueError(
                "LR_Schedules/{}.py not found".format(opts['lr_schedule']))
        opts["command"] = ' '.join(sys.argv)
        set_defaults(opts, lr_schedule)
        logger.print_to_file_and_screen(
            "Command line: " + opts["command"], opts)
        logger.print_to_file_and_screen(
            opts["summary_str"].format(**opts), opts)
        opts["summary_str"] = ""
        logger.print_to_file_and_screen(opts, opts)
        train(opts)
