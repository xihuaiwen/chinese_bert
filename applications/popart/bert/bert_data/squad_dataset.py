# Copyright 2019 Graphcore Ltd.
import os
import numpy as np
import random
import pickle
import json
import fractions
import math
import subprocess
from logging import getLogger
from functools import reduce

from .dataset import DataSet
from .data_sampler import SequentialSampler, ShuffledSampler, DistributedDataSampler
from .tokenization import FullTokenizer
from .squad_utils import read_squad_examples, convert_examples_to_features, RawResult, write_predictions, InputFeatures

logger = getLogger(__name__)


def generate_synthetic_features(sequence_length, vocab_length, batch_size):
    features = []
    for i in range(batch_size):
        features.append(InputFeatures(
            i,
            None,
            None,
            None,
            None,
            None,
            np.random.randint(0, vocab_length, size=sequence_length),
            None,
            np.random.randint(0, 2, size=sequence_length),
            0,
            None,
            None,
            np.random.randint(0, sequence_length, size=1),
            np.random.randint(0, sequence_length, size=1),
            None,
            np.random.randint(0, sequence_length+1, size=1)
        ))
    return features


class SquadDataLoader(object):
    def __init__(self,
                 features,
                 sequence_length=None,
                 batch_size=1,
                 dtype=np.int32,
                 sampler=None):
        self.features = features
        self.batch_size = batch_size
        self.dtype = dtype
        self.sequence_length = sequence_length

        self.sampler = sampler
        if sampler is None:
            self.sampler = SequentialSampler(features)
        self.num_batches = len(self.sampler)//self.batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.feature_iterator = iter([self.features[idx] for idx in self.sampler])
        return self

    def __next__(self):
        items = [next(self.feature_iterator) for _ in range(self.batch_size)]

        indicies = []
        positions = []
        segments = []
        sequence_mask_idx = []
        start_pos = []
        end_pos = []
        uid = []

        for item in items:
            indicies.append(item.input_ids)
            padding_max = self.sequence_length if self.sequence_length is not None else len(item.input_ids)
            padding_length = len(item.input_ids) - item.padding_start_index
            position_padding = np.full(padding_length, padding_max)
            position_ids = np.arange(0, item.padding_start_index)
            positions.append(np.concatenate((position_ids, position_padding)).astype(np.int32))
            segments.append(item.segment_ids)
            sequence_mask_idx.append(item.padding_start_index)
            start_pos.append(item.start_position)
            end_pos.append(item.end_position)
            uid.append(item.unique_id)
            # Including impossible samples during training is under investigation. T12851
            # if item.is_impossible:
            #     logger.warning("Impossible sample exists in the dataset. "
            #                 f"start pos: {item.start_position}, end pos: {item.end_position}")

        inputs = []
        for i in [indicies, positions, segments, sequence_mask_idx, start_pos, end_pos, uid]:
            inputs.append(np.stack(i))

        return inputs


class BertDataTransform(object):
    '''
    Masks the indices that are larger than the vocab_length
    '''
    def __init__(self, dataloader, vocab_length, sequence_length, embedding_dict,  positional_dict, merge_both_embeddings, is_training=True):
        self.dataloader = dataloader
        self.vocab_length = vocab_length
        self.sequence_length = sequence_length
        self.is_training = is_training
        self.embedding_dict = embedding_dict
        self.positional_dict = positional_dict
        self.merge_both_embeddings = merge_both_embeddings

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        self.dataloader_iterator = iter(self.dataloader)
        return self

    def __next__(self):
        items = next(self.dataloader_iterator)
        # Specific BERT Post Processing. TODO: Find a better place for this processing
        # The vocab_length may be smaller than the original vocab... In this case with the custom_op
        # Out of Bounds indicies over a certain threshold will cause numerical issues.
        # 100 is unknown token [UNK]
        # 0 in the label is padding
        OOB = items[0] >= self.vocab_length
        items[0][OOB] = 100

        # Force use of uint32 for all inputs.
        for i in range(len(items)):
            if self.is_training or i < 4:
                items[i] = items[i].astype(np.uint32)
        if self.embedding_dict is not None:
            items[0] = np.take(self.embedding_dict, items[0], 0)
        if self.positional_dict is not None:
            positional_expanded = np.take(self.positional_dict, items[1], 0)
            if self.merge_both_embeddings:
                items[0] += positional_expanded
            else:
                items[1] = positional_expanded
        return items


def load_or_cache_features(input_file,
                           vocab_file,
                           sequence_length,
                           is_training=True,
                           cache_file=None,
                           overwrite_cache=False,
                           do_lower_case=False):
    if cache_file is None:
        cache_file = input_file + f".{sequence_length}.cache"

    if os.path.exists(cache_file) and not overwrite_cache:
        examples = None
        logger.info(f"Loading Cache {cache_file}")
        with open(cache_file, "rb") as f:
            features = pickle.load(f)
    else:
        logger.info("Reading Examples")
        examples = read_squad_examples(input_file=input_file,
                                       is_training=is_training,
                                       version_2_with_negative=False)

        # google-research/bert uses sequence_length 384 with doc_stride 128
        # TODO: Find a good value for the doc_stride with sequence_length <384
        doc_stride = 128
        if sequence_length < 384:
            doc_stride = 64

        logger.info("Converting to Features")
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=FullTokenizer(vocab_file, do_lower_case=do_lower_case),
                                                max_seq_length=sequence_length,
                                                doc_stride=doc_stride,
                                                max_query_length=64,
                                                is_training=is_training)

        logger.info(f"Saving Cache {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(features, f)

    return features, examples


class SquadDataSet(DataSet):
    def __init__(self,
                 features,
                 examples,
                 input_file,
                 is_training,
                 output_dir=None,
                 evaluate_script=None,
                 do_lower_case=False,
                 n_extra=0,
                 **kwargs):
        super().__init__(**kwargs)

        self.features = features
        self.examples = examples
        self.is_training = is_training

        self.input_file = input_file
        self.output_dir = output_dir

        self.do_lower_case = do_lower_case

        if not self.is_training and self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            # If examples is None, features was loaded from the cache
            # So the examples need to be recreated.
            if self.examples is None:
                self.examples = read_squad_examples(input_file=self.input_file,
                                                    is_training=self.is_training,
                                                    version_2_with_negative=False)

        self.results = []
        self.evaluate_script = evaluate_script
        self.n_extra = n_extra

    def add_results(self, data, logits):
        # Results will be batched. Flatten to individual results
        start_logits, end_logits = [
            logit.reshape(-1, logit.shape[-1]).tolist()
            for logit in logits]
        for i, unique_id in enumerate(data["uid"]):
            self.results.append(RawResult(
                unique_id=unique_id,
                start_logits=start_logits[i],
                end_logits=end_logits[i]
            ))

    def write_predictions(self, epoch=None):
        if self.is_training:
            raise RuntimeError("Predictions cannot be written for training datasets")

        if self.output_dir is None:
            raise RuntimeError("Predictions cannot be written when output_dir is None")

        suffix = f"_{epoch}" if epoch is not None else ""
        predictions_file = os.path.join(self.output_dir, f"predictions{suffix}.json")
        nbest_file = os.path.join(self.output_dir, f"nbest_predictions{suffix}.json")
        null_log_odds_file = os.path.join(self.output_dir, f"null_odds{suffix}.json")
        self.results = self.results[:len(self.results) - self.n_extra]
        write_predictions(self.examples,
                          self.features,
                          self.results,
                          20, 30,
                          self.do_lower_case,
                          predictions_file,
                          nbest_file,
                          null_log_odds_file,
                          True,
                          False, 0)

        if self.evaluate_script is not None:
            evaluation = subprocess.check_output(["python", self.evaluate_script, self.input_file, predictions_file])
            evaluation = json.loads(evaluation)
            f1 = evaluation["f1"]
            exact_match = evaluation["exact_match"]
            status_string = f"F1 Score: {f1} | Exact Match: {exact_match}"
            if epoch is not None:
                status_string = f"Epoch: {epoch:3}{args.epochs - 1} | " + status_string
            logger.info(status_string)


def get_bert_dataset(tensor_shapes,
                     input_file,
                     output_dir,
                     sequence_length,
                     vocab_file,
                     vocab_length,
                     batch_size,
                     batches_per_step,
                     embedding_dict,
                     positional_dict,
                     merge_both_embeddings=False,
                     replication_factor=1,
                     accumulation_factor=1,
                     shuffle=True,
                     is_training=True,
                     overwrite_cache=False,
                     no_drop_remainder=False,
                     evaluate_script=None,
                     synthetic=False,
                     do_lower_case=False,
                     max_pipeline_stage=1,
                     seed=0,
                     mpi_size=1,
                     mpi_rank=0,
                     is_distributed=False):
    samples_per_step = batch_size * batches_per_step * \
        replication_factor * accumulation_factor

    div_factor = batch_size * replication_factor * accumulation_factor * batches_per_step

    pad = 0

    if synthetic:
        features = generate_synthetic_features(
            sequence_length, vocab_length, samples_per_step)
        examples = None
        output_dir = None
        logger.info("Generating synthetic dataset")
    else:
        features, examples = load_or_cache_features(
            input_file,
            vocab_file,
            sequence_length,
            is_training,
            overwrite_cache=overwrite_cache,
            do_lower_case=do_lower_case)

    if no_drop_remainder and not synthetic:
        # dataset will be padded to be divisible by batch-size and samples-per-step
        pad = int(np.ceil(len(features)/div_factor)) * div_factor - len(features)

    if is_distributed:
        sampler = DistributedDataSampler(
            features, seed, shuffle,
            mpi_size, mpi_rank, padding=False, padding_sub=pad, div_factor=div_factor)
        pad = sampler.get_subpadding_size()
    elif shuffle:
        sampler = ShuffledSampler(features, seed, pad)
    else:
        sampler = SequentialSampler(features, pad)
    if no_drop_remainder and not synthetic:
        logger.info(f"no_drop_remainder: Dataset padded by {pad} samples")

    dl = SquadDataLoader(
        features,
        sequence_length=sequence_length,
        batch_size=samples_per_step,
        sampler=sampler
        )

    bert_ds = BertDataTransform(
        dl,
        vocab_length,
        sequence_length,
        embedding_dict,
        positional_dict,
        merge_both_embeddings,
        is_training=is_training)

    if not is_training:
        # Add uid to the data dictionary so evaluation script can be run
        tensor_shapes += [
            ("start", None),
            ("end", None),
            ("uid", None)]

    ds = SquadDataSet(
        features,
        examples,
        input_file,
        is_training,
        output_dir,
        evaluate_script,
        do_lower_case=do_lower_case,
        n_extra=pad,
        loader=bert_ds,
        tensor_shapes=tensor_shapes,
        batches_per_step=batches_per_step,
        replication_factor=replication_factor,
        accumulation_factor=accumulation_factor)
    return ds
