# Copyright 2019 Graphcore Ltd.
from .attention_wrapper import AttentionWrapperNoAssert
from .decoder import dynamic_decode
from .helper import TrainingHelperNoCond, GreedyEmbeddingHelperNoCond
"""
This Module includes edits to tf.contrib.seq2seq to allow models to run on IPU

attention_wrapper.py
Changes:
- Removed tf.Assert (not supported) on batch_sizes from attention_mechanisms and inputs

decoder.py
Changes:
- Forced is_xla to be True (current detection does not work)
- Removed loop_condition (allows for cycle counts on h/w)

helper.py
Changes:
- Removed tf.cond's for early exit (allows for cycle counts on h/w)
"""
