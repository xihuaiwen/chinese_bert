# Copyright 2020 Graphcore Ltd.
"""Serialised BERT Encoder layers."""
import numpy as np

from bert_model import ExecutionMode
from pingpong.bert_layers import Attention, FeedForward, MaskLM
from pingpong.layers import Add, Dropout, Embedding, Norm
from pingpong.nn import Block
from pingpong.scope_manager import ScopeProvider

__all__ = [
    "EmbeddingSerialised",
    "BertEmbedding",
    "AttentionSplitIO",
    "AttentionSplitHidden",
    "FeedForwardSplitHidden",
    "FeedForwardSplitIO",
    "MaskLMSerialised",
]


def generate_simplified_periodic_pos_data(dtype, shape, scale=4):
    def value(x, y):
        return .02 / .707 * np.cos(2 * scale * np.pi * x * y / shape[1])

    X, Y = np.mgrid[:shape[0], :shape[1]]
    return np.vectorize(value)(X, Y).astype(dtype)


def detach(builder, input_x, pass_through_creation=1):
    return builder.customOp(
        opName="Detach",
        opVersion=1,
        domain="ai.graphcore",
        inputs=[input_x],
        attributes={"pass_through_creation": pass_through_creation})[0]


def constant_tensor(builder, value, dtype=None, debug_name=""):
    value = np.array(value)
    if dtype is not None:
        value = value.astype(dtype)
    return builder.aiOnnx.constant(value, debug_name)


class EmbeddingSerialised(Block):
    def __init__(self,
                 scope,
                 input_dim: int,
                 output_dim: int,
                 num_splits: int,
                 dtype: str = "float16",
                 weight_initializer: np.array = None,
                 sparse_grad: bool = False,
                 custom: bool = False,
                 detach: bool = False,
                 weight_transposed: bool = False,
                 skip_scopes = True,
                 **kwargs):
        """Turns non-negative integers (indexes/tokens) into dense vectors
        of fixed size.

        Args:
            input_dim (int): Size of the vocabulary, i.e. maximum integer index + 1.
            output_dim (int): Dimension of the dense embedding.
            dtype (str, optional): Data type of output embeddings. Defaults to 'float16'.
            weight_initializer (np.array, optional): Initializer for the `embeddings` matrix. Defaults to None.
            sparse_grad (bool, optional): If True, gradient w.r.t. weight will be a 'row_sparse'. Defaults to False.

        Returns:
            str:  Output tensor of shape (x0, x1, ... xN-1, output_dim) where (x0, x1, .... xN-1) is shape of input tensor.
        """
        super().__init__(params=[], scope=scope, **kwargs)
        self.dtype = dtype
        self.popart_dtype = 'FLOAT' if dtype == np.float32 else 'FLOAT16'

        if input_dim % num_splits:
            raise ValueError('`input_dim` must be a multiple of `num_splits`.')
        if hasattr(scope, 'ping_pong_phase'):
            raise ValueError(
                'Split embedding cannot have a single ping pong scope.')

        layers = []
        self.split_input_dim = input_dim // num_splits
        # If masks are pre-fetched in ping-pong phase 0 & 1
        # embedding ping pong phase starts at phase = 2.
        if skip_scopes:
            _ = kwargs['scope_provider'].get_scope(name=f'Token/skip',
                                                   ping_pong_phase='next',
                                                   skip_scope=True)
        for i in range(num_splits):
            scope = kwargs['scope_provider'].get_scope(
                name=f'split{i}', ping_pong_phase='next')
            layers.append(
                Embedding(scope,
                          self.split_input_dim,
                          output_dim,
                          custom=custom,
                          detach=detach,
                          weight_transposed=weight_transposed,
                          dtype=dtype,
                          **kwargs))
        self.layers = layers
        self.custom = custom
        self.detach = detach

        self._kwargs = {'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, x_in: str):
        x_even_sum = None
        x_odd_sum = None

        for i, layer in enumerate(self.layers):
            with self.scope_provider(self.builder, layer.scope):
                x_split = self.builder.aiOnnx.sub([
                    x_in,
                    constant_tensor(self.builder, i * self.split_input_dim,
                                    np.uint32)
                ])
                mask = self.builder.aiOnnx.less([
                    x_split,
                    constant_tensor(self.builder, self.split_input_dim,
                                    np.uint32)
                ])
                mask = detach(self.builder, mask)
                masked_indices = self.builder.aiOnnx.mul([x_split,
                                                         self.builder.aiOnnx.cast([mask], "UINT32")])
                x_out = layer(masked_indices)
                fp_mask = self.builder.aiOnnx.cast([mask], self.popart_dtype)
                fp_mask = self.builder.aiOnnx.unsqueeze([fp_mask], [1])
                x_out = self.builder.aiOnnx.mul([x_out, fp_mask])

                if i % 2:
                    if i == 1:
                        x_odd_sum = x_out
                    else:
                        x_odd_sum = self.builder.aiOnnx.add([x_out, x_odd_sum])
                else:
                    if i == 0:
                        x_even_sum = x_out
                    else:
                        x_even_sum = self.builder.aiOnnx.add(
                            [x_out, x_even_sum])

        # final accumulation in the last layer's scope
        with self.scope_provider(self.builder, self.layers[-1].scope):
            return self.builder.aiOnnx.add([x_odd_sum, x_even_sum])


class BertEmbedding(Block):
    def __init__(self, vocab_size, hidden_size, sequence_length,
                 max_positional_length, num_vocab_splits, epsilon,
                 apply_dropout, dropout_prob, mode, dtype, detach,
                 weight_transposed, custom=True, **kwargs):
        scope_provider = kwargs['scope_provider']
        super().__init__(scope_provider.get_scope('Embeddings'), **kwargs)
        if num_vocab_splits > 1:
            self.token_embedding = EmbeddingSerialised(
                scope_provider.get_scope('Token'),
                input_dim=vocab_size,
                output_dim=hidden_size,
                num_splits=num_vocab_splits,
                custom=custom,
                dtype=dtype,
                detach=detach,
                weight_transposed=weight_transposed,
                **kwargs)
        else:
            self.token_embedding = Embedding(
                scope_provider.get_scope('Token', ping_pong_phase='next'),
                input_dim=vocab_size,
                output_dim=hidden_size,
                custom=custom,
                dtype=dtype,
                detach=detach,
                weight_transposed=weight_transposed,
                **kwargs)
        num_segments = 2
        self.segment_embedding = Embedding(
            scope_provider.get_scope(
                'Segment', ping_pong_phase='next'), num_segments,
            hidden_size, dtype, **kwargs)

        self.position_embedding = Embedding(
            scope_provider.get_scope('Position', ping_pong_phase='prev'),
            max_positional_length,
            hidden_size,
            dtype,
            weight_initializer=generate_simplified_periodic_pos_data(
                dtype, (max_positional_length, hidden_size)),
            **kwargs)

        self.add = Add(scope_provider.get_scope(
            'Sum', ping_pong_phase='prev'), **kwargs)
        self.norm = Norm(scope_provider.get_scope('Norm', ping_pong_phase='prev'),
                         hidden_size, epsilon, dtype, **kwargs)
        self.apply_dropout = apply_dropout
        if apply_dropout:
            self.dropout = Dropout(
                scope_provider.get_scope(
                    'Dropout', ping_pong_phase='prev'), dropout_prob,
                **kwargs)
        self.total_ping_pong_phases = self.total_phases()

    def forward(self, indices, positions, segments):
        # Size of act = [batch_size * seq_len, hidden_size]
        x = self.add([
            self.token_embedding(indices),
            self.segment_embedding(segments),
            self.position_embedding(positions)
        ])
        x = self.norm(x)
        if self.apply_dropout:
            return self.dropout(x)
        else:
            return x


class AttentionSplitIO(Block):
    def __init__(self, name: str, num_splits, hidden_size, num_heads,
                 serialize_matmul, avail_mem_prop, epsilon, dropout,
                 dropout_prob, attn_dropout, attn_dropout_prob, batch_size, sequence_length, dtype, task,
                 num_mask_tokens, use_default_mem_proportion, **kwargs):
        scope_provider = kwargs['scope_provider']
        if hidden_size % num_splits:
            raise ValueError('Hidden size must be a multiple of num_splits.')
        super().__init__(params=[], scope=scope_provider.get_scope(name), **kwargs)
        attention_splits = []
        self.split_size = hidden_size // num_splits
        self.name = name
        for i in range(num_splits):
            attention_splits.append(
                Attention(f"{name}Split{i}",
                          hidden_size // num_splits,
                          hidden_size,
                          num_heads,
                          serialize_matmul,
                          avail_mem_prop,
                          epsilon,
                          dropout,
                          dropout_prob,
                          attn_dropout,
                          attn_dropout_prob,
                          batch_size,
                          sequence_length,
                          dtype,
                          task,
                          num_mask_tokens,
                          use_default_mem_proportion=use_default_mem_proportion,
                          residual=False))
        self.layers = attention_splits
        self.accum_scope = scope_provider.get_scope(
            f'{self.name}/AttnAccum', 'next')
        self.norm = Norm(
            scope_provider.get_scope(
                f'{name}/AttnNorm', self.accum_scope.ping_pong_phase),
            hidden_size, epsilon, dtype, **kwargs)
        if dropout:
            self.dropout = Dropout(
                scope_provider.get_scope(f'{name}/AttnDropout',
                                         self.accum_scope.ping_pong_phase), dropout_prob,
                **kwargs)
        else:
            self.dropout = lambda x: x

    def forward(self, x_in: str, masks: str):
        split_attention_out = []

        for i, attention_split in enumerate(self.layers):
            with self.scope_provider(self.builder, self.layers[i].scope):
                x_split = self.builder.aiOnnxOpset9.slice(
                    [x_in],
                    axes=[1],
                    starts=[i * self.split_size],
                    ends=[(i + 1) * self.split_size])
                if i > 1:
                    attention_split.mask = attention_split[i % 2].mask
                split_attention_out.append(attention_split(x_split, masks))

        with self.scope_provider(self.builder, self.accum_scope):
            x = self.builder.aiOnnx.concat(split_attention_out, axis=1)
            x = self.dropout(x)
            x = self.builder.aiOnnx.add([x_in, x], 'Residual')
            x = self.norm(x)
        return x


class AttentionSplitHidden(Block):
    def __init__(self, name: str, num_splits, hidden_size, num_heads,
                 serialize_matmul, avail_mem_prop, epsilon, dropout,
                 dropout_prob, attn_dropout, attn_dropout_prob, batch_size, sequence_length, dtype, task,
                 num_mask_tokens, use_default_mem_proportion, **kwargs):
        scope_provider = kwargs['scope_provider']
        # AttentionSplitHidden splits the num_heads, keeping size_per_head same.
        # Since hidden_size = num_heads * size_per_head , num_heads and hiddden_size
        # should be multiple of num_splits.

        if hidden_size % num_splits:
            raise ValueError('Hidden size must be a multiple of num_splits.')

        if num_heads % num_splits:
            raise ValueError('Num heads must be a multiple of num_splits.')

        super().__init__(params=[], scope=scope_provider.get_scope(name), **kwargs)
        attention_splits = []
        self.split_size = hidden_size // num_splits
        self.name = name
        for i in range(num_splits):
            attention_splits.append(
                Attention(f"Split{i}",
                          hidden_size,
                          self.split_size,
                          num_heads // num_splits,
                          serialize_matmul,
                          avail_mem_prop,
                          epsilon,
                          dropout,
                          dropout_prob,
                          attn_dropout,
                          attn_dropout_prob,
                          batch_size,
                          sequence_length,
                          dtype,
                          task,
                          num_mask_tokens,
                          residual=False,
                          use_default_mem_proportion=use_default_mem_proportion,
                          **kwargs))
        self.layers = attention_splits
        self.accum_scope = scope_provider.get_scope(f'AttnAccum', 'next')
        self.norm = Norm(
            scope_provider.get_scope(
                f'AttnNorm', self.accum_scope.ping_pong_phase),
            hidden_size, epsilon, dtype, **kwargs)
        if dropout:
            self.dropout = Dropout(scope_provider.get_scope(f'AttnDropout',
                                                            self.accum_scope.ping_pong_phase),
                                   dropout_prob,
                                   dtype=dtype,
                                   **kwargs)
        else:
            self.dropout = lambda x: x

    def forward(self, x_in: str, masks: str):
        x_odd_sum = None
        x_even_sum = None

        for i, attention_split in enumerate(self.layers):
            with self.scope_provider(self.builder, self.layers[i].scope):
                if i > 1:
                    attention_split.mask = self.layers[i % 2].mask
                x_out = attention_split(x_in, masks)
                if i % 2:
                    if i == 1:
                        x_odd_sum = x_out
                    else:
                        x_odd_sum = self.builder.aiOnnx.add([x_out, x_odd_sum])
                else:
                    if i == 0:
                        x_even_sum = x_out
                    else:
                        x_even_sum = self.builder.aiOnnx.add(
                            [x_out, x_even_sum])

        # final accumulation in the last layer's scope
        with self.scope_provider(self.builder, self.layers[-1].scope):
            x = self.builder.aiOnnx.add([x_odd_sum, x_even_sum])

        with self.scope_provider(self.builder, self.accum_scope):
            x = self.dropout(x)
            x = self.builder.aiOnnx.add([x_in, x], 'Residual')
            x = self.norm(x)
        return x


class FeedForwardSplitHidden(Block):
    def __init__(self, name, num_splits, input_size, ff_size, dropout,
                 dropout_prob, epsilon, use_default_memory_proportion,
                 available_memory_proportion, **kwargs):
        scope_provider = kwargs['scope_provider']
        super().__init__(params=[], scope=scope_provider.get_scope(name), **kwargs)
        ffwd_splits = []
        self.split_size = ff_size // num_splits
        self.name = name
        for i in range(num_splits):
            ffwd_splits.append(
                FeedForward(f'Split{i}',
                            input_size,
                            self.split_size,
                            dropout,
                            dropout_prob,
                            epsilon,
                            residual=False,
                            increment_scope=True,
                            use_default_memory_proportion=use_default_memory_proportion,
                            available_memory_proportion=available_memory_proportion,
                            **kwargs))
        self.layers = ffwd_splits
        self.accum_scope = scope_provider.get_scope(f'FFAccum', 'next')
        self.norm = Norm(
            scope_provider.get_scope(
                f'FFNorm', self.accum_scope.ping_pong_phase), input_size,
            epsilon, **kwargs)
        if dropout:
            self.dropout = Dropout(
                scope_provider.get_scope(
                    f'FFDropout', self.accum_scope.ping_pong_phase),
                dropout_prob, **kwargs)
        else:
            self.dropout = lambda x: x
        self.total_ping_pong_phases = self.total_phases()

    def forward(self, x_in: str):
        x_odd_sum = None
        x_even_sum = None

        for i, ffwd_split in enumerate(self.layers):
            with self.scope_provider(self.builder, self.layers[i].scope):
                x_out = ffwd_split(x_in)
                if i % 2:
                    if i == 1:
                        x_odd_sum = x_out
                    else:
                        x_odd_sum = self.builder.aiOnnx.add([x_out, x_odd_sum])
                else:
                    if i == 0:
                        x_even_sum = x_out
                    else:
                        x_even_sum = self.builder.aiOnnx.add(
                            [x_out, x_even_sum])

        # final accumulation in the last layer's scope
        with self.scope_provider(self.builder, self.layers[-1].scope):
            x = self.builder.aiOnnx.add([x_odd_sum, x_even_sum])

        with self.scope_provider(self.builder, self.accum_scope):
            x = self.dropout(x)
            x = self.builder.aiOnnx.add([x_in, x], 'Residual')
            x = self.norm(x)
        return x


class FeedForwardSplitIO(Block):
    def __init__(self, name, num_splits, input_size, ff_size, dropout,
                 dropout_prob, epsilon, use_default_memory_proportion,
                 available_memory_proportion, **kwargs):
        scope_provider = kwargs['scope_provider']
        super().__init__(params=[], scope=scope_provider.get_scope(name), **kwargs)
        ffwd_splits = []
        self.split_size = input_size // num_splits
        self.name = name
        for i in range(num_splits):
            ffwd_splits.append(
                FeedForward(f'{name}/Split{i}',
                            self.split_size,
                            ff_size,
                            dropout,
                            dropout_prob,
                            epsilon,
                            residual=False,
                            use_default_memory_proportion=use_default_memory_proportion,
                            available_memory_proportion=available_memory_proportion,
                            **kwargs))
        self.layers = ffwd_splits
        self.accum_scope = scope_provider.get_scope(f'{name}/FFAccum', 'next')
        self.norm = Norm(
            scope_provider.get_scope(
                f'{name}/FFNorm', self.accum_scope.ping_pong_phase),
            input_size, epsilon, **kwargs)
        if dropout:
            self.dropout = Dropout(
                scope_provider.get_scope(f'{name}/FFDropout',
                                         self.accum_scope.ping_pong_phase), dropout_prob,
                **kwargs)
        else:
            self.dropout = lambda x: x
        self.total_ping_pong_phases = self.total_phases()

    def forward(self, x_in: str):
        split_ffwd_out = []

        for i, ffwd_split in enumerate(self.layers):
            with self.scope_provider(self.builder, self.layers[i].scope):
                x_split = self.builder.aiOnnxOpset9.slice(
                    [x_in],
                    axes=[1],
                    starts=[i * self.split_size],
                    ends=[(i + 1) * self.split_size])
                split_ffwd_out.append(ffwd_split(x_split))

        with self.scope_provider(self.builder, self.accum_scope):
            x = self.builder.aiOnnx.concat(split_ffwd_out, axis=1)
            x = self.dropout(x)
            x = self.builder.aiOnnx.add([x_in, x], 'Residual')
            x = self.norm(x)
        return x


class MaskLMSerialised(Block):
    def __init__(self, num_splits, vocab_size, hidden_size, sequence_length,
                 batch_size, num_mask_tokens, projection_weights, **kwargs):
        scope_provider = kwargs['scope_provider']
        super().__init__(params=[],
                         scope=scope_provider.get_scope(name='MLMSerialised'),
                         **kwargs)
        self.slice_scope = scope_provider.get_scope('Slice', 'next')
        self.batch_size = batch_size
        self.vocab_length = vocab_size
        self.hidden_size = hidden_size
        self.sequence_len = sequence_length
        self.num_mask_tokens = num_mask_tokens
        layers = []
        for i in range(num_splits):
            layers.append(
                MaskLM(f'Split{i}',
                       vocab_size // num_splits,
                       hidden_size,
                       sequence_length,
                       batch_size,
                       num_mask_tokens,
                       projection_weights[i],
                       slice_input=False,
                       **kwargs))
        self.layers = layers
        self.total_ping_pong_phases = self.total_phases()

    def forward(self, x_in):
        with self.scope_provider(self.builder, self.slice_scope):
            x = self.builder.reshape_const(
                self.builder.aiOnnx, [x_in],
                [self.batch_size, self.sequence_len, self.hidden_size])

            x = self.builder.aiOnnxOpset9.slice([x],
                                                axes=[1],
                                                starts=[0],
                                                ends=[self.num_mask_tokens])
        projection_splits = []
        for layer in self.layers:
            projection_splits.append(layer(x))

        # Stack outputs in the last split scope
        with self.scope_provider(self.builder, self.layers[-1].scope):
            x = self.builder.aiOnnx.concat(projection_splits, axis=2)

        return x
