# Copyright 2019 Graphcore Ltd.
import argparse
import os
import time
import warnings

import numpy as np
import tensorflow as tf

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import cross_replica_optimizer
from tensorflow.python.ipu import gradient_accumulation_optimizer
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import utils
from tensorflow.python.ipu import embedding_ops
from tensorflow.python.ipu import rnn_ops
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.keras.layers import Dense

try:
    from gcprofile import save_tf_report
    use_poplar_text_report = False
except ImportError:
    use_poplar_text_report = True
    report_dest = 'profile_data'
    warnings.warn('Could not import gcprofile, falling back to text reports.', ImportWarning)


# Hyper-parameters
DTYPE = np.float16
LEARNING_RATE = 1e-5

# Discrete observations
DIS_OBS_DIMS = [12, 24, 20, 32]
DIS_OBS_CARDINALITY = [750, 6, 1501, 40]

# Continuous observations
CONT_OBS_DIMS = [[12, 32], [24, 8], [20, 8], [32, 16]]

# Policy network params
DIS_OBS_EMB_SIZE = [8, 8, 8, 16]
LSTM_HIDDEN_SIZE = 1024

# Action space
NUM_ACTIONS = 128


class EnvGenerator:
    """Generate a batch worth of observations and simulated rewards according to current policy."""

    def __init__(self, batch_size, time_steps):
        dis_obs = []
        cont_obs = []

        for (dim, cardinality) in zip(DIS_OBS_DIMS, DIS_OBS_CARDINALITY):
            dis_obs.append(
                np.random.randint(cardinality, size=(batch_size, time_steps, dim)).astype(np.int32))

        for index, dim in enumerate(CONT_OBS_DIMS):
            cont_obs.append(np.random.normal(size=(batch_size, time_steps, *dim)).astype(DTYPE))

        # Simulated reward
        rewards = [np.random.normal(size=(batch_size, time_steps)).astype(DTYPE)]

        # Initial state
        state_in = [np.random.normal(size=(batch_size, 2, LSTM_HIDDEN_SIZE)).astype(DTYPE)]

        self.batch = tuple(dis_obs + cont_obs + rewards + state_in)
        self.counter = 0

    def __next__(self):
        self.counter += 1
        return self.batch

    def reset_counter(self):
        self.counter = 0

    def get_counter(self):
        return self.counter

    def __call__(self):
        return self

    def __iter__(self):
        return self


# make core of policy network
def create_policy(*infeed_data):
    """Act according to current policy and generate action probability. """

    dis_obs = list(infeed_data[:4])
    cont_obs = list(infeed_data[4:8])
    state_in = infeed_data[-1]

    # Look up embedding for all the discrete obs
    emb_lookup = []
    with tf.variable_scope("popnn_lookup"):
        for index, obs in enumerate(dis_obs):
            emb_matrix = tf.get_variable(f'emb_matrix{index}', [DIS_OBS_CARDINALITY[index], DIS_OBS_EMB_SIZE[index]],
                                         DTYPE)
            emb_lookup.append(embedding_ops.embedding_lookup(emb_matrix, obs, name=f'emb_lookup{index}'))

    # Clip some continuous observations
    cont_obs[-1] = tf.clip_by_value(cont_obs[-1], -5.0, 5.0, name="clip")

    # Concat groups of observations
    obs_concat = []
    for d_obs, c_obs in zip(emb_lookup, cont_obs):
        obs_concat.append(tf.concat([d_obs, c_obs], axis=3, name="concat_obs"))

    # Fully connected transformations
    num_output = 8
    obs_concat[-1] = Dense(num_output, dtype=DTYPE)(obs_concat[-1])
    # Reduce max
    obs_concat = [tf.reduce_max(obs, axis=2) for obs in obs_concat]

    # Final concat of all the observations
    lstm_input = tf.concat(obs_concat, axis=2, name="concat_all")

    # LSTM layer
    lstm_input = tf.transpose(lstm_input, perm=[1, 0, 2],
                              name="pre_lstm_transpose")  # PopnnLSTM uses time-major tensors
    lstm_cell = rnn_ops.PopnnLSTM(num_units=LSTM_HIDDEN_SIZE, dtype=DTYPE, partials_dtype=DTYPE, name="lstm")
    lstm_output, state_out = lstm_cell(lstm_input, training=True,
                                       initial_state=tf.nn.rnn_cell.LSTMStateTuple(state_in[:, 0], state_in[:, 1]))
    lstm_output = tf.transpose(lstm_output, perm=[1, 0, 2], name="post_lstm_transpose")
    logits = Dense(NUM_ACTIONS, name="logits", dtype=DTYPE)(lstm_output)
    log_prob = tf.nn.log_softmax(logits, name="prob")

    # make action selection op (outputs int actions, sampled from policy)
    actions = tf.multinomial(logits=tf.reshape(logits, (-1, NUM_ACTIONS)), num_samples=1)
    actions = tf.reshape(actions, (args.batch_size, args.time_steps))

    action_masks = tf.one_hot(actions, NUM_ACTIONS, dtype=DTYPE)
    action_prob = tf.reduce_sum(action_masks * log_prob, axis=-1)

    return action_prob


def build_train_op(previous_loss, *infeed_data):
    """Construct loss and optimizer."""
    with ipu_scope("/device:IPU:0"):
        action_prob = create_policy(*infeed_data)
        loss = tf.reduce_sum(action_prob * infeed_data[-2])
        opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        if args.accumulate_grad:
            opt = gradient_accumulation_optimizer.GradientAccumulationOptimizer(
                opt, num_mini_batches=args.num_mini_batches)
        opt = cross_replica_optimizer.CrossReplicaOptimizer(opt)
        train_op = opt.minimize(loss)
        with tf.control_dependencies([train_op]):
            loss = tf.identity(loss)
        return previous_loss + loss


def train(replication_factor, batch_size, batch_per_step, profile, num_iter, time_steps):
    """Launch training."""

    # Set up in-feeds for the data
    with tf.device('cpu'):
        data_generator = EnvGenerator(batch_size, time_steps)
        items = next(data_generator)
        output_types = tuple((tf.dtypes.as_dtype(i.dtype) for i in items))
        output_shapes = tuple((tf.TensorShape(i.shape) for i in items))
        total_bytes = 0
        for i in items:
            total_bytes += i.nbytes
        print(f'Input data size = {total_bytes/1000000} MB/batch')
        dataset = tf.data.Dataset.from_generator(data_generator,
                                                 output_types=output_types,
                                                 output_shapes=output_shapes)
        infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, "InfeedQueue",
                                                       replication_factor=replication_factor)
        data_init = infeed_queue.initializer

    # Compile loss op
    with ipu_scope("/device:IPU:0"):
        total_loss = ipu_compiler.compile(lambda: loops.repeat(batch_per_step,
                                                               build_train_op,
                                                               infeed_queue=infeed_queue,
                                                               inputs=[tf.constant(0.0, dtype=DTYPE)]))
    # Set up report op optionally.
    if profile:
        with tf.device('cpu'):
            report = gen_ipu_ops.ipu_event_trace()

    # Set up session on IPU
    opts = utils.create_ipu_config(profiling=profile,
                                   use_poplar_text_report=use_poplar_text_report,
                                   profile_execution=profile,
                                   max_cross_replica_sum_buffer_size=10000000,
                                   merge_infeed_io_copies=True)
    opts = utils.auto_select_ipus(opts, [replication_factor])
    utils.configure_ipu_system(opts)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

    # Initialize variables
    utils.move_variable_initialization_to_cpu()
    sess.run([tf.global_variables_initializer(), data_init])

    # Run training and time
    total_time = 0.0
    total_samples = 0
    skip_iterations = 5  # Initially the infeed may buffer extra input data and
    # first run for IPU includes XLA compile, so skipping these iterations for calculating items/sec.
    for iters in range(num_iter):
        data_generator.reset_counter()
        t0 = time.perf_counter()
        sess.run(total_loss)
        t1 = time.perf_counter()

        if profile:
            raw_reports = sess.run(report)
            if use_poplar_text_report:
                # extract the report
                rep = utils.extract_all_strings_from_event_trace(raw_reports)
                print("Writing profiling report to %s" % report_dest)
                with open(report_dest, "w") as f:
                    f.write(rep)
            else:
                os.makedirs('profile_rl', exist_ok=True)
                save_tf_report(raw_reports, log_dir='profile_rl')
                print("Writing profiling report to profile_rl")
            break

        if iters > skip_iterations:
            total_time += (t1 - t0)
            total_samples += (batch_size * batch_per_step * replication_factor)
            print("Average %.1f items/sec" % (total_samples / total_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device.")
    parser.add_argument("--time_steps", type=int, default=16, help="Input sequence length.")
    parser.add_argument("--batch_per_step", type=int, default=10000,
                        help="Number of batches to run per step (on the device)")
    parser.add_argument("--num_iter", type=int, default=10, help="Number of training steps.")
    parser.add_argument("--num_ipus", type=int, default=16,
                        help="Number of IPUs to use for data-parallel replication")
    parser.add_argument('--accumulate_grad', default=False, dest='accumulate_grad',
                        action='store_true', help="Flag that turns on gradient accumulation.")
    parser.add_argument('--num_mini_batches', type=int, default=128,
                        help="Number of batches to accumulate gradients over, if accumulate_grad flag is on")
    parser.add_argument('--profile', default=False, dest='profile', action='store_true',
                        help='Collect profiling information.')
    parser.add_argument('--data', dest="data", type=str, default="real",
                        help="Run inference on real data (transfer images host -> device) "
                             "or using on-device synthetic data",
                        choices=["real", "synthetic"])
    args = parser.parse_args()
    if args.data == "synthetic":
        syn_flags = "--use_synthetic_data --synthetic_data_initializer=random"
        if 'TF_POPLAR_FLAGS' in os.environ:
            os.environ["TF_POPLAR_FLAGS"] += syn_flags
        else:
            os.environ["TF_POPLAR_FLAGS"] = syn_flags
    train(args.num_ipus, args.batch_size, args.batch_per_step, args.profile, args.num_iter, args.time_steps)
