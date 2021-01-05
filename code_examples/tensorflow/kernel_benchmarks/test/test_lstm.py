# Copyright 2020 Graphcore Ltd.
from pathlib import Path

import pytest
# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestTensorFlowLSTMBenchmarks(SubProcessChecker):
    """High-level integration tests for TensorFlow LSTM synthetic benchmarks"""

    @pytest.mark.category1
    def test_help(self):
        self.run_command("python3 lstm.py -h",
                         working_path,
                         "usage: lstm.py")

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_default(self):
        self.run_command("python3 lstm.py",
                         working_path,
                         [r"(\w+.\w+) items/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_inference_b256_s25_h1024(self):
        self.run_command("python3 lstm.py --batch-size 256 --timesteps 25 --hidden-size 1024 --use-data --popnn",
                         working_path,
                         [r"(\w+.\w+) items/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_inference_b128_s50_h1536(self):
        self.run_command("python3 lstm.py --batch-size 128 --timesteps 50 --hidden-size 1536 --popnn --save-graph",
                         working_path,
                         [r"(\w+.\w+) items/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_inference_b64_s25_h2048(self):
        self.run_command("python3 lstm.py --batch-size 64 --timesteps 25 --hidden-size 2048 --steps=6 --popnn",
                         working_path,
                         [r"(\w+.\w+) items/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_inference_b1024_s150_h256(self):
        self.run_command("python3 lstm.py --batch-size 1024 --timesteps 150 --hidden-size 256 --use-zero-values --batches-per-step 100",
                         working_path,
                         [r"(\w+.\w+) items/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_train_b64_s25_h512(self):
        self.run_command("python3 lstm.py --batch-size 64 --timesteps 25 --hidden-size 512 --train --use-data",
                         working_path,
                         [r"(\w+.\w+) items/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(2)
    def test_replicas(self):
        self.run_command("python3 lstm.py --replicas 2",
                         working_path,
                         [r"(\w+.\w+) items/sec"])
