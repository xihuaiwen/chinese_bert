# Copyright 2020 Graphcore Ltd.
import unittest
import os
import pytest
import tensorflow as tf
from examples_tests.test_util import SubProcessChecker

working_path = os.path.dirname(__file__)


@pytest.mark.category1
@pytest.mark.ipus(2)
class TensorFlow2Imdb(SubProcessChecker):
    """Integration tests for TensorFlow 2 IMDB example"""

    @unittest.skipIf(tf.__version__[0] != '2', "Needs TensorFlow 2")
    def test_default_commandline(self):
        self.run_command("python imdb.py",
                         working_path,
                         "Epoch 2/")
