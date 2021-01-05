# Copyright 2020 Graphcore Ltd.
from pathlib import Path
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent


class TestBuildAndRun(SubProcessChecker):

    def setUp(self):
        ''' Compile the start here and complete versions of the tutorial code '''
        self.run_command("make clean", working_path, [])
        self.run_command("make all", working_path, [])

    def tearDown(self):
        self.run_command("make clean", working_path, [])

    @pytest.mark.category1
    def test_run_start_here(self):
        ''' Check that the start here version of the tutorial code runs '''

        self.run_command("./tut4_start_here",
                         working_path,
                         ["Program complete", "Memory Usage:"])

    @pytest.mark.category1
    def test_run_complete(self):
        ''' Check that the complete version of the tutorial code runs '''

        self.run_command("./tut4_complete",
                         working_path,
                         ["Program complete",
                          "Memory Usage:",
                          "popops::Transpose2d",
                          "poplin::ReduceAdd"])
