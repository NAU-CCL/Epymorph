import unittest

from pyparsing import ParseBaseException

from epymorph.parser.move_steps import MoveSteps, move_steps


class TestMoveSteps(unittest.TestCase):
    def test_successful(self):
        cases = [
            '[move-steps: per-day=2; duration=[2/3, 1/3]]',
            '[move-steps: per-day=1; duration=[2/3]]',
            '[ move-steps : per-day = 3 ; duration = [0.25, 0.25, 0.5] ]',
            '[move-steps: per-day=3; duration=[0.25, 0.25, 0.5]]',
            '[move-steps:per-day=3;duration=[0.25, 0.25, 0.5]]'
        ]
        exps = [
            MoveSteps([2/3, 1/3]),
            MoveSteps([2/3]),
            MoveSteps([0.25, 0.25, 0.5]),
            MoveSteps([0.25, 0.25, 0.5]),
            MoveSteps([0.25, 0.25, 0.5])
        ]
        for c, e in zip(cases, exps):
            a = move_steps.parse_string(c)[0]
            self.assertEqual(a, e, f"{str(a)} did not match {str(e)}")

    def test_failures(self):
        cases = [
            '[move-steps: duration=[2/3, 1/3]; per-day=2]',
            '[move-steps: per-day=2; duration=[]]',
            '[move-steps-2: per-day=2; duration=[]]'
        ]
        for c in cases:
            with self.assertRaises(ParseBaseException):
                move_steps.parse_string(c)[0]
