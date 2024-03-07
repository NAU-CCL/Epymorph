# pylint: disable=missing-docstring
import unittest

import numpy as np
from pyparsing import ParseBaseException

from epymorph.data_shape import Shapes
from epymorph.geo.spec import CentroidDType
from epymorph.movement.parser import (Attribute, DailyClause, MoveSteps,
                                      attribute, daily, move_steps)
from epymorph.movement.parser_util import Duration


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
            MoveSteps([2 / 3, 1 / 3]),
            MoveSteps([2 / 3]),
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
                move_steps.parse_string(c)


class TestAttribute(unittest.TestCase):
    def test_successful(self):
        cases = [
            '[attrib: source=geo; name=commuters; shape=NxN; dtype=int; description="hey1"]',
            '[attrib: source=params; name=move_control; shape=TxN; dtype=float; description="hey2"]',
            '[attrib: source=params; name=move_control; shape=TxN; dtype=float;\n    description="hey3"]',
            '[attrib:\nsource=params;\nname=theta;\nshape=S;\ndtype=str;\ndescription="hey4"]',
            '[attrib: source=geo; name=centroids; shape=N; dtype=[(longitude, float), (latitude, float)]; description="hey5"]',
        ]
        exps = [
            Attribute('commuters', Shapes.NxN, np.int64, 'geo', 'hey1'),
            Attribute('move_control', Shapes.TxN, np.float64, 'params', 'hey2'),
            Attribute('move_control', Shapes.TxN, np.float64, 'params', 'hey3'),
            Attribute('theta', Shapes.S, np.str_, 'params', 'hey4'),
            Attribute('centroids', Shapes.N, CentroidDType, 'geo', 'hey5'),
        ]
        for c, e in zip(cases, exps):
            a = attribute.parse_string(c)[0]
            self.assertEqual(a, e, f"{str(a)} did not match {str(e)}")

    def test_failures(self):
        cases = [
            '[attrib: source=blah; name=commuters; shape=NxN; dtype=int; description="hey1"]',
            '[attrib: source=params; name=move_control; shape=TxN; dtype=uint8; description="hey2"]',
            '[attrib: source=params; name=move_control; shape=TxA; dtype=float; description="hey3"]',
        ]
        for c in cases:
            with self.assertRaises(ParseBaseException):
                move_steps.parse_string(c)


class TestDailyMoveClause(unittest.TestCase):
    def test_successful_01(self):
        case = '[mtype: days=[M,Th,Sa]; leave=2; duration=1d; return=4; function=\ndef(t):\n    return 1\n]'
        exp = DailyClause(
            days=['M', 'Th', 'Sa'],
            leave_step=1,
            duration=Duration(1, 'd'),
            return_step=3,
            function='def(t):\n    return 1'
        )
        act = daily.parse_string(case)[0]
        self.assertEqual(act, exp)

    def test_successful_02(self):
        case = '[mtype: days=all; leave=1; duration=2w; return=2; function=\ndef(t):\n    return 1\n]'
        exp = DailyClause(
            days=['M', 'T', 'W', 'Th', 'F', 'Sa', 'Su'],
            leave_step=0,
            duration=Duration(2, 'w'),
            return_step=1,
            function='def(t):\n    return 1'
        )
        act = daily.parse_string(case)[0]
        self.assertEqual(act, exp)

    def test_failed_01(self):
        # Invalid leave step
        with self.assertRaises(ParseBaseException):
            case = '[mtype: days=all; leave=0; duration=2w; return=2; function=\ndef(t):\n    return 1\n]'
            daily.parse_string(case)

    def test_failed_02(self):
        # Invalid days value
        with self.assertRaises(ParseBaseException):
            case = '[mtype: days=[X]; leave=0; duration=2w; return=2; function=\ndef(t):\n    return 1\n]'
            daily.parse_string(case)
