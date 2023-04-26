import unittest

from pyparsing import ParseBaseException

from epymorph.parser.common import Duration
from epymorph.parser.move_clause import Daily, daily


class TestDailyMoveClause(unittest.TestCase):
    def test_successful_01(self):
        case = '[mtype: days=[M,Th,Sa]; leave=1; duration=1d; return=2; function=\ndef(t):\n    return 1\n]'
        exp = Daily(
            days=['M', 'Th', 'Sa'],
            leave_step=1,
            duration=Duration(1, 'd'),
            return_step=2,
            f='def(t):\n    return 1'
        )
        act = daily.parse_string(case)[0]
        self.assertEqual(act, exp)

    def test_successful_02(self):
        case = '[mtype: days=all; leave=0; duration=2w; return=1; function=\ndef(t):\n    return 1\n]'
        exp = Daily(
            days=['M', 'T', 'W', 'Th', 'F', 'Sa', 'Su'],
            leave_step=0,
            duration=Duration(2, 'w'),
            return_step=1,
            f='def(t):\n    return 1'
        )
        act = daily.parse_string(case)[0]
        self.assertEqual(act, exp)
