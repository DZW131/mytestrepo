import json
import unittest

import numpy as np

from train_sshr import is_validation_improvement


class ValidationSelectionRecordingTest(unittest.TestCase):
    def test_numpy_comparison_returns_json_serializable_bool(self):
        decision = is_validation_improvement(
            np.float64(0.5990463331720243),
            np.float64(0.44944187540826697),
        )

        self.assertIs(type(decision), bool)
        self.assertTrue(decision)
        self.assertEqual(json.loads(json.dumps({'is_best_val': decision})), {
            'is_best_val': True,
        })

    def test_validation_only_selection_semantics_are_unchanged(self):
        self.assertTrue(is_validation_improvement(np.float64(0.4), None))
        self.assertFalse(
            is_validation_improvement(np.float64(0.4), np.float64(0.4))
        )
        self.assertFalse(
            is_validation_improvement(np.float64(0.3), np.float64(0.4))
        )
        self.assertFalse(is_validation_improvement(None, np.float64(0.4)))


if __name__ == '__main__':
    unittest.main()
