import unittest

from experiments.meanflow_eval.backends import effective_execute_steps


class MP1ExecutionLengthTests(unittest.TestCase):
    def test_root_checkpoint_uses_native_slice_length(self):
        self.assertEqual(effective_execute_steps(4, 2, 4), 3)
        self.assertEqual(effective_execute_steps(8, 2, 4), 4)
        self.assertEqual(effective_execute_steps(4, 2, 2), 2)

    def test_explicit_length_is_checked_without_silent_clamping(self):
        self.assertEqual(effective_execute_steps(4, 2, 4, 2), 2)
        for requested in (0, -1, 4):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                effective_execute_steps(4, 2, 4, requested)
        with self.assertRaises(ValueError):
            effective_execute_steps(1, 3, 4)


if __name__ == "__main__":
    unittest.main()
