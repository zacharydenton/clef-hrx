import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import memory_guard as guard


class MemoryGuardTests(unittest.TestCase):
    def test_pressure_thresholds(self):
        before = {'MemAvailable': 80 * guard.GIB, 'SwapFree': 10 * guard.GIB}
        self.assertFalse(guard.pressure(before, {'MemAvailable': 24 * guard.GIB, 'SwapFree': 9 * guard.GIB}))
        self.assertTrue(guard.pressure(before, {'MemAvailable': 24 * guard.GIB - 1, 'SwapFree': 10 * guard.GIB}))
        self.assertTrue(guard.pressure(before, {'MemAvailable': 80 * guard.GIB, 'SwapFree': 9 * guard.GIB - 1}))

    def test_refused_admission_does_not_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(guard, 'memory', return_value={'MemAvailable': 71 * guard.GIB, 'SwapFree': 0}), \
                    patch.object(guard.subprocess, 'Popen') as launch:
                code = guard.main(['--output', str(root / 'run'), '--memory-gib', '40',
                                   '--lock', str(root / 'lock'), '--', sys.executable, '-c', 'pass'])
                self.assertEqual(code, 75)
                launch.assert_not_called()
            self.assertEqual(json.loads((root / 'run/status.json').read_text())['state'], 'admission_refused')

    def test_busy_lock_does_not_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / 'lock').open('a') as lock, patch.object(guard.subprocess, 'Popen') as launch:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(guard.main(['--output', str(root / 'run'), '--memory-gib', '40',
                                            '--lock', str(root / 'lock'), '--', sys.executable, '-c', 'pass']), 75)
                launch.assert_not_called()

    def test_pressure_kills_only_owned_group(self):
        command = [sys.executable, '-c', 'import time; time.sleep(30)']
        other = subprocess.Popen(command, start_new_session=True)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                samples = [{'MemAvailable': 80 * guard.GIB, 'SwapFree': 10 * guard.GIB},
                           {'MemAvailable': 23 * guard.GIB, 'SwapFree': 10 * guard.GIB}]
                with patch.object(guard, 'memory', side_effect=samples):
                    self.assertEqual(guard.main(['--output', str(root / 'run'), '--memory-gib', '40',
                                                '--lock', str(root / 'lock'), '--', *command]), 75)
                state = json.loads((root / 'run/status.json').read_text())
                self.assertEqual(state['state'], 'held_for_memory')
                self.assertLess(state['returncode'], 0)
                self.assertIsNone(other.poll())
        finally:
            guard.stop_owned(other)

    def test_success_and_child_failure_are_recorded(self):
        for expected in [0, 3]:
            with self.subTest(code=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.object(guard, 'memory', return_value={'MemAvailable': 80 * guard.GIB, 'SwapFree': 0}):
                    actual = guard.main(['--output', str(root / 'run'), '--memory-gib', '40',
                                         '--lock', str(root / 'lock'), '--', sys.executable,
                                         '-c', f'print("finished"); raise SystemExit({expected})'])
                self.assertEqual(actual, expected)
                state = json.loads((root / 'run/status.json').read_text())
                self.assertEqual(state['state'], 'complete' if expected == 0 else 'failed')
                self.assertIn('finished', (root / 'run/run.log').read_text())


if __name__ == '__main__':
    unittest.main()
