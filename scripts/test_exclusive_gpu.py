import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import exclusive_gpu


class ExclusiveGpuTests(unittest.TestCase):
    def test_stopping_benchmark_does_not_stop_other_group(self):
        command = [sys.executable, '-c', 'import time; time.sleep(30)']
        owned = subprocess.Popen(command, start_new_session=True)
        other = subprocess.Popen(command, start_new_session=True)
        try:
            self.assertEqual(exclusive_gpu.foreign_clients([owned.pid, other.pid], owned.pid),
                             [other.pid])
            exclusive_gpu.stop_owned_group(owned)
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(other.poll())
        finally:
            for process in (owned, other):
                if process.poll() is None:
                    exclusive_gpu.stop_owned_group(process)

    def test_busy_preflight_does_not_launch_command(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'guard.jsonl'
            with patch.object(sys, 'argv', ['exclusive_gpu', '--log', str(log), '--', 'unused']), \
                    patch.object(exclusive_gpu, 'clients', return_value=[123]), \
                    patch.object(exclusive_gpu.subprocess, 'Popen') as launch:
                with self.assertRaises(SystemExit) as result:
                    exclusive_gpu.main()
                self.assertEqual(result.exception.code, 75)
                launch.assert_not_called()
            self.assertEqual(json.loads(log.read_text())['kfd_pids'], [123])

    def test_vanished_client_is_ignored_but_unreadable_client_is_foreign(self):
        with patch.object(exclusive_gpu.os, 'getpgid',
                          side_effect=[ProcessLookupError, PermissionError]):
            self.assertEqual(exclusive_gpu.foreign_clients([10, 11], 12), [11])


if __name__ == '__main__':
    unittest.main()
