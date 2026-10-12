"""CPU checks for memory admission, server ownership and scoring contracts."""
import copy
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

from eval_workflowevals_llama import (
    GIB, check_pilot, kill_owned, main, memory_pressure, required_memory,
    server_command, speed_summary, validated_response,
)


class MemorySafety(unittest.TestCase):
    def test_reserve_is_additional_to_model_and_workspace(self):
        self.assertEqual(required_memory(50 * GIB, 32768, 16 * GIB), 78 * GIB)
        self.assertEqual(required_memory(50 * GIB, 65536, 16 * GIB), 90 * GIB)
        # Measured scratch grows conservatively with context squared.
        self.assertEqual(required_memory(50 * GIB, 65536, 16 * GIB, 60 * GIB), 106 * GIB)

    def test_low_memory_and_new_swap_both_trip_guard(self):
        before = {'MemAvailable': 90 * GIB, 'SwapFree': 60 * GIB}
        self.assertIsNone(memory_pressure(before, before, 16 * GIB))
        self.assertIsNotNone(memory_pressure(before, {**before, 'MemAvailable': 15 * GIB}, 16 * GIB))
        self.assertIsNotNone(memory_pressure(before, {**before, 'SwapFree': 58 * GIB}, 16 * GIB))

    @patch('eval_workflowevals_llama.os.killpg')
    def test_guard_targets_only_owned_live_process_group(self, kill):
        process = Mock(pid=12345)
        process.poll.return_value = None
        kill_owned(process, signal.SIGKILL)
        kill.assert_called_once_with(12345, signal.SIGKILL)
        kill.reset_mock()
        process.poll.return_value = 0
        kill_owned(process, signal.SIGKILL)
        kill.assert_not_called()

    @patch('eval_workflowevals_llama.subprocess.Popen')
    @patch('eval_workflowevals_llama.memory', return_value={'MemAvailable': GIB, 'SwapFree': 60 * GIB})
    def test_low_memory_refuses_before_starting_or_creating_run(self, mem, popen):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model.gguf'
            model.write_bytes(b'fixture')
            output = root / 'results'
            args = ['runner', '--binary', 'not-started', '--llama-source', 'unused',
                    '--model', str(model), '--manifest', 'unused', '--output', str(output),
                    '--context', '32768', '--smoke-request', 'unused', '--gpu-lock', str(root / 'gpu.lock')]
            with patch('sys.argv', args):
                self.assertEqual(main(), 75)
            popen.assert_not_called()
            self.assertFalse(output.exists())

    def test_single_slot_and_entire_request_microbatch(self):
        command = server_command(Path('server'), Path('model'), 65536, 8000)
        self.assertEqual(command[command.index('-np') + 1], '1')
        for flag in ('-c', '-b', '-ub'):
            self.assertEqual(command[command.index(flag) + 1], '65536')
        self.assertEqual(command[command.index('--host') + 1], '127.0.0.1')

    def test_64k_requires_successful_matching_long_invoice_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = dict(binary_sha256='binary', model_sha256='weights', harness_sha256='runner', context=32768)
            records = {'manifest.json': {'configuration': config},
                       'process.json': {'replay_exact': True, 'peak_available_memory_drop_bytes': 60 * GIB},
                       'summary.json': {'smoke': True, 'memory_guard_triggered': False,
                                        'transport_errors': 0, 'speed': {'calls': 3}}}
            for name, value in records.items():
                (output / name).write_text(json.dumps(value))
            (output / 'llama-calls.jsonl').write_text((json.dumps({'input_tokens': 30342}) + '\n') * 3)
            self.assertEqual(check_pilot(output, {**config, 'context': 65536}), 60 * GIB)
            with self.assertRaises(ValueError):
                check_pilot(output, {**config, 'model_sha256': 'different'})
            records['summary.json']['memory_guard_triggered'] = True
            (output / 'summary.json').write_text(json.dumps(records['summary.json']))
            with self.assertRaises(ValueError):
                check_pilot(output, config)


class ResponseContract(unittest.TestCase):
    def setUp(self):
        self.request = {'questions': {
            'route': {'type': 'choice', 'criteria': {'a': 'A', 'b': 'B'}},
            'severity': {'type': 'score', 'criteria': ['low', 'medium', 'high']},
            'truth': {'type': 'noul'},
        }}
        self.raw = {'model': 'clef', 'usage': {'input_tokens': 30342, 'output_tokens': 0}, 'answers': {
            'route': {'type': 'choice', 'choice': 'a', 'confidence': 0.75, 'probabilities': {'a': 0.75, 'b': 0.25}},
            'severity': {'type': 'score', 'score': 1.6, 'confidence': 0.7,
                         'probabilities': {'0': 0.1, '1': 0.2, '2': 0.7},
                         'legend': {'0': 'low', '1': 'medium', '2': 'high'}},
            'truth': {'type': 'noul', 'noul': 0.123456789},
        }}

    def test_preserves_native_probabilities_and_expectation(self):
        result = validated_response(self.request, self.raw, 30342)
        self.assertEqual(result.answers['truth'].noul, 0.123456789)
        self.assertEqual(result.answers['severity'].score, 1.6)
        self.assertEqual(result.answers['severity'].probabilities, {0: 0.1, 1: 0.2, 2: 0.7})

    def test_rejects_truncation_invalid_probabilities_and_schema_changes(self):
        for mutate in (
            lambda r: r['usage'].update(input_tokens=30000),
            lambda r: r['usage'].update(output_tokens=1),
            lambda r: r['answers'].pop('truth'),
            lambda r: r['answers']['route'].update(choice='b'),
            lambda r: r['answers']['route']['probabilities'].update(a=0.9),
            lambda r: r['answers']['truth'].update(noul=float('nan')),
            lambda r: r['answers']['severity'].update(score=2),
        ):
            raw = copy.deepcopy(self.raw)
            mutate(raw)
            with self.assertRaises(ValueError):
                validated_response(self.request, raw, 30342)

    def test_throughput_excludes_first_request_of_each_server_session(self):
        rows = [{'first_request': first, 'input_tokens': tokens, 'wall_seconds': seconds}
                for first, tokens, seconds in [(True, 999, 100), (False, 100, 1),
                                               (True, 999, 100), (False, 900, 3)]]
        result = speed_summary(rows)
        self.assertEqual(result['resident_input_tokens_per_second'], 250)
        self.assertEqual(result['resident_calls'], 2)


if __name__ == '__main__':
    unittest.main()
