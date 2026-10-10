"""Contract checks for the local adapter used by the upstream workflow scorer."""
import copy
import unittest
from eval_workflowevals import response_from_raw


class ResponseContract(unittest.TestCase):
    def setUp(self):
        self.request = {'questions': {
            'truth': {'type': 'noul', 'criteria': None},
            'route': {'type': 'choice', 'criteria': {'a': 'A', 'b': 'B'}},
            'severity': {'type': 'score', 'criteria': ['low', 'medium', 'high']},
        }}
        self.raw = {'input_tokens': 123, 'questions': [
            {'question_id': 'truth', 'option_ids': ['true', 'false'], 'probabilities': [0.1234567, 0.8765433]},
            {'question_id': 'route', 'option_ids': ['a', 'b'], 'probabilities': [0.75, 0.25]},
            {'question_id': 'severity', 'option_ids': ['0', '1', '2'], 'probabilities': [0.1, 0.2, 0.7]},
        ]}

    def test_probabilities_and_expected_score_preserved(self):
        result = response_from_raw(self.request, self.raw)
        self.assertEqual(result.answers['truth'].noul, 0.1234567)
        self.assertEqual(result.answers['route'].choice, 'a')
        self.assertEqual(result.answers['severity'].probabilities, {0: 0.1, 1: 0.2, 2: 0.7})
        self.assertAlmostEqual(result.answers['severity'].score, 1.6)
        self.assertEqual(result.usage.input_tokens, 123)
        self.assertEqual(result.usage.output_tokens, 0)

    def test_invalid_probabilities_rejected_without_normalization(self):
        for values in ([0.7, 0.7], [float('nan'), 1.0], [-0.1, 1.1]):
            raw = copy.deepcopy(self.raw)
            raw['questions'][1]['probabilities'] = values
            with self.assertRaises(ValueError):
                response_from_raw(self.request, raw)

    def test_schema_mismatch_rejected(self):
        for mutate in (
            lambda q: q.pop(),
            lambda q: q.append(q[0]),
            lambda q: q[1].update(option_ids=['a', 'c']),
            lambda q: q[1].update(option_ids=['a', 'a']),
        ):
            raw = copy.deepcopy(self.raw)
            mutate(raw['questions'])
            with self.assertRaises(ValueError):
                response_from_raw(self.request, raw)


if __name__ == '__main__':
    unittest.main()
