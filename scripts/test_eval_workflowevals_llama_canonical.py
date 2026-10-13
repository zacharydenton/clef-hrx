"""Guard input identity and keep frozen-runner resumes separate."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import eval_workflowevals_llama as frozen
import eval_workflowevals_llama_canonical as canonical


class CanonicalState(unittest.TestCase):
    def test_preserves_float_precision_unicode_and_schema_without_mutation(self):
        request = {'state': {'ratio': 2.901720335244817, 'amount': 100.0,
                             'currency': '€', 'nested': [False, None, -0.0]},
                   'questions': {'ok': {'type': 'noul', 'instructions': 'Pay?'}}}
        before = copy.deepcopy(request)
        actual = canonical.canonical_request(request)
        self.assertEqual(actual['state'],
                         '{"amount":100.0,"currency":"€","nested":[false,null,-0.0],"ratio":2.901720335244817}')
        self.assertEqual(json.loads(actual['state']), request['state'])
        self.assertEqual(actual['questions'], request['questions'])
        self.assertEqual(request, before)

    def test_text_is_not_serialized_twice_and_nonfinite_state_is_rejected(self):
        request = {'state': '{"ratio":1.0}'}
        self.assertIs(canonical.canonical_request(request), request)
        for number in [float('nan'), float('inf'), -float('inf')]:
            with self.assertRaises(ValueError):
                canonical.canonical_request({'state': {'value': number}})

    def test_both_sources_determine_resume_identity(self):
        digest = canonical.adapter_digest(b'base', b'canonical')
        self.assertNotEqual(digest, canonical.adapter_digest(b'changed', b'canonical'))
        self.assertNotEqual(digest, canonical.adapter_digest(b'base', b'changed'))

    def test_entrypoint_uses_canonical_transport_archives_source_and_restores_base(self):
        client, sha, write = frozen.LlamaClient, frozen.sha256, frozen.write_json
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'manifest.json'
            def run():
                self.assertNotEqual(frozen.sha256(frozen.__file__), sha(frozen.__file__))
                self.assertEqual(frozen.sha256(__file__), sha(__file__))
                with patch.object(client, 'request', return_value='response') as call:
                    instance = frozen.LlamaClient.__new__(frozen.LlamaClient)
                    self.assertEqual(instance.request({'state': {'a': 1.0}}), 'response')
                    call.assert_called_once_with({'state': '{"a":1.0}'})
                frozen.write_json(manifest, {'test': True})
                self.assertTrue(manifest.with_name('canonical-runner.py').is_file())
                return 75
            with patch.object(frozen, 'main', side_effect=run):
                self.assertEqual(canonical.main(), 75)
        self.assertIs(frozen.LlamaClient, client)
        self.assertIs(frozen.sha256, sha)
        self.assertIs(frozen.write_json, write)


if __name__ == '__main__':
    unittest.main()
