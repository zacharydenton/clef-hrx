import json
from pathlib import Path
import tempfile
import unittest
from compare_workflowevals import compare

class WorkflowGates(unittest.TestCase):
    def fixture(self, path, scores=(0.5,0.7), precision='bf16', complete=True):
        path.mkdir()
        manifest={'workflow_commit':'pin','dependency_lock_sha256':'lock','datasets':{},
                  'configuration':{'model':'clef','checkpoint_revision':'pin','precision':precision,'max_length':65536}}
        report={'complete':complete,'limited_smoke_run':False,'speed':{},'workflows':{}}
        for name,cases,value in zip(['invoice','other'],[150,555],scores):
            report['workflows'][name]={'cases':cases,'planned_cases':cases,'failed_cases':0,
                'scores':[{'label_set_id':'labels-consensus','metrics':{'exact_actions':value},
                           'metric_denominators':{'exact_actions':cases}}]}
        (path/'native-manifest.json').write_text(json.dumps(manifest))
        (path/'summary.json').write_text(json.dumps(report))
        return path
    def test_per_workflow_regression_is_not_hidden_by_other_gain(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            old=self.fixture(root/'old')
            new=self.fixture(root/'new',(0.49,0.99))
            self.assertFalse(compare(old,new)['passed'])
            self.assertTrue(compare(old,old)['passed'])
    def test_partial_and_cross_precision_runs_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=self.fixture(root/'old')
            for name,options in [('partial',{'complete':False}),('int4',{'precision':'int4'})]:
                with self.assertRaises(ValueError): compare(old,self.fixture(root/name,**options))

if __name__=='__main__': unittest.main()
