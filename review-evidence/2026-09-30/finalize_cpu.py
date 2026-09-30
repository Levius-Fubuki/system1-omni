"""Remove duplicate carried regression files and retain the authoritative suite logs."""
import json
import os
import pathlib
import subprocess
import time

root = pathlib.Path('/root/cua-review-20260930')
py = '/root/autodl-tmp/system1-chain-validation-20260929/venv/bin/python'
records = json.loads((root/'results/checks.json').read_text())
manifest = json.loads((root/'source-manifest.json').read_text())
for n in (33,37,38):
    repo = root/f'pr{n}'
    duplicate = repo/'tests/cua_s1/test_review_graph_lifetime.py'
    existing = repo/'tests/cua_s1/test_graph_capture_stream.py'
    assert duplicate.read_bytes() == existing.read_bytes()
    duplicate.unlink()
    subprocess.run(['git','add','tests/cua_s1/test_review_graph_lifetime.py'],cwd=repo,check=True)
    subprocess.run(['git','commit','-m','Deduplicate existing graph stream ownership regressions'],cwd=repo,check=True)
    out = root/'results'/f'pr{n}'/'cpu-tests.log'
    out.rename(out.with_name('cpu-tests-with-duplicate-regressions.log'))
    command = [py,'-m','pytest','tests/cua_s1','-q']
    start = time.time()
    with out.open('w') as log:
        result = subprocess.run(command,cwd=repo,env=dict(os.environ,PYTHONPATH='src:recipe/cua_s1'),stdout=log,stderr=subprocess.STDOUT)
    records.append({'pr':n,'name':'cpu-tests','command':command,'exit':result.returncode,'seconds':round(time.time()-start,3)})
    (root/'results/checks.json').write_text(json.dumps(records,indent=2)+'\n')
    assert result.returncode == 0
    manifest[str(n)]['validation_revision'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    print(n,out.read_text().splitlines()[-1],flush=True)
(root/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
