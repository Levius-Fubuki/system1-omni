import io
import json
import os
import pathlib
import subprocess
import tarfile
import time

root = pathlib.Path('/root/cua-review-20260930')
repo = root/'repository'
py = '/root/autodl-tmp/system1-chain-validation-20260929/venv/bin/python'
weights = '/root/autodl-tmp/system1-chain-validation-20260929/weights'
sources = {17: 'codex/core-only-pr17', 18: 'codex/core-only-pr18', 22: 'codex/cua-review-gpu-fixes',
           33: 'codex/review-gpu-pr33', 37: 'codex/review-gpu-pr37', 38: 'codex/review-gpu-pr38'}
supports = {n: f'codex/review-layout-pr{n}' for n in sources}

def git(*args, cwd=repo):
    return subprocess.check_output(['git', *args], cwd=cwd)

if not repo.exists():
    subprocess.run(['git', 'clone', '-b', sources[38], str(root/'review-final-source.bundle'), str(repo)], check=True)
git('config', 'user.name', 'Levius-Fubuki')
git('config', 'user.email', 'Levius-Fubuki@users.noreply.github.com')
manifest = {}
for n, branch in sources.items():
    target = root/f'pr{n}'
    ref = 'origin/'+branch
    if target.exists():
        manifest[str(n)] = {'runtime_revision': git('rev-parse', ref).decode().strip(),
                            'validation_revision': git('rev-parse', 'HEAD', cwd=target).decode().strip(),
                            'support_revision': git('rev-parse', 'origin/'+supports[n]).decode().strip()}
        assert not git('diff', ref, 'HEAD', '--', 'src', cwd=target)
        continue
    git('worktree', 'add', '-b', f'codex/validation-pr{n}-20260930', str(target), ref)
    archive = git('archive', 'origin/'+supports[n], 'tests/cua_s1', 'recipe/cua_s1')
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(target, filter='data')
    if n >= 22:
        (target/'tests/cua_s1/test_review_graph_lifetime.py').write_bytes((root/'test_review_graph_lifetime.py').read_bytes())
        (target/'recipe/cua_s1/check_review_eviction.py').write_bytes((root/'check_review_eviction.py').read_bytes())
        for name in ['diagnose_graph_buckets.py', 'check_graph_capture_stream.py', 'check_graph_pool.py']:
            (target/'recipe/cua_s1'/name).write_bytes(git('show', 'origin/'+supports[38]+':recipe/cua_s1/'+name))
    git('add', 'tests', 'recipe', cwd=target)
    git('commit', '-m', f'Restore archived validation support for PR {n} GPU review', cwd=target)
    runtime_sha = git('rev-parse', ref).decode().strip()
    assert not git('diff', runtime_sha, 'HEAD', '--', 'src', cwd=target)
    manifest[str(n)] = {'runtime_revision': runtime_sha, 'validation_revision': git('rev-parse', 'HEAD', cwd=target).decode().strip(),
                        'support_revision': git('rev-parse', 'origin/'+supports[n]).decode().strip()}
(root/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
env = dict(os.environ, PYTHONPATH='src:recipe/cua_s1', PYTHONUNBUFFERED='1',
           HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
records = json.loads((root/'results/checks.json').read_text()) if (root/'results/checks.json').exists() else []
steps = []
for n in sources:
    steps.append((n, 'cpu-tests', [py, '-m', 'pytest', 'tests/cua_s1', '-q']))

def add(n, name, script, *args, model=True):
    out = root/'results'/f'pr{n}'
    out.mkdir(parents=True, exist_ok=True)
    command = [py, 'recipe/cua_s1/'+script]
    if model:
        command += ['--weights', weights]
    command += ['--output', str(out/name), *args]
    steps.append((n, name, command))

add(22, 'streams.json', 'check_graph_capture_stream.py', model=False)
add(22, 'pools.json', 'check_graph_pool.py', model=False)
add(22, 'eviction.json', 'check_review_eviction.py')
add(33, 'eviction.json', 'check_review_eviction.py')
add(37, 'eviction.json', 'check_review_eviction.py')
add(38, 'shared-survivor.json', 'check_review_eviction.py', '--shared')
add(17, 'image-reuse', 'benchmark_image_reuse.py', '--correctness-only')
add(18, 'last-logits', 'benchmark_last_logits.py', '--warmup', '2', '--runs', '2', '--iterations', '10',
    '--case', '320x240-short-q2', '--case', '640x480-distinct-q8')
add(22, 'graph-matrix', 'benchmark_graph_runtime.py', '--warmup', '2', '--runs', '2', '--iterations', '10',
    '--graph-max-tokens', '4096', '--case', '320x240-short-q2', '--case', '320x240-short-q8',
    '--case', '640x480-short-q8', '--case', '640x480-long-q8', '--case', '640x480-distinct-q8')
add(22, 'cache.json', 'check_graph_cache.py')
add(33, 'admission-parity.json', 'check_graph_admission_runtime.py')
add(37, 'bucket-parity.json', 'check_graph_buckets.py', '--worker')
add(37, 'boundaries.json', 'check_graph_buckets.py', '--worker', '--boundaries-only')
add(37, 'http.json', 'check_bucket_worker_http.py')
add(37, 'bucket-matrix', 'benchmark_graph_buckets.py', '--kind', 'worker', '--runs', '2',
    '--case', 'hot_four', '--case', 'churn_twelve')
add(38, 'auto-parity.json', 'check_auto_worker.py')
add(38, 'shared.json', 'check_shared_graph_policy.py')
add(38, 'http.json', 'check_bucket_worker_http.py', '--mode', 'auto')
add(38, 'auto-matrix', 'benchmark_graph_buckets.py', '--kind', 'worker', '--tuned-exact-window', '32',
    '--include-auto', '--runs', '2', '--case', 'hot_four', '--case', 'churn_twelve', '--case', 'hot_cold',
    '--case', 'shifting_hot', '--case', 'mixed_holdout')
for n, name, command in steps:
    if any(r['pr'] == n and r['name'] == name and r['exit'] == 0 for r in records):
        continue
    out = root/'results'/f'pr{n}'
    out.mkdir(parents=True, exist_ok=True)
    for old in [out/name, out/(name+'.log')]:
        if old.exists():
            old.rename(old.with_name(old.name+'.attempt-'+str(len(records))))
    print('START', n, name, flush=True)
    start = time.time()
    with (out/(name+'.log')).open('w') as log:
        result = subprocess.run(command, cwd=root/f'pr{n}', env=env, stdout=log, stderr=subprocess.STDOUT)
    records.append({'pr': n, 'name': name, 'command': command, 'exit': result.returncode, 'seconds': round(time.time()-start, 3)})
    (root/'results/checks.json').write_text(json.dumps(records, indent=2)+'\n')
    print('END', n, name, result.returncode, records[-1]['seconds'], flush=True)
    if result.returncode:
        print('HALTED: failed step', flush=True)
        raise SystemExit(1)
print('COMPLETE', flush=True)
