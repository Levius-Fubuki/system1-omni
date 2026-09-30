"""Full-checkpoint graph eviction, surviving replay and retained-pool accounting."""
import argparse
import hashlib
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shared', action='store_true')
    args = p.parse_args()
    assert not args.output.exists()
    import torch
    from diagnose_graph_buckets import prepare_values
    from evaluate_multimodal import environment
    from profile_multimodal import GOAL, case_matrix, fixture, repository_state, write_json
    from models.cua_s1.multimodal.graph_runtime import GraphConfig, GraphRuntime
    from models.cua_s1.multimodal.model import MultimodalEngine
    from models.cua_s1.multimodal.protocol import parse_request

    root = Path(__file__).resolve().parents[2]
    source = repository_state(root)
    assert not source['dirty']
    report = {'status': 'running', 'repository': source, 'environment': environment(),
              'source_sha256': {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                                for f in (root/'src/models/cua_s1/multimodal').glob('*.py')},
              'shared': args.shared, 'checks': []}
    engine = MultimodalEngine(str(args.weights/'Qwen3.5-4B'), str(args.weights/'cua-s1-4b-0.2/multimodal'))
    config = GraphConfig(min_uses=1, max_shapes=2, max_bytes=4 << 30, max_tokens=4096)
    if hasattr(config, 'max_captures'):
        config = replace(config, max_captures=32, capture_budget_ms=20000)
    if args.shared:
        from models.cua_s1.multimodal.graph_shared import SharedGraphRuntime
        runtime = SharedGraphRuntime(engine.model, config)
    else:
        runtime = GraphRuntime(engine.model, config)

    def check(values, mode='exact'):
        if args.shared:
            runtime.select_mode(mode)
        expected = engine.model(**values, logits_to_keep=1, use_cache=False).logits[0, -1, :]
        before = dict(runtime.stats)
        with runtime.request() if hasattr(runtime, 'request') else nullcontext():
            actual = runtime.forward(values)
        equal = torch.equal(expected, actual)
        report['checks'].append({'mode': mode, 'tokens': values['inputs_embeds'].shape[1],
                                 'equal': equal, 'max_abs': (expected.float()-actual.float()).abs().max().item(),
                                 'stats_delta': {k: v-before[k] for k,v in runtime.stats.items()}})
        assert equal, report['checks'][-1]
        return report['checks'][-1]

    with torch.no_grad():
        case = next(c for c in case_matrix() if c['id'] == '320x240-short-q2')
        request = parse_request(fixture(case, args.output.parent/'eviction-fixtures'))
        request = replace(request, questions=tuple(replace(q, goal=' '.join([GOAL]*30)) for q in request.questions))
        prepared = prepare_values(engine, request)
        values = {n: {'inputs_embeds': prepared['inputs_embeds'][:, :n].clone(),
                      'position_ids': prepared['position_ids'][:, :, :n].clone(),
                      'attention_mask': prepared['attention_mask'][:, :n].clone()} for n in (255, 319, 385)}
        assert check(values[255])['stats_delta']['captures'] == 1
        survivor = list(runtime.cache.entries.values())[0]
        mode = 'rule-bucket' if args.shared else 'exact'
        assert check(values[319], mode)['stats_delta']['captures'] == 1
        victim = list(runtime.cache.entries.values())[-1]
        victim_pool = victim.pool.pool_id
        assert check(values[255])['stats_delta']['replays'] == 1
        assert check(values[385], mode)['stats_delta']['evictions'] == 1
        assert victim.pool is None and not victim.blocks
        assert survivor in runtime.cache.entries.values()
        for offset in (0.0, 0.01, -0.02):
            changed = dict(values[255], inputs_embeds=values[255]['inputs_embeds'] + offset)
            assert check(changed)['stats_delta']['replays'] == 1
        pools = []
        for entry in runtime.cache.entries.values():
            snapshot = torch.cuda.memory_snapshot(mempool_id=entry.pool.pool_id, include_traces=False)
            reserved = sum(s['total_size'] for s in snapshot)
            allocated = sum(s['allocated_size'] for s in snapshot)
            external = sum(b.external_bytes for b in entry.blocks.values())
            assert entry.bytes == reserved + external
            pools.append({'reserved': reserved, 'allocated': allocated, 'external': external, 'accounted': entry.bytes})
        assert any(p['reserved'] > p['allocated'] for p in pools)
        torch.cuda.empty_cache()
        assert not torch.cuda.memory_snapshot(mempool_id=victim_pool, include_traces=False)
        retained = list(runtime.cache.entries.values())
        runtime.invalidate()
        assert all(e.pool is None and not e.blocks for e in retained)
        report.update(status='complete', stats=dict(runtime.stats), pools=pools,
                      victim_released=True, surviving_graph_replays_after_eviction=3,
                      referenced_entries_explicitly_retired=True)
    write_json(args.output, report)
    print('full-checkpoint eviction and three surviving replays verified', flush=True)


if __name__ == '__main__':
    main()
