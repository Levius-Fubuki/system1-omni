"""Recompute language-boundary numerical gates from native/control JSON."""
import argparse, json, math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('folder',type=Path);args=p.parse_args()
root=args.folder
native=json.loads((root/'native.json').read_text())['questions']
control=json.loads((root/'controls.json').read_text())['questions']
assert len(native)==len(control)==8
err=lambda a,b:max(abs(x-y) for x,y in zip(a,b,strict=True))
reference_error=max(err(r['bf16_reference_probabilities'],r['fp32_unmerged_probabilities']) for r in control)
native_error=max(err(a['probabilities'],b['fp32_unmerged_probabilities']) for a,b in zip(native,control,strict=True))
allowance=2*reference_error+0.01
assert native_error<=allowance
rows=[]
for a,b in zip(native,control,strict=True):
    assert (a['case'],a['question'],a['sequence'])==(b['case'],b['question'],b['sequence'])
    assert a['repeat_equal'] and b['input_embedding_equal']
    for name in ['last_hidden_state','candidate_logits','probabilities']:
        assert all(math.isfinite(x) for x in a[name])
    assert len(a['last_hidden_state'])==len(b['merged_bf16_hidden'])==2560
    assert len(a['probabilities'])==len(a['candidate_logits'])==len(b['fp32_unmerged_probabilities'])
    assert all(0<=p<=1 for p in a['probabilities']) and abs(sum(a['probabilities'])-1)<1e-6
    m=max(a['candidate_logits']);es=[math.exp(x-m) for x in a['candidate_logits']];total=sum(es)
    assert err(a['probabilities'],[x/total for x in es])<1e-7
    expected=sorted(range(len(b['fp32_unmerged_probabilities'])),key=lambda i:b['fp32_unmerged_probabilities'][i],reverse=True)
    margin=1 if len(expected)==1 else b['fp32_unmerged_probabilities'][expected[0]]-b['fp32_unmerged_probabilities'][expected[1]]
    top=max(range(len(a['probabilities'])),key=lambda i:a['probabilities'][i])
    assert margin<0.05 or top==expected[0]
    rows.append({'case':a['case'],'question':a['question'],'sequence':a['sequence'],'image_tokens':a['image_tokens'],'probability_error_vs_fp32':err(a['probabilities'],b['fp32_unmerged_probabilities']),'top_equal':top==expected[0],'fp32_margin':margin,'max_hidden_error_vs_merged_bf16':err(a['last_hidden_state'],b['merged_bf16_hidden']),'max_logit_error_vs_merged_bf16':err(a['candidate_logits'],b['merged_bf16_logits'])})
report={'status':'pass','questions':8,'native_forwards':16,'bf16_reference_max_probability_error':reference_error,'native_max_probability_error':native_error,'probability_allowance':allowance,'all_top_choices_equal':all(r['top_equal'] for r in rows),'cases':rows,'scope':'native language at fixed BF16 reference vision boundary; not native image processing or vision execution'}
(root/'verified-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
