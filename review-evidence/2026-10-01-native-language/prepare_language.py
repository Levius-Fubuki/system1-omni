import gc, hashlib, json, os, pathlib, sys, time
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from safetensors.torch import load_file
from models.cua_s1.multimodal.model import MultimodalEngine, BASE_REVISION, ADAPTER_REVISION
root=pathlib.Path('/root/cua-native-multimodal-20261001')
bundle=pathlib.Path('/root/autodl-tmp/cua-multimodal-fixtures/final-a')
weights=pathlib.Path('/root/autodl-tmp/system1-chain-validation-20260929/weights')
out=root/'weights-language-merged'
assert not out.exists()
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
manifest=json.loads((bundle/'manifest.json').read_text())
records=[]
engine=MultimodalEngine(str(weights/'Qwen3.5-4B'),str(weights/'cua-s1-4b-0.2/multimodal'))
merged=engine.model.merge_and_unload()
merged.model.language_model.save_pretrained(out,max_shard_size='5GB')
merged.config.to_json_file(out/'config.json')
(out/'cua_s1_language_export.json').write_text(json.dumps({'format':'cua-s1-multimodal-language-merged/1','base_revision':BASE_REVISION,'adapter_revision':ADAPTER_REVISION,'image_features':'external unmerged BF16 reference boundary'}))
print('MERGED_LANGUAGE_SAVED',flush=True)
with torch.no_grad():
    for entry in manifest['questions']:
        ts=load_file(str(bundle/entry['tensors_file']))
        ids=ts['input_ids'].cuda()
        embeds=merged.model.language_model.embed_tokens(ids)
        embeds[0,ts['image_token_indices']]=ts['image_features'].cuda()
        assert torch.equal(embeds.cpu(),ts['inputs_embeds'])
        hidden=merged.model.language_model(inputs_embeds=embeds,position_ids=ts['position_ids'].cuda(),use_cache=False).last_hidden_state[0,-1]
        rows=merged.get_output_embeddings().weight[ts['candidate_token_ids'].cuda()]
        logits=torch.nn.functional.linear(hidden,rows)
        probs=torch.softmax(logits.float(),dim=-1)
        records.append({'case':entry['case'],'question':entry['question'],'sequence':ids.shape[1],'input_embedding_equal':True,'bf16_reference_probabilities':ts['candidate_probabilities'].tolist(),'bf16_reference_logits':ts['candidate_logits'].float().tolist(),'merged_bf16_probabilities':probs.tolist(),'merged_bf16_logits':logits.float().tolist(),'merged_bf16_hidden':hidden.float().tolist()})
        print('MERGED_CONTROL',entry['case'],entry['question'],flush=True)
del engine,merged,hidden,rows,embeds,ids,probs,logits
gc.collect(); torch.cuda.empty_cache()
engine=MultimodalEngine(str(weights/'Qwen3.5-4B'),str(weights/'cua-s1-4b-0.2/multimodal'),dtype='float32')
model=engine.model.get_base_model()
with torch.no_grad():
    for record,entry in zip(records,manifest['questions']):
        ts=load_file(str(bundle/entry['tensors_file']))
        hidden=model.model.language_model(inputs_embeds=ts['inputs_embeds'].float().cuda(),position_ids=ts['position_ids'].cuda(),use_cache=False).last_hidden_state[0,-1]
        rows=model.get_output_embeddings().weight[ts['candidate_token_ids'].cuda()]
        logits=torch.nn.functional.linear(hidden,rows)
        record['fp32_unmerged_probabilities']=torch.softmax(logits,dim=-1).tolist()
        record['fp32_unmerged_logits']=logits.tolist()
        record['fp32_unmerged_hidden']=hidden.tolist()
        print('FP32_CONTROL',entry['case'],entry['question'],flush=True)
report={'tolerance':'max native probability error <= 2 * max BF16 reference error + 0.01; choice equal for FP32 top-two margin >= 0.05','scope':'language-only FP32 control at fixed BF16 image/input embedding boundary; not full FP32 vision','base_revision':BASE_REVISION,'adapter_revision':ADAPTER_REVISION,'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(),'tf32':False,'questions':records,'reference_manifest_sha256':hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest()}
(root/'controls.json').write_text(json.dumps(report,indent=2)+'\n')
print('CONTROLS_COMPLETE',flush=True)
