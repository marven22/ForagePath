"""Feature-matched rich GraphSAGE/HGT ablation on the V15 two-tower protocol."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_factorized_ai_experiment_v15 as v15
import run_plant_bee_gnn_benchmark_v1 as g

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'interim'/'factorized_model_v15'/'factorized_event_records.csv.gz'; OUT=ROOT/'results'/'phase1_rich_gnns_v1'

def safe_numeric(value):
    """Encode unavailable trait/coverage fields as zero, never NaN."""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(numeric) or not np.isfinite(numeric) else float(numeric)

def rich_nodes(frames):
    d=pd.concat(frames,ignore_index=True); bees=sorted(d.bee.unique()); plants=sorted(d.plant.unique()); bi={x:i for i,x in enumerate(bees)}; pi={x:i for i,x in enumerate(plants)}
    # Same static traits and missingness supplied to the two-tower model, packed
    # into a 130-dim inductive node vector; no exact species identity is used.
    x=np.zeros((len(bees)+len(plants),130),dtype=np.float32); types=np.zeros(len(x),dtype=np.int64); x[:len(bees),0]=1; x[len(bees):,1]=1; types[len(bees):]=1
    bm=d.drop_duplicates('bee').set_index('bee'); pm=d.drop_duplicates('plant').set_index('plant')
    for name,i in bi.items():
        row=bm.loc[name]
        for field in v15.BEE_CATS:
            x[i,2+v15.hash_index(field+str(row.get(field,'unknown')),96)]=1
        for j,field in enumerate(v15.BEE_FLAGS): x[i,98+j]=safe_numeric(row.get(field,0))
    # Some held-out folds contain a trait column with fewer than two observed
    # values.  Pandas then reports an undefined standard deviation.  Treating
    # that feature as unscaled (scale=1) is the mathematically conservative
    # choice; allowing NaN through would make the neural logits undefined.
    means, scales = {}, {}
    for f in v15.PLANT_TRAITS:
        values = pd.to_numeric(pm[f], errors='coerce')
        mean, scale = values.mean(), values.std()
        means[f] = 0.0 if pd.isna(mean) else float(mean)
        scales[f] = 1.0 if pd.isna(scale) or scale < 1e-6 else float(scale)
    for name,i in pi.items():
        node=len(bees)+i; row=pm.loc[name]
        for field in v15.PLANT_CATS: x[node,2+v15.hash_index(field+str(row.get(field,'unknown')),96)]=1
        for j,field in enumerate(v15.PLANT_TRAITS):
            value=pd.to_numeric(pd.Series([row.get(field)]),errors='coerce').iloc[0]; missing=pd.isna(value) or not np.isfinite(value); x[node,101+j]=(0 if missing else (np.log1p(max(value,0))-np.log1p(max(means[field],0)))/scales[field]); x[node,111+j]=float(missing)
        for j,field in enumerate(v15.PLANT_FLAGS): x[node,121+j]=safe_numeric(row.get(field,0))
    if not np.isfinite(x).all():
        raise FloatingPointError("Rich node construction produced a non-finite input feature")
    return bi,pi,torch.tensor(x),torch.tensor(types)

def main():
    OUT.mkdir(parents=True,exist_ok=True); g.make_node_features=rich_nodes
    # The trait-augmented HGT has larger, denser node inputs than the compact
    # model.  Conservative optimization prevents attention-logit divergence.
    g.GNN_LEARNING_RATE=.001; g.GNN_GRADIENT_CLIP=1.0
    data=pd.read_csv(DATA,parse_dates=['date']); rows=[]; parts=[]; defs=v15.splits(data)
    for n,(name,train,test) in enumerate(defs,1):
        print(name,flush=True); all_rows,pred=g.evaluate(name,train.copy(),test.copy()); rows += [r for r in all_rows if r['model'] in {'pu_graphsage','pu_hgt'}]; parts.append(pred[['split','event_id','study','interaction_observed','p_pu_graphsage','p_pu_hgt']]); pd.DataFrame(rows).to_csv(OUT/'metrics_partial.csv',index=False); (OUT/'progress.json').write_text(json.dumps({'completed_split':name,'completed_splits':n,'total_splits':len(defs)},indent=2),encoding='utf8')
    m=pd.DataFrame(rows); folds=m[m.split.str.startswith('unseen_study_fold_')]; macro=folds.groupby('model',as_index=False)[['average_precision','roc_auc','brier']].mean(); macro.insert(0,'split','unseen_study_macro_average'); macro['rows']=int(folds.rows.sum()/2); macro['events']=int(folds.events.sum()/2); macro['positive_rate']=folds.positive_rate.mean(); m=pd.concat([m,macro[m.columns]],ignore_index=True); m.to_csv(OUT/'metrics.csv',index=False); pd.concat(parts,ignore_index=True).to_csv(OUT/'predictions.csv.gz',index=False,compression='gzip')
if __name__=='__main__':main()
