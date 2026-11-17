"""Evaluate the compact PU GraphSAGE and HGT models on the V15 protocol.

The protocol (candidate rows, outer splits, and fit/calibration partitions) is
identical to the two-tower experiment. The graph models receive taxonomic node
features and week context; they do not receive exact species identity.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import run_factorized_ai_experiment_v15 as v15
import run_plant_bee_gnn_benchmark_v1 as gnn

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'interim'/'factorized_model_v15'/'factorized_event_records.csv.gz'
OUT=ROOT/'results'/'phase1_shared_gnns_v1'
KEEP={'pu_graphsage','pu_hgt'}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(DATA,parse_dates=['date']); rows=[]; predictions=[]; definitions=v15.splits(data)
    for index,(name,train,test) in enumerate(definitions,1):
        print(name,flush=True)
        all_rows, all_predictions=gnn.evaluate(name,train.copy(),test.copy())
        rows += [row for row in all_rows if row['model'] in KEEP]
        columns=['split','event_id','study','site','year','week','bee','plant','interaction_observed','p_pu_graphsage','p_pu_hgt']
        predictions.append(all_predictions[columns])
        pd.DataFrame(rows).to_csv(OUT/'metrics_partial.csv',index=False)
        pd.concat(predictions,ignore_index=True).to_csv(OUT/'predictions_partial.csv.gz',index=False,compression='gzip')
        (OUT/'progress.json').write_text(json.dumps({'completed_split':name,'completed_splits':index,'total_splits':len(definitions)},indent=2),encoding='utf8')
    metrics=pd.DataFrame(rows)
    fold=metrics[metrics.split.str.startswith('unseen_study_fold_')]
    macro=fold.groupby('model',as_index=False)[['average_precision','roc_auc','brier']].mean()
    macro.insert(0,'split','unseen_study_macro_average'); macro['rows']=int(fold.rows.sum()/len(KEEP)); macro['events']=int(fold.events.sum()/len(KEEP)); macro['positive_rate']=fold.positive_rate.mean()
    metrics=pd.concat([metrics,macro[metrics.columns]],ignore_index=True)
    metrics.to_csv(OUT/'metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(OUT/'predictions.csv.gz',index=False,compression='gzip')
    (OUT/'summary.json').write_text(json.dumps({'data':'factorized_model_v15','splits':'exact V15 outer splits and event calibration partitions','models':sorted(KEEP),'label':'recorded interaction conditional on same-event co-occurrence; non-detections receive weak PU loss, not biological-negative labels'},indent=2),encoding='utf8')
if __name__=='__main__': main()
