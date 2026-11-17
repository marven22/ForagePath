"""Run strict, heuristic, and CatBoost baselines on the two-tower V15 protocol."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
import run_factorized_ai_experiment_v15 as v15

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'interim'/'factorized_model_v15'/'factorized_event_records.csv.gz'
OUT=ROOT/'results'/'phase1_shared_baselines_v1'
SEED=20260819
CATS=v15.BEE_CATS+v15.PLANT_CATS+['sampling_grain']
NUM=v15.PLANT_TRAITS+v15.BEE_FLAGS+v15.PLANT_FLAGS+['week_sin','week_cos','log_event_candidate_pairs']

def calibrate(y,a,b):
    if len(np.unique(y))<2: return 1/(1+np.exp(-b))
    m=LogisticRegression(C=1,solver='lbfgs',random_state=SEED).fit(a.reshape(-1,1),y)
    return m.predict_proba(b.reshape(-1,1))[:,1]

def feature(frame):
    x=frame[CATS+NUM].copy()
    for c in CATS:x[c]=x[c].fillna('unknown').astype(str)
    for c in NUM:x[c]=pd.to_numeric(x[c],errors='coerce').fillna(0)
    return x

def score(split,model,test,p):
    y=test.interaction_observed.to_numpy(int)
    return {'split':split,'model':model,'rows':len(test),'events':int(test.event_id.nunique()),'positive_rate':float(y.mean()),'average_precision':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p))}

def run(split,outer,test):
    internal=outer.event_id.map(lambda x:v15.bucket(str(x),7)); cal=outer[internal.eq(1)]; fit=outer[~internal.isin([0,1])]
    prior=float(fit.interaction_observed.mean()); positive=set(map(tuple,fit.loc[fit.interaction_observed.eq(1),['bee','plant']].to_numpy()))
    strict=np.array([.95 if (b,p) in positive else .05 for b,p in zip(test.bee,test.plant)])
    br=(fit.groupby('bee').interaction_observed.mean()+prior).to_dict(); pr=(fit.groupby('plant').interaction_observed.mean()+prior).to_dict()
    def degree(f):return np.array([np.sqrt(br.get(b,prior)*pr.get(p,prior)) for b,p in zip(f.bee,f.plant)])
    degree_p=calibrate(cal.interaction_observed.to_numpy(int),degree(cal),degree(test))
    model=CatBoostClassifier(iterations=300,depth=7,learning_rate=.04,loss_function='Logloss',verbose=False,random_seed=SEED,l2_leaf_reg=8,allow_writing_files=False)
    model.fit(feature(fit),fit.interaction_observed,cat_features=CATS)
    cp=model.predict_proba(feature(cal))[:,1]; tp=model.predict_proba(feature(test))[:,1]
    cat=calibrate(cal.interaction_observed.to_numpy(int),np.log(np.clip(cp,1e-5,1-1e-5)/(1-np.clip(cp,1e-5,1-1e-5))),np.log(np.clip(tp,1e-5,1-1e-5)/(1-np.clip(tp,1e-5,1-1e-5))))
    scored = [
        score(split,'strict_training_evidence',test,strict),
        score(split,'degree_product_heuristic',test,degree_p),
        score(split,'full_feature_catboost',test,cat),
    ]
    # Persist row-level held-out predictions.  The original benchmark retained
    # only aggregate metrics, which prevented valid paired uncertainty analysis.
    # These are produced from the same deterministic split and model protocol.
    base = test[['split', 'event_id', 'study', 'site', 'year', 'week', 'bee', 'plant', 'interaction_observed']].copy() if 'split' in test.columns else test[['event_id', 'study', 'site', 'year', 'week', 'bee', 'plant', 'interaction_observed']].copy()
    base.insert(0, 'split', split)
    base['p_strict_training_evidence'] = strict
    base['p_degree_product_heuristic'] = degree_p
    base['p_full_feature_catboost'] = cat
    return scored, base

def main():
    OUT.mkdir(parents=True,exist_ok=True); data=pd.read_csv(DATA,parse_dates=['date']); rows=[]; predictions=[]
    for name,train,test in v15.splits(data):
        print(name,flush=True)
        split_rows, split_predictions = run(name,train.copy(),test.copy())
        rows += split_rows
        predictions.append(split_predictions)
        # Persist after each completed split so a long benchmark is resumable
        # and progress remains inspectable if the host stops unexpectedly.
        pd.DataFrame(rows).to_csv(OUT/'metrics_partial.csv',index=False)
        pd.concat(predictions, ignore_index=True).to_csv(OUT/'predictions_partial.csv.gz', index=False, compression='gzip')
        (OUT/'progress.json').write_text(json.dumps({'completed_split':name,'completed_models':len(rows),'total_splits':len(v15.splits(data))},indent=2),encoding='utf8')
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT/'predictions.csv.gz', index=False, compression='gzip')
    (OUT/'summary.json').write_text(json.dumps({'data':'factorized_model_v15','splits':'exact run_factorized_ai_experiment_v15.splits','models':['strict_training_evidence','degree_product_heuristic','full_feature_catboost'],'label':'recorded interaction conditional on same-event co-occurrence; non-detections are not biological negatives'},indent=2),encoding='utf8')
if __name__=='__main__':main()
