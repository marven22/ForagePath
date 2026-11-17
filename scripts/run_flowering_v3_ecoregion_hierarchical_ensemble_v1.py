"""V3 ecoregion ensemble plus empirical-Bayes plant-level probability calibration."""
from __future__ import annotations
import json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from run_flowering_established_model_benchmark_v1 import FULL_NUMERIC, TabularNet, make_expanded, screening_training_frame
from run_flowering_model_tuning_v2 import enhanced_weather, seasonal_tuned
from run_flowering_model_benchmark_v1 import evaluate

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'snapshots'/'us-weekly-flowering-2009-2025-v3'/'derived'/'weekly_flowering_modeling.parquet'; OUT=ROOT/'results'/'flowering_v3_ecoregion_hierarchical_ensemble_v1'; CATS=['npn_species_id','ecoregion_level3_code']; SEED=42
def seed(): random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED) if torch.cuda.is_available() else None
def encode(train,*targets):
 med=train[FULL_NUMERIC].median(numeric_only=True); mean=train[FULL_NUMERIC].fillna(med).mean(); std=train[FULL_NUMERIC].fillna(med).std().replace(0,1); levels={c:{v:i+1 for i,v in enumerate(train[c].fillna('unknown').astype(str).unique())} for c in CATS}
 def f(x): return ((x[FULL_NUMERIC].fillna(med)-mean)/std).to_numpy('float32'),np.column_stack([x[c].fillna('unknown').astype(str).map(levels[c]).fillna(0).astype('int64') for c in CATS])
 return f(train),[f(x) for x in targets],[len(levels[c])+1 for c in CATS]
def cat_predict(train,val,test):
 ex,y,w=make_expanded(train); cols=FULL_NUMERIC+CATS; med={c:float(ex[c].median()) for c in FULL_NUMERIC}
 def f(x):
  z=x[cols].copy()
  for c in FULL_NUMERIC:z[c]=z[c].fillna(med[c])
  for c in CATS:z[c]=z[c].fillna('unknown').astype(str)
  return z
 kw=dict(loss_function='Logloss',iterations=220,depth=7,learning_rate=.10,l2_leaf_reg=6,random_seed=SEED,verbose=False,allow_writing_files=False)
 if torch.cuda.is_available():kw.update(task_type='GPU',devices='0')
 m=CatBoostClassifier(**kw);m.fit(f(ex),y,sample_weight=w,cat_features=CATS);return m.predict_proba(f(val))[:,1],m.predict_proba(f(test))[:,1]
def ft_predict(train,val,test):
 (xn,xc),[(vn,vc),(tn,tc)],cards=encode(train,val,test); y=(train.positive_checks/train.observation_checks).to_numpy('float32');w=train.observation_checks.to_numpy('float32');dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');m=TabularNet(xn.shape[1],cards,'transformer').to(dev);opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=2e-4)
 for _ in range(16):
  order=np.random.permutation(len(train));m.train()
  for s in range(0,len(order),512):
   i=order[s:s+512];a=torch.from_numpy(xn[i]).to(dev);b=torch.from_numpy(xc[i]).to(dev);yy=torch.from_numpy(y[i]).to(dev);ww=torch.from_numpy(w[i]).to(dev);loss=(torch.nn.functional.binary_cross_entropy_with_logits(m(a,b),yy,reduction='none')*ww).sum()/ww.sum();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 def f(n,c):
  out=[];m.eval()
  with torch.no_grad():
   for s in range(0,len(n),512):out.append(torch.sigmoid(m(torch.from_numpy(n[s:s+512]).to(dev),torch.from_numpy(c[s:s+512]).to(dev))).cpu().numpy())
  return np.concatenate(out)
 return f(vn,vc),f(tn,tc)
def brier(frame,p):return float(np.average((p-frame.positive_checks/frame.observation_checks)**2,weights=frame.observation_checks))
def logit(p):p=np.clip(p,1e-5,1-1e-5);return np.log(p/(1-p))
def fit_offsets(frame,p,tau):
 """Binomial random intercepts with Normal(0, tau²) shrinkage."""
 z=logit(p);result={};prior=1/(tau*tau)
 for species,g in frame.assign(_z=z).groupby('npn_species_id'):
  b=0.0
  for _ in range(25):
   q=1/(1+np.exp(-(g._z.to_numpy()+b)));n=g.observation_checks.to_numpy(float);y=g.positive_checks.to_numpy(float);grad=np.sum(y-n*q)-prior*b;h=-np.sum(n*q*(1-q))-prior;new=np.clip(b-grad/h,-2,2)
   if abs(new-b)<1e-7:break
   b=new
  result[species]=b
 return result
def apply_offsets(frame,p,offsets):return 1/(1+np.exp(-(logit(p)+frame.npn_species_id.map(offsets).fillna(0).to_numpy())))
def main():
 seed();OUT.mkdir(parents=True,exist_ok=True);d=enhanced_weather(pd.read_parquet(DATA).loc[lambda x:x.weather_available].copy());full=d.loc[d.data_split.eq('train')].copy();tr=screening_training_frame(full);va=d.loc[d.data_split.eq('validation')].copy().reset_index(drop=True);te=d.loc[d.data_split.eq('test')].copy().reset_index(drop=True);cv,ct=cat_predict(tr,va,te);fv,ft=ft_predict(tr,va,te);weights=np.linspace(0,1,101);fw=min(weights,key=lambda w:brier(va,w*fv+(1-w)*cv));rv=fw*fv+(1-fw)*cv;rt=fw*ft+(1-fw)*ct
 # Global calibration is fitted only on the validation year.
 # A grouped-binomial calibration fit: each weekly row contributes both its
 # observed flowering and non-flowering checks, without exploding the data.
 x=np.r_[rv,rv];y=np.r_[np.ones(len(rv)),np.zeros(len(rv))];sw=np.r_[va.positive_checks.to_numpy(),va.negative_checks.to_numpy()];cal=LogisticRegression(C=1,solver='lbfgs',random_state=SEED).fit(logit(x).reshape(-1,1),y,sample_weight=sw);gv=cal.predict_proba(logit(rv).reshape(-1,1))[:,1];gt=cal.predict_proba(logit(rt).reshape(-1,1))[:,1]
 taus=[.10,.20,.35,.50,.75,1.0];candidates=[]
 for tau in taus:
  offsets=fit_offsets(va,gv,tau);candidates.append((tau,brier(va,apply_offsets(va,gv,offsets)),offsets))
 tau,_,offsets=min(candidates,key=lambda x:x[1]);ht=apply_offsets(te,gt,offsets);hv=apply_offsets(va,gv,offsets)
 rows=[]
 for name,pv,pt in [('catboost_ecoregion',cv,ct),('ft_transformer_ecoregion',fv,ft),('raw_ensemble_ecoregion',rv,rt),('global_calibrated_ensemble',gv,gt),('hierarchical_plant_calibrated_ensemble',hv,ht)]:
  for split,fr,p in [('validation',va,pv),('test',te,pt)]:m,_,_=evaluate(name,fr,p);rows.append({'split':split,**m})
 support=full.groupby('npn_species_id').observation_checks.sum();te['support']=te.npn_species_id.map(support).fillna(0);te['tier']=pd.cut(te.support,[-1,99,999,np.inf],labels=['sparse_<100','moderate_100_999','rich_1000+']).astype(str);tier=[]
 for label,g in te.groupby('tier'):
  for name,p in [('global',gt),('hierarchical',ht)]:tier.append({'tier':label,'model':name,'checks':int(g.observation_checks.sum()),'weighted_brier':brier(g,p[g.index])})
 pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(candidates,columns=['tau','validation_brier','offsets']).drop(columns='offsets').to_csv(OUT/'hierarchical_tau_search.csv',index=False);pd.DataFrame(tier).to_csv(OUT/'tier_metrics.csv',index=False);pd.DataFrame({'weekly_record_id':te.weekly_record_id,'global_probability':gt,'hierarchical_probability':ht}).to_parquet(OUT/'test_predictions.parquet',index=False);summary={'v3_ecoregion_codes':int(d.ecoregion_level3_code.nunique(dropna=True)),'ft_weight':float(fw),'selected_random_effect_prior_sd_tau':float(tau),'plant_offsets_fitted_from':'2023 validation only','offset_count':len(offsets),'policy':'Offsets are binomial plant random intercepts shrunk toward zero; plants absent from validation receive zero adjustment.'};(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(tier).to_string(index=False));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
