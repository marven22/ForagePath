"""Benchmark strict, ML, and PU graph models for recorded plant-bee interactions.

The outcome is a recorded interaction conditional on both taxa being observed
in the same study-site-date event.  A zero is an unlabeled opportunity pair,
not evidence of biological incompatibility.  The neural models use a weak
positive-unlabeled loss: positives are reinforced; unlabeled pairs are only a
light counterweight rather than declared negatives.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim" / "interaction_benchmark_v14" / "temporal_joint_availability_candidates.csv.gz"
OUT = ROOT / "results" / "plant_bee_gnn_benchmark_v1"
SEED = 20260819
torch.manual_seed(SEED); np.random.seed(SEED)


def bucket(value: str, modulus: int = 7) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % modulus


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    r = 2 * np.pi * (out.week.to_numpy(float) - 1) / 52.1775
    out["week_sin"], out["week_cos"] = np.sin(r), np.cos(r)
    return out


def probability_calibration(y: np.ndarray, scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(np.unique(y)) < 2:
        return 1 / (1 + np.exp(-target))
    model = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED)
    model.fit(scores.reshape(-1, 1), y)
    return model.predict_proba(target.reshape(-1, 1))[:, 1]


def logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 1e-5, 1 - 1e-5)
    return np.log(value / (1 - value))


def metric_row(split: str, model: str, test: pd.DataFrame, probability: np.ndarray) -> dict[str, object]:
    y = test.interaction_observed.to_numpy(int)
    return {"split": split, "model": model, "rows": len(test), "events": int(test.event_id.nunique()),
            "positive_rate": float(y.mean()), "average_precision": float(average_precision_score(y, probability)),
            "roc_auc": float(roc_auc_score(y, probability)), "brier": float(brier_score_loss(y, probability))}


def cat_features(frame: pd.DataFrame) -> pd.DataFrame:
    # No species IDs: this makes the tabular model a real unseen-taxon test.
    return add_context(frame)[["bee_family", "bee_tribe", "bee_genus", "plant_family", "plant_genus", "week_sin", "week_cos"]].fillna("unknown")


def run_catboost(fit: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    x_fit, x_cal, x_test = cat_features(fit), cat_features(calibration), cat_features(test)
    categorical = list(range(5))
    model = CatBoostClassifier(iterations=160, depth=7, learning_rate=0.05, loss_function="Logloss", verbose=False,
                               random_seed=SEED, l2_leaf_reg=8.0, allow_writing_files=False)
    model.fit(x_fit, fit.interaction_observed, cat_features=categorical)
    cal = model.predict_proba(x_cal)[:, 1]
    raw = model.predict_proba(x_test)[:, 1]
    # Conventional supervised comparison: target remains recorded interaction,
    # not a biological yes/no edge.
    return probability_calibration(calibration.interaction_observed.to_numpy(int), logit(cal), logit(raw))


def make_node_features(frames: list[pd.DataFrame]) -> tuple[dict[str, int], dict[str, int], torch.Tensor, torch.Tensor]:
    all_rows = pd.concat(frames, ignore_index=True)
    bees, plants = sorted(all_rows.bee.unique()), sorted(all_rows.plant.unique())
    bee_i, plant_i = {x:i for i,x in enumerate(bees)}, {x:i for i,x in enumerate(plants)}
    # Fixed hashed taxonomy features permit unseen species to enter with known traits.
    dim = 130; x = np.zeros((len(bees)+len(plants), dim), dtype=np.float32); types=np.zeros(len(x),dtype=np.int64)
    x[:len(bees),0]=1; x[len(bees):,1]=1; types[len(bees):]=1
    bee_meta = all_rows.drop_duplicates("bee").set_index("bee")
    plant_meta = all_rows.drop_duplicates("plant").set_index("plant")
    for name,i in bee_i.items():
        for field,slot in (("bee_family",2),("bee_tribe",34),("bee_genus",66)):
            value=str(bee_meta.loc[name,field]); x[i,slot+bucket(field+value,32)]=1
    for name,i in plant_i.items():
        node=len(bees)+i
        for field,slot in (("plant_family",2),("plant_genus",34)):
            value=str(plant_meta.loc[name,field]); x[node,slot+bucket(field+value,32)]=1
    return bee_i, plant_i, torch.tensor(x), torch.tensor(types)


def edges_from_fit(fit: pd.DataFrame, bee_i: dict[str,int], plant_i: dict[str,int], n_bee: int) -> tuple[torch.Tensor, torch.Tensor]:
    pairs=fit.loc[fit.interaction_observed.eq(1),["bee","plant"]].drop_duplicates()
    src=np.r_[pairs.bee.map(bee_i).to_numpy(), n_bee+pairs.plant.map(plant_i).to_numpy()]
    dst=np.r_[n_bee+pairs.plant.map(plant_i).to_numpy(), pairs.bee.map(bee_i).to_numpy()]
    rel=np.r_[np.zeros(len(pairs),dtype=np.int64),np.ones(len(pairs),dtype=np.int64)]
    return torch.tensor(np.stack([src,dst])), torch.tensor(rel)


def mean_aggregate(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src,dst=edge_index; out=torch.zeros_like(h); out.index_add_(0,dst,h[src]); degree=torch.zeros(len(h),device=h.device); degree.index_add_(0,dst,torch.ones_like(dst,dtype=torch.float)); return out/degree.clamp_min(1).unsqueeze(1)


class Sage(nn.Module):
    def __init__(self, inputs:int=130, hidden:int=48):
        super().__init__(); self.a=nn.Linear(inputs,hidden); self.b=nn.Linear(hidden*2,hidden); self.c=nn.Linear(hidden*2,hidden); self.dec=nn.Sequential(nn.Linear(hidden*2+5,48),nn.ReLU(),nn.Linear(48,1))
    def encode(self,x,edge,types):
        h=F.relu(self.a(x)); h=F.relu(self.b(torch.cat([h,mean_aggregate(h,edge)],1))); return F.relu(self.c(torch.cat([h,mean_aggregate(h,edge)],1)))
    def score(self,h,bi,pi,week): return self.dec(torch.cat([h[bi]*h[pi],torch.abs(h[bi]-h[pi]),week],1)).squeeze(1)


class HGT(nn.Module):
    """Small relation-aware heterogeneous graph transformer implemented in PyTorch."""
    def __init__(self, inputs:int=130, hidden:int=48):
        super().__init__(); self.input=nn.ModuleList([nn.Linear(inputs,hidden),nn.Linear(inputs,hidden)]); self.q=nn.ModuleList([nn.Linear(hidden,hidden),nn.Linear(hidden,hidden)]); self.k=nn.ModuleList([nn.Linear(hidden,hidden),nn.Linear(hidden,hidden)]); self.v=nn.ModuleList([nn.Linear(hidden,hidden),nn.Linear(hidden,hidden)]); self.rel=nn.Parameter(torch.randn(2,hidden,hidden)*.05); self.out=nn.ModuleList([nn.Linear(hidden,hidden),nn.Linear(hidden,hidden)]); self.dec=nn.Sequential(nn.Linear(hidden*2+5,48),nn.ReLU(),nn.Linear(48,1))
    def encode(self,x,edge,types,relations):
        h=torch.empty((len(x),self.input[0].out_features),device=x.device)
        for t in (0,1): h[types==t]=F.relu(self.input[t](x[types==t]))
        src,dst=edge; q=torch.empty_like(h); k=torch.empty_like(h); v=torch.empty_like(h)
        for t in (0,1):
            m=types==t; q[m]=self.q[t](h[m]); k[m]=self.k[t](h[m]); v[m]=self.v[t](h[m])
        rel_v=torch.bmm(v[src].unsqueeze(1),self.rel[relations]).squeeze(1)
        score=(q[dst]*rel_v).sum(1)/(h.shape[1]**.5); maxes=torch.full((len(h),),-1e9,device=x.device); maxes.scatter_reduce_(0,dst,score,reduce="amax",include_self=True); ex=torch.exp(score-maxes[dst]); den=torch.zeros(len(h),device=x.device); den.index_add_(0,dst,ex); agg=torch.zeros_like(h); agg.index_add_(0,dst,rel_v*(ex/den[dst]).unsqueeze(1))
        z=torch.empty_like(h)
        for t in (0,1):
            m=types==t; z[m]=F.relu(self.out[t](h[m]+agg[m]))
        return z
    def score(self,h,bi,pi,week): return self.dec(torch.cat([h[bi]*h[pi],torch.abs(h[bi]-h[pi]),week],1)).squeeze(1)


def gnn_scores(kind: str, fit: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    bee_i, plant_i, x, types=make_node_features([fit,calibration,test]); n_bee=len(bee_i); edge,relations=edges_from_fit(fit,bee_i,plant_i,n_bee)
    model=Sage() if kind=="pu_graphsage" else HGT()
    # Rich-node ablations can request a smaller learning rate.  The compact
    # benchmark retains its original .008 default, preserving its protocol.
    learning_rate = float(globals().get("GNN_LEARNING_RATE", .008))
    gradient_clip = float(globals().get("GNN_GRADIENT_CLIP", 0.0))
    opt=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    def context(frame: pd.DataFrame) -> torch.Tensor:
        rows=add_context(frame)
        grain=rows.get("sampling_grain",pd.Series("unknown",index=rows.index)).fillna("unknown").astype(str).map(lambda x: bucket(x,5)/5).to_numpy(float)
        event_size=pd.to_numeric(rows.get("log_event_candidate_pairs",pd.Series(0,index=rows.index)),errors="coerce").fillna(0).to_numpy(float)
        study_size=pd.to_numeric(rows.get("log_study_events",pd.Series(0,index=rows.index)),errors="coerce").fillna(0).to_numpy(float)
        return torch.tensor(np.c_[rows.week_sin,rows.week_cos,grain,event_size,study_size],dtype=torch.float32)
    rows=add_context(fit); bi=torch.tensor(rows.bee.map(bee_i).to_numpy()); pi=torch.tensor(n_bee+rows.plant.map(plant_i).to_numpy()); week=context(fit); y=torch.tensor(rows.interaction_observed.to_numpy(),dtype=torch.bool)
    # Positive-unlabeled risk: unlabeled pairs receive only 0.20 of the usual
    # negative pressure, rather than being declared false biological links.
    for epoch in range(25):
        opt.zero_grad(); h=model.encode(x,edge,types,relations) if kind=="pu_hgt" else model.encode(x,edge,types); score=model.score(h,bi,pi,week); loss=F.softplus(-score[y]).mean()+.20*F.softplus(score[~y]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{kind} produced a non-finite loss at epoch {epoch}")
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        opt.step()
    def raw(frame):
        rows=add_context(frame); a=torch.tensor(rows.bee.map(bee_i).to_numpy()); b=torch.tensor(n_bee+rows.plant.map(plant_i).to_numpy()); w=context(frame)
        with torch.no_grad():
            h=model.encode(x,edge,types,relations) if kind=="pu_hgt" else model.encode(x,edge,types); value=model.score(h,a,b,w).numpy()
            if not np.isfinite(value).all():
                raise FloatingPointError(f"{kind} produced non-finite scores after training")
            return value
    return probability_calibration(calibration.interaction_observed.to_numpy(int),raw(calibration),raw(test))


def strict_scores(fit: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    pairs=set(map(tuple,fit.loc[fit.interaction_observed.eq(1),["bee","plant"]].to_numpy()))
    return np.array([.95 if (b,p) in pairs else .05 for b,p in zip(test.bee,test.plant)])


def degree_scores(fit: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    prior=fit.interaction_observed.mean(); b=(fit.groupby("bee").interaction_observed.mean()+prior).to_dict(); p=(fit.groupby("plant").interaction_observed.mean()+prior).to_dict()
    def raw(f): return np.array([np.sqrt(b.get(x,prior)*p.get(y,prior)) for x,y in zip(f.bee,f.plant)])
    return probability_calibration(calibration.interaction_observed.to_numpy(int),raw(calibration),raw(test))


def evaluate(split: str, train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[dict[str,object]],pd.DataFrame]:
    buckets=train.event_id.map(lambda x:bucket(str(x))); fit=train.loc[~buckets.isin([0,1])].copy(); calibration=train.loc[buckets.eq(1)].copy()
    predictions={"strict_training_evidence":strict_scores(fit,test),"degree_product_heuristic":degree_scores(fit,calibration,test),"trait_context_catboost":run_catboost(fit,calibration,test),"pu_graphsage":gnn_scores("pu_graphsage",fit,calibration,test),"pu_hgt":gnn_scores("pu_hgt",fit,calibration,test)}
    rows=[metric_row(split,k,test,v) for k,v in predictions.items()]
    out=test[["event_id","study","site","year","week","bee","plant","interaction_observed"]].copy(); out.insert(0,"split",split)
    for k,v in predictions.items(): out["p_"+k]=v
    return rows,out


def splits(data:pd.DataFrame):
    yield "held_event",data.loc[data.event_id.map(lambda x:bucket(str(x)))!=0],data.loc[data.event_id.map(lambda x:bucket(str(x)))==0]
    yield "unseen_pair",data.loc[(data.bee+'|'+data.plant).map(lambda x:bucket(x))!=0],data.loc[(data.bee+'|'+data.plant).map(lambda x:bucket(x))==0]
    yield "unseen_plant",data.loc[data.plant.map(lambda x:bucket(x))!=0],data.loc[data.plant.map(lambda x:bucket(x))==0]
    yield "unseen_bee",data.loc[data.bee.map(lambda x:bucket(x))!=0],data.loc[data.bee.map(lambda x:bucket(x))==0]
    latest=data.groupby(["study","site"]).year.transform("max"); eligible=data.groupby(["study","site"]).year.transform("nunique")>=2; yield "future_year",data.loc[~(eligible&(data.year==latest))],data.loc[eligible&(data.year==latest)]
    # Predefined five-fold study holdout: every study is held out exactly once,
    # while keeping the benchmark practical to reproduce on a CPU workstation.
    study_fold=data.study.map(lambda x:bucket(str(x),modulus=5))
    for fold in range(5): yield f"unseen_study_fold::{fold}",data.loc[study_fold!=fold],data.loc[study_fold==fold]


def main():
    OUT.mkdir(parents=True,exist_ok=True); data=pd.read_csv(DATA,parse_dates=["date"])
    metrics=[]; predictions=[]
    for name,train,test in splits(data):
        print(name, len(train),len(test),flush=True); rows,out=evaluate(name,train,test); metrics+=rows; predictions.append(out)
    metric=pd.DataFrame(metrics); studies=metric.loc[metric.split.str.startswith("unseen_study_fold::")]
    macro=studies.groupby("model",as_index=False).agg(rows=("rows","sum"),events=("events","sum"),positive_rate=("positive_rate","mean"),average_precision=("average_precision","mean"),roc_auc=("roc_auc","mean"),brier=("brier","mean")); macro.insert(0,"split","unseen_study_macro_average"); metric=pd.concat([metric,macro],ignore_index=True)
    metric.to_csv(OUT/"metrics.csv",index=False); pd.concat(predictions,ignore_index=True).to_csv(OUT/"test_predictions.csv.gz",index=False,compression="gzip")
    primary=metric.loc[metric.split.isin(["held_event","unseen_pair","unseen_plant","unseen_bee","future_year","unseen_study_macro_average"])]
    summary={"candidate_rows":int(len(data)),"observed_interactions":int(data.interaction_observed.sum()),"studies":int(data.study.nunique()),"models":["strict_training_evidence","degree_product_heuristic","trait_context_catboost","pu_graphsage","pu_hgt"],"outcome":"recorded interaction conditional on both taxa being recorded in the same study-site-date event","unlabeled_policy":"same-event non-detections receive weak PU loss, not a claim of biological incompatibility","best_average_precision":primary.loc[primary.groupby("split").average_precision.idxmax(),["split","model","average_precision"]].to_dict(orient="records")}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
