"""Read-only CPU diagnostics: force signal, adapter reliance and nested ridge probes.

Diagnostic artifacts are new files; encoder checkpoints and training data are not
modified. Ridge probes are exploratory, not replacements for encoder acceptance.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import torch

from force_gap_forecast_v1.common import write_json
from force_gap_forecast_v1.data import FOLDS, TRAIN_IDS, load_pairs, normalized_features, window
from force_gap_motion_force_v1.metrics import prediction_parts, compose_prediction
from force_gap_motion_force_v1.model import MotionForceForecast


def correlation(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def load_model(path, original=False):
    p = torch.load(path, map_location="cpu", weights_only=True)
    model = MotionForceForecast(p["stats"], p["state_dict"] if original else None)
    if not original:
        model.load_state_dict(p["state_dict"])
    return model.eval(), p["stats"]


def data_audit(episodes, output):
    rows = []
    for tid in TRAIN_IDS:
        real, sim = episodes[(tid, 0)], episodes[(tid, 1)]
        row = {"trajectory": tid, "copied_q_dq": bool(np.array_equal(real.q, sim.q) and np.array_equal(real.dq, sim.dq)),
               "real_abs_peak_xyz": np.abs(real.force).max(0).tolist(), "sim_abs_peak_xyz": np.abs(sim.force).max(0).tolist(),
               "peak_ratio_xyz": (np.abs(sim.force).max(0) / np.maximum(np.abs(real.force).max(0), 1e-8)).tolist(),
               "correlation_xyz": [correlation(real.force[:, i], sim.force[:, i]) for i in range(3)],
               "delta_correlation_xyz": [correlation(np.diff(real.force[:, i]), np.diff(sim.force[:, i])) for i in range(3)]}
        # Magnitude of paired change events avoids choosing an advantageous axis/sign.
        dr, ds = np.linalg.norm(np.diff(real.force, axis=0), axis=1), np.linalg.norm(np.diff(sim.force, axis=0), axis=1)
        row["change_magnitude_correlation"] = correlation(dr, ds)
        rows.append(row)
    summary = {"trajectories": rows, "copied_motion_count": sum(row["copied_q_dq"] for row in rows),
               "median_peak_ratio_xyz": np.median([row["peak_ratio_xyz"] for row in rows], axis=0).tolist(),
               "peak_ratio_p10_p90_xyz": np.quantile([row["peak_ratio_xyz"] for row in rows], [.1, .9], axis=0).tolist(),
               "median_correlation_xyz": [float(np.median([row["correlation_xyz"][i] for row in rows if row["correlation_xyz"][i] is not None])) for i in range(3)],
               "median_delta_correlation_xyz": [float(np.median([row["delta_correlation_xyz"][i] for row in rows if row["delta_correlation_xyz"][i] is not None])) for i in range(3)],
               "median_change_magnitude_correlation": float(np.median([row["change_magnitude_correlation"] for row in rows]))}
    write_json(output / "force_signal_audit.json", summary)
    print("Force signal:", {k:v for k,v in summary.items() if k!="trajectories"}, flush=True)
    examples = [6, 12, 24]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
    for ax, tid in zip(axes, examples):
        real, sim = episodes[(tid, 0)], episodes[(tid, 1)]
        t = real.time - real.time[0]
        ax.plot(t, real.force[:, 2], color="#25344a", label="Real Fz")
        ax.plot(t, sim.force[:, 2], color="#e87924", label="Sim Fz")
        ax.set(title=f"traj_{tid}: force amplitude and waveform mismatch", ylabel="Fz (N)")
        ax.grid(alpha=.2); ax.legend()
    axes[-1].set_xlabel("Paired reference time (s)")
    fig.savefig(output / "force_signal_examples.png", dpi=160)
    plt.close(fig)
    return summary


def metrics(pred, real, sim, scale):
    row = {"current_rmse_n": float(np.sqrt(np.mean((pred["current"]-real.force)**2)))}
    current = float(np.sqrt(np.mean(((pred["current"]-real.force)/scale)**2)))
    futures = []
    for step in range(1, 6):
        index = np.arange(len(real.force))
        index = index[(index+step<real.ends) & (index+step<sim.ends)]
        err = pred["future"][index, step-1]-real.force[index+step]
        row[f"future_{step}_rmse_n"] = float(np.sqrt(np.mean(err**2)))
        futures.append(float(np.sqrt(np.mean((err/scale)**2))))
    row["score"] = .5*(current+float(np.mean(futures)))
    return row


def averaged(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}


def adapter_audit(root, episodes, output):
    totals = {k:[] for k in ["motion_only", "fusion", "force_at_train_mean"]}
    details = []
    selections = json.loads((root/"selections.json").read_text())
    for seed, choices in selections.items():
        for choice in choices["full"]:
            fold, gain = choice["fold"], choice["selected_gain"]
            model, stats = load_model(root/"stages"/f"seed{seed}_fold{fold}_full_refit/model.pt")
            differences, sizes, gate_means, gate_stds = [], [], [], []
            for tid in FOLDS[fold]:
                real, sim = episodes[(tid, 0)], episodes[(tid, 1)]
                parts = prediction_parts(model, sim, stats, 1)
                pred = compose_prediction(parts, sim, 1, gain)
                baseline = compose_prediction(parts, sim, 1, 0.)
                changed = copy.deepcopy(sim)
                changed.force = np.repeat(np.asarray(stats["sim"]["mean"][:3], np.float32)[None], len(sim.force), axis=0)
                # A train-mean force trace has zero physical dF; this is an
                # inference sensitivity diagnostic, not a retrained ablation.
                ablated = compose_prediction(prediction_parts(model, changed, stats, 1), sim, 1, gain)
                scale = np.asarray(stats["target"]["scale"])
                for name, prediction in [("motion_only", baseline), ("fusion", pred), ("force_at_train_mean", ablated)]:
                    totals[name].append(metrics(prediction, real, sim, scale))
                sizes.append(np.sqrt(np.mean((pred["future"]-baseline["future"])**2)))
                differences.append(np.sqrt(np.mean((pred["future"]-ablated["future"])**2)))
                if gain:
                    features = normalized_features(sim, stats, "sim")
                    x = torch.from_numpy(np.stack([window(features, t, sim.starts[t]) for t in range(len(features))]))
                    with torch.no_grad():
                        h, _, _, _ = model.encode(x, 1)
                        identity = x.new_zeros((*x.shape[:2],2)); identity[...,1]=1
                        force_hidden,_=model.force_residual.gru(torch.cat((x[...,:6],identity),-1))
                        gate=torch.sigmoid(model.force_residual.gate(torch.cat((h,force_hidden[:,-1]),-1))).numpy()
                    gate_means.append(gate.mean());gate_stds.append(gate.std(axis=0).mean())
            row={"seed":int(seed),"fold":fold,"gain":gain,"forecast_correction_rms_n":float(np.mean(sizes)),
                 "forecast_force_sensitivity_rms_n":float(np.mean(differences)),
                 "gate_mean":float(np.mean(gate_means)) if gate_means else None,
                 "gate_temporal_std_mean":float(np.mean(gate_stds)) if gate_stds else None}
            details.append(row)
            print("Adapter:",row,flush=True)
    result={"same_device_oof_metrics":{k:averaged(v) for k,v in totals.items()},"folds":details,
            "limitation":"Train-mean replacement can be out-of-distribution; it measures reliance and is not proof of conditional irrelevance."}
    write_json(output/"adapter_force_reliance.json",result)
    return result


def probe_features(model, stats, episodes, ids):
    rows={}
    scale=np.asarray(stats["target"]["scale"])
    for tid in ids:
        sim=episodes[(tid,1)]
        parts=prediction_parts(model,sim,stats,1)
        base=compose_prediction(parts,sim,1,0.)
        # All features are causal; no future target or observed future state.
        motion=np.column_stack((base["h"],base["current"]/scale,base["z"][:,3:6]/scale,
                                base["future"].reshape(len(sim.force),-1)/np.tile(scale,5)))
        fs=normalized_features(sim,stats,"sim")[:,:6]
        fr=normalized_features(episodes[(tid,0)],stats,"real")[:,:6]
        sim_history=np.stack([window(fs,t,sim.starts[t]).reshape(-1) for t in range(len(fs))])
        real=episodes[(tid,0)]
        real_history=np.stack([window(fr,t,real.starts[t]).reshape(-1) for t in range(len(fr))])
        rows[tid]={"motion":motion,"motion_sim":np.column_stack((motion,sim_history)),
                   "motion_real":np.column_stack((motion,real_history)),"base":base}
    return rows


def fit_ridge(rows, episodes, ids, scale, name, alphas):
    candidates={alpha:[] for alpha in alphas}
    for step in range(6):
        xs,ys,ws=[],[],[]
        for tid in ids:
            real,sim=episodes[(tid,0)],episodes[(tid,1)]
            indices=np.arange(len(real.force))
            indices=indices[(indices+step<real.ends)&(indices+step<sim.ends)]
            base=rows[tid]["base"]["current"] if step==0 else rows[tid]["base"]["future"][:,step-1]
            xs.append(rows[tid][name][indices]);ys.append((real.force[indices+step]-base[indices])/scale)
            ws.append(np.full(len(indices),1/(len(ids)*len(indices))))
        x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)
        mx=(x*w[:,None]).sum(0);sx=np.maximum(np.sqrt(((x-mx)**2*w[:,None]).sum(0)),1e-3)
        my=(y*w[:,None]).sum(0);z=(x-mx)/sx
        gram=z.T@(z*w[:,None]);right=z.T@((y-my)*w[:,None])
        for alpha in alphas:
            coef=linalg.solve(gram+alpha*np.eye(gram.shape[0]),right,assume_a="pos")
            candidates[alpha].append((mx,sx,my,coef))
    return candidates


def predict_ridge(row, fitted, scale, name, gain):
    corrections=[(((row[name]-mx)/sx)@coef+my)*scale for mx,sx,my,coef in fitted]
    return {"current":row["base"]["current"]+gain*corrections[0],
            "future":row["base"]["future"]+gain*np.stack(corrections[1:],axis=1)}


def ridge_probe(base_run, episodes, seeds, output):
    alphas=[.001,.01,.1,1.,10.];gains=[0.,.25,.5,1.]
    results={name:[] for name in ["motion_baseline","motion","motion_sim","motion_real"]};choices=[]
    for seed in seeds:
        for fold,outer_valid in enumerate(FOLDS):
            outer_train=sorted(set(TRAIN_IDS)-set(outer_valid))
            order=np.random.default_rng(42+fold).permutation(outer_train)
            inner_valid=sorted(map(int,order[:6]));inner_train=sorted(set(outer_train)-set(inner_valid))
            prefix=base_run/"stages"/f"seed{seed}_fold{fold}_motion_only_student"
            inner,si=load_model(Path(str(prefix)+"_select")/"model.pt",original=True)
            inner_rows=probe_features(inner,si,episodes,outer_train);scale_i=np.asarray(si["target"]["scale"])
            outer,so=load_model(Path(str(prefix)+"_refit")/"model.pt",original=True)
            outer_rows=probe_features(outer,so,episodes,TRAIN_IDS);scale_o=np.asarray(so["target"]["scale"])
            baseline=averaged([metrics(inner_rows[tid]["base"],episodes[(tid,0)],episodes[(tid,1)],scale_i) for tid in inner_valid])
            for tid in outer_valid:
                row=metrics(outer_rows[tid]["base"],episodes[(tid,0)],episodes[(tid,1)],scale_o)
                results["motion_baseline"].append({"seed":seed,"trajectory":tid,**row})
            for name in ["motion","motion_sim","motion_real"]:
                candidates=fit_ridge(inner_rows,episodes,inner_train,scale_i,name,alphas)
                best_score=baseline["score"];selected_alpha=1.;selected_gain=0.
                for alpha,fitted in candidates.items():
                    for gain in gains[1:]:
                        rows=[metrics(predict_ridge(inner_rows[tid],fitted,scale_i,name,gain),
                                      episodes[(tid,0)],episodes[(tid,1)],scale_i) for tid in inner_valid]
                        score=averaged(rows)["score"]
                        if score<best_score:
                            best_score,selected_alpha,selected_gain=score,alpha,gain
                fitted=fit_ridge(outer_rows,episodes,outer_train,scale_o,name,[selected_alpha])[selected_alpha]
                for tid in outer_valid:
                    prediction=predict_ridge(outer_rows[tid],fitted,scale_o,name,selected_gain)
                    results[name].append({"seed":seed,"trajectory":tid,
                        **metrics(prediction,episodes[(tid,0)],episodes[(tid,1)],scale_o)})
                choice={"seed":seed,"fold":fold,"features":name,"alpha":selected_alpha,"gain":selected_gain,
                        "inner_relative_improvement":1-best_score/baseline["score"]}
                choices.append(choice);print("Ridge:",choice,flush=True)
            del inner,outer
    metric_keys=["score","current_rmse_n",*[f"future_{step}_rmse_n" for step in range(1,6)]]
    summaries={name:{key:float(np.mean([r[key] for r in rows])) for key in metric_keys} for name,rows in results.items()}
    audit={"seeds":seeds,"alphas":alphas,"gains":gains,"summaries":summaries,"choices":choices,"rows":results,
           "protocol":"Original fold-matched frozen checkpoints. Ridge fit/standardization on inner train only; alpha/gain selected on inner validation; refit on outer train; outer test untouched.",
           "limitations":["Exploratory analysis, not encoder acceptance or a replacement model.",
                          "No peak/event guards in probe selection; compare force probe to equally tuned motion-only ridge control.",
                          "motion_real uses causal actual Real sensor values as a diagnostic positive control; current target is therefore directly observable."]}
    write_json(output/"nested_causal_ridge_probe.json",audit)
    print("Ridge summaries:",summaries,flush=True)
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--probe-seeds",type=int,nargs="+",default=[43])
    args=parser.parse_args();root=args.run_dir.resolve()
    recipe=json.loads((root/"recipe.json").read_text());base=Path(recipe["base_run"])
    output=root/"cause_analysis";output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    episodes,_=load_pairs(recipe["real_root"],recipe["sim_root"],recipe["labels"])
    data_audit(episodes,output)
    adapter_audit(root,episodes,output)
    ridge_probe(base,episodes,args.probe_seeds,output)
    print("Diagnostics written to",output,flush=True)


if __name__=="__main__":main()
