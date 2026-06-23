#!/usr/bin/env python3
"""AT-KD vs Vanilla-KD reproducer.

Stratified k-fold CV, multi-seed. Trains a small student under three loss settings
(student-only / vanilla-KD / AT-KD) with the same teacher, reports F1 / Acc / ECE +
McNemar AT-KD vs Vanilla-KD on pooled out-of-fold predictions.

Usage:
    python train.py --data-dir /path/to/ImageFolder --teacher swinv2_tiny_window8_256 \
        --student mobilenetv3_large_100 --img-size 256 --folds 5 --epochs 60
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.loss import ATKDLoss, VanillaKDLoss
from src.models import TeacherModel, StudentModel
from src.data import build_loaders, labels_of
from src.stats import ece_score, mcnemar_p, bootstrap_f1_ci


SCENARIOS = ("student_only", "vanilla_kd", "atkd")


def set_seed(seed: int):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(student, loader, device, nc):
    student.eval(); ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits, _ = student(x)
        ps.append(F.softmax(logits, 1).cpu().numpy()); ys.append(y.numpy())
    y = np.concatenate(ys); probs = np.concatenate(ps); pred = probs.argmax(1)
    return {
        "y": y, "pred": pred, "probs": probs,
        "f1": float(f1_score(y, pred, average="macro")),
        "acc": float(accuracy_score(y, pred)),
        "ece": ece_score(probs, y),
    }


def train_one_fold(args, scn, train_loader, val_loader, test_loader, nc, device):
    teacher = None
    if scn != "student_only":
        teacher = TeacherModel(args.teacher, num_classes=nc, pretrained=True, freeze=True).to(device)
    student = StudentModel(args.student, num_classes=nc, pretrained=False).to(device)

    if scn == "student_only":
        criterion = torch.nn.CrossEntropyLoss()
    elif scn == "vanilla_kd":
        criterion = VanillaKDLoss(temperature=args.temperature).to(device)
    elif scn == "atkd":
        criterion = ATKDLoss(temperature=args.temperature, gamma=args.gamma_at).to(device)
    else:
        raise ValueError(scn)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    best_val = -1.0; best_state = None; bad = 0

    for ep in range(args.epochs):
        student.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad()
            ctx = torch.amp.autocast("cuda", dtype=torch.float16) if scaler else torch.enable_grad()
            with ctx:
                s_logits, s_feat = student(x)
                if scn == "student_only":
                    loss = criterion(s_logits, y)
                else:
                    with torch.no_grad():
                        t_logits, t_feat = teacher(x)
                    if scn == "vanilla_kd":
                        loss = criterion(s_logits, t_logits, y)
                    else:
                        loss = criterion(s_logits, t_logits, s_feat, t_feat, y)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
        sched.step()
        m = evaluate(student, val_loader, device, nc)
        if m["f1"] > best_val + 1e-4:
            best_val = m["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        student.load_state_dict(best_state)
    test = evaluate(student, test_loader, device, nc)
    del student
    if teacher is not None: del teacher
    torch.cuda.empty_cache()
    return {"val_f1": best_val, **test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--teacher", default="swinv2_tiny_window8_256")
    ap.add_argument("--student", default="mobilenetv3_large_100")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--gamma-at", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs/atkd_reproduce")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    labels, classes = labels_of(args.data_dir)
    nc = len(classes)
    print(f"#### dataset={args.data_dir}  n={len(labels)}  classes={classes}  "
          f"teacher={args.teacher}  student={args.student}  device={device}", flush=True)

    results = {scn: [] for scn in scenarios}
    for seed in seeds:
        set_seed(seed)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (tr_all, te) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
            tr, va = train_test_split(tr_all, test_size=0.2, stratify=labels[tr_all],
                                      random_state=seed)
            trl, val_l, te_l = build_loaders(args.data_dir, tr, va, te, args.img_size,
                                             args.batch_size, args.workers)
            for scn in scenarios:
                set_seed(seed * 100 + fold)
                t0 = time.time()
                r = train_one_fold(args, scn, trl, val_l, te_l, nc, device)
                r.update({"scenario": scn, "seed": seed, "fold": fold, "time_s": time.time() - t0})
                results[scn].append(r)
                print(f"  seed={seed} fold={fold} {scn:14s} val_F1={r['val_f1']:.3f} "
                      f"test_F1={r['f1']:.3f} acc={r['acc']:.3f} ece={r['ece']:.3f} "
                      f"({r['time_s']:.0f}s)", flush=True)

    summary = {}
    pooled = {}
    for scn in scenarios:
        f1 = np.array([r["f1"] for r in results[scn]])
        ece = np.array([r["ece"] for r in results[scn]])
        y_all = np.concatenate([r["y"] for r in results[scn]])
        p_all = np.concatenate([r["pred"] for r in results[scn]])
        lo, hi = bootstrap_f1_ci(y_all, p_all)
        summary[scn] = {
            "f1_mean": float(f1.mean()), "f1_std": float(f1.std()),
            "f1_ci95": [lo, hi], "ece_mean": float(ece.mean()),
            "n_folds": int(len(f1)),
        }
        pooled[scn] = {"y": y_all, "pred": p_all}

    # paired McNemar atkd vs vanilla_kd on pooled predictions, when both ran
    if "atkd" in pooled and "vanilla_kd" in pooled:
        y = pooled["atkd"]["y"]
        # sanity: pooled labels should align since splits are seed-locked across scenarios
        if np.array_equal(y, pooled["vanilla_kd"]["y"]):
            summary["atkd_vs_vanilla_p"] = mcnemar_p(y, pooled["vanilla_kd"]["pred"],
                                                    pooled["atkd"]["pred"])

    json.dump({"args": vars(args), "summary": summary,
               "per_run": [{k: (v if not isinstance(v, np.ndarray) else v.tolist())
                            for k, v in r.items()} for scn in scenarios for r in results[scn]]},
              open(out / "result.json", "w"), indent=2)

    print(f"\n#### SUMMARY ({args.folds}-fold x {len(seeds)} seeds)")
    for scn in scenarios:
        s = summary[scn]
        print(f"  {scn:14s} F1={s['f1_mean']:.3f}+-{s['f1_std']:.3f} "
              f"[95% {s['f1_ci95'][0]:.3f},{s['f1_ci95'][1]:.3f}]  ECE={s['ece_mean']:.3f}")
    if "atkd_vs_vanilla_p" in summary:
        print(f"  McNemar AT-KD vs Vanilla-KD: p={summary['atkd_vs_vanilla_p']:.4f}")
    print(f"saved -> {out/'result.json'}")


if __name__ == "__main__":
    main()
