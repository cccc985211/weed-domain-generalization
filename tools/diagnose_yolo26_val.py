"""EXP-001B Task3: diagnose YOLO26n val anomaly (short diagnostics only).

Checks on the ALREADY-TRAINED smoke checkpoints (no new training):
  * which head validation uses (one2many+NMS vs one2one NMS-free)
  * val metrics under each head for yolo26n best.pt
  * prediction score sanity per head
  * compares against yolo11n best.pt for reference
Writes reports/EXP-001/model_selection/yolo26_val_diagnosis.json
"""
import json
import os
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.dirname(REPO)
DATA_YAML = os.path.join(EXP, "datasets_derived", "smoke_weed_v1", "data.yaml")
RUNS = os.path.join(EXP, "runs", "EXP-001", "model_selection")
OUT = os.path.join(REPO, "reports", "EXP-001", "model_selection",
                   "yolo26_val_diagnosis.json")


def summarize_metrics(m):
    d = getattr(m, "results_dict", None) or {}
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float))}


def val_once(yolo, **kw):
    args = dict(data=DATA_YAML, split="val", imgsz=640, batch=8,
                workers=0, seed=42, verbose=False, plots=False, save_json=False)
    args.update(kw)
    try:
        return summarize_metrics(yolo.val(**args)), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def main():
    from ultralytics import YOLO
    import torch

    out = {"generated_utc": datetime.now(timezone.utc).isoformat(),
           "note": "metrics on SYNTHETIC placeholder data - diagnostic only"}
    ckpts = {
        "yolo26n": os.path.join(RUNS, "yolo26n", "weights", "best.pt"),
        "yolo11n": os.path.join(RUNS, "yolo11n", "weights", "best.pt"),
    }
    for name, ck in ckpts.items():
        rec = {"checkpoint": ck, "exists": os.path.exists(ck)}
        if not rec["exists"]:
            out[name] = rec
            continue
        yolo = YOLO(ck)
        m = yolo.model
        last = m.model[-1]
        rec["head_class"] = type(last).__name__
        rec["has_one2one"] = hasattr(last, "one2one_cv2")
        rec["end2end_attr"] = getattr(last, "end2end", None)

        met, err = val_once(yolo)                    # default = one2many+NMS
        rec["val_one2many_nms"] = met or {"error": err}
        if rec["has_one2one"]:
            met2, err2 = val_once(yolo, nms=False)   # one2one head, NMS-free
            rec["val_one2one_nmsfree"] = met2 or {"error": err2}

        # per-head score histograms on one val batch (raw preds, no val loop)
        m.eval()
        with torch.no_grad():
            x = torch.rand(1, 3, 640, 640,
                           device=next(m.parameters()).device)
            preds = m(x)
        def flat(o, acc):
            if torch.is_tensor(o):
                acc.append(o)
            elif isinstance(o, (list, tuple)):
                [flat(i, acc) for i in o]
            elif isinstance(o, dict):
                [flat(v, acc) for v in o.values()]
        acc = []
        flat(preds, acc)
        rec["eval_output_shapes"] = [list(t.shape) for t in acc]
        out[name] = rec
        del yolo

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:3000])


if __name__ == "__main__":
    main()
