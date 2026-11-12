#!/usr/bin/env python3
"""Score your predictions on the frozen test splits and compare to our RF baselines.

Usage:
  python3 score.py PACK [SPLIT] predictions.csv [--target COL]
  python3 score.py PACK [SPLIT] --template out.csv [--target COL]
  python3 score.py PACK [SPLIT] --diagnose [--target COL]

  SPLIT: scaffold_split (default) | random_split | cluster_split
         (short forms scaffold / random / cluster also accepted)
  predictions.csv: either `row_index` (0-based row in the ml_ready CSV) plus prediction
    column(s), or prediction column(s) only, in the same order as the test rows.
    Every column whose name starts with `pred` is scored (e.g. `pred`, or
    `pred_seed0 ... pred_seed4` for several training seeds); with 2+ columns the
    mean +- sd across columns is printed too.
    Each prediction column needs one finite value per labelled test row.
  --diagnose: label diagnostics for the chosen split (no predictions needed).
Baselines come from BASELINES_5seed.csv (5 RF seeds, mean +- sd) if present,
otherwise BASELINES.csv (single seed). Needs only Python 3 (no numpy)."""
import csv, json, sys, os, math
H = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(H, "manifest.json")))
SPLITS = ("random_split", "scaffold_split", "cluster_split")
DEFAULT_SPLIT = "scaffold_split"
csv.field_size_limit(10**8)

def die(msg): sys.exit("error: " + msg)
def fnum(s):
    try: v = float(s)
    except (TypeError, ValueError): return None
    return v if math.isfinite(v) else None
def mean(v): return sum(v) / len(v)
def sd(v):  # sample sd (n-1)
    if len(v) < 2: return float("nan")
    m = mean(v); return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
def metrics(y, yh):
    n = len(y); mu = mean(y)
    ss = sum((t - mu) ** 2 for t in y); se = sum((t - p) ** 2 for t, p in zip(y, yh))
    return dict(R2=1 - se / ss if ss > 0 else float("nan"), RMSE=math.sqrt(se / n),
                MAE=sum(abs(t - p) for t, p in zip(y, yh)) / n)

# ---- argument parsing -------------------------------------------------------
a = sys.argv[1:]
if not a or a[0] in ("-h", "--help"): sys.exit(__doc__)
opts = {}
for flag in ("--target", "--template"):
    if flag in a:
        i = a.index(flag)
        if i + 1 >= len(a): die(f"{flag} needs a value")
        opts[flag] = a[i + 1]; del a[i:i + 2]
diagnose = "--diagnose" in a
a = [x for x in a if x != "--diagnose"]
if any(x.startswith("--") for x in a): die(f"unknown option {[x for x in a if x.startswith('--')][0]}\n" + __doc__)
pack = a[0]
if pack not in M: die(f"unknown pack '{pack}'. Packs: {', '.join(sorted(M))}")
rest = a[1:]
split = DEFAULT_SPLIT
def as_split(s):
    s2 = s if s.endswith("_split") else s + "_split"
    return s2 if s2 in SPLITS else None
if rest and as_split(rest[0]) and not os.path.isfile(rest[0]):
    split = as_split(rest[0]); rest = rest[1:]
pred_path = rest[0] if rest else None
if len(rest) > 1: die(f"too many arguments: {rest}\n" + __doc__)
if not (diagnose or "--template" in opts or pred_path): die("need predictions.csv, --template OUT or --diagnose\n" + __doc__)

targets = M[pack][2]
tcol = opts.get("--target", targets[0])
if tcol not in targets: die(f"target '{tcol}' not in {pack} targets {targets}")
csv_path = os.path.join(H, "packs", pack, M[pack][1])
if not os.path.isfile(csv_path): die(f"missing {csv_path}: put the pack's folder under packs/")
rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
if split not in rows[0]:
    have = [s for s in SPLITS if s in rows[0]]
    die(f"pack '{pack}' has no {split} (available: {', '.join(have)}). Pass one of those as SPLIT.")
smi_col = "smiles" if "smiles" in rows[0] else "solute_smiles"
test = [(i, r) for i, r in enumerate(rows) if r[split] == "test" and fnum(r.get(tcol)) is not None]

# ---- template ---------------------------------------------------------------
if "--template" in opts:
    out = opts["--template"]; extra = ["solvent_smiles"] if "solvent_smiles" in rows[0] else []
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["row_index", smi_col] + extra + ["pred"])
        for i, r in test: w.writerow([i, r[smi_col]] + [r[c] for c in extra] + [""])
    print(f"wrote {len(test)} test rows ({pack} {split} target={tcol}) to {out}")
    if not (diagnose or pred_path): sys.exit(0)

# ---- diagnostics ------------------------------------------------------------
if diagnose:
    print(f"{pack} {split} target={tcol}  (labels from {os.path.basename(csv_path)})")
    by = {}
    for r in rows: by.setdefault(r[split], []).append(r)
    order = [s for s in ("train", "valid", "test") if s in by] + sorted(s for s in by if s not in ("train", "valid", "test"))
    lab = {}
    print(f"  {'fold':6} {'rows':>7} {'labelled':>9} {'mean':>10} {'sd':>10} {'min':>10} {'max':>10}")
    for s in order:
        v = [x for x in (fnum(r.get(tcol)) for r in by[s]) if x is not None]; lab[s] = v
        if v: print(f"  {s:6} {len(by[s]):7d} {len(v):9d} {mean(v):10.4g} {sd(v):10.4g} {min(v):10.4g} {max(v):10.4g}")
        else: print(f"  {s:6} {len(by[s]):7d} {0:9d}")
    tr, te = lab.get("train", []), lab.get("test", [])
    if len(tr) > 1 and te:
        print(f"  test-vs-train label mean shift: {(mean(te) - mean(tr)) / sd(tr):+.3f} train-sd units "
              f"(test mean {mean(te):.4g} vs train mean {mean(tr):.4g}, train sd {sd(tr):.4g})")
        mu = mean(tr)
        print(f"  predict-the-train-mean on test (n={len(te)}): RMSE={math.sqrt(mean([(t - mu) ** 2 for t in te])):.4g} "
              f"MAE={mean([abs(t - mu) for t in te]):.4g}")
    trs = {r[smi_col] for r in by.get("train", [])}
    nov = sum(1 for _, r in test if r[smi_col] in trs)
    print(f"  labelled test rows whose {smi_col} string exactly equals a train {smi_col}: {nov} of {len(test)}")
    if "solvent_smiles" in rows[0]:
        trp = {(r[smi_col], r["solvent_smiles"]) for r in by.get("train", [])}
        print(f"  ... same ({smi_col}, solvent_smiles) pair in train: {sum(1 for _, r in test if (r[smi_col], r['solvent_smiles']) in trp)} of {len(test)}")
    flags = [c for c in rows[0] if c == "disagreement_flag" or c.startswith("flag_")]
    for c in flags:
        k = sum(1 for _, r in test if r[c].strip().lower() in ("true", "1", "yes"))
        print(f"  labelled test rows with {c}=True: {k} of {len(test)}")
    if "n_values" in rows[0]:
        k = sum(1 for _, r in test if (fnum(r["n_values"]) or 0) > 1)
        print(f"  labelled test rows with n_values>1 (label is a median of several reports): {k} of {len(test)}")
    rng = [c for c in rows[0] if c.startswith("range_")]
    if rng: print(f"  (source-disagreement range column(s) in CSV: {', '.join(rng)})")
    if not pred_path: sys.exit(0)
    print()

# ---- scoring ----------------------------------------------------------------
with open(pred_path, newline="") as predictions_file:
    reader = csv.DictReader(predictions_file)
    columns = reader.fieldnames or []
    pcols = [c for c in columns if c and c.strip().lower().startswith("pred")]
    if columns.count("row_index") > 1:
        die(f"{pred_path}: duplicate row_index column")
    if len(pcols) != len(set(pcols)):
        die(f"{pred_path}: duplicate prediction column name")
    P = list(reader)
if not P: die(f"{pred_path} has no rows")
if not pcols: die(f"{pred_path}: no prediction column (need a column named `pred` or starting with `pred`)")
yby = {i: float(r[tcol]) for i, r in test}
if "row_index" in columns:
    predictions_by_row = {}
    ignored = 0
    for record_number, prediction in enumerate(P, start=1):
        raw_index = prediction["row_index"]
        try:
            row_index = int(raw_index)
        except (TypeError, ValueError):
            die(f"{pred_path}: invalid row_index {raw_index!r} in prediction record {record_number}; expected an integer")
        if row_index not in yby:
            ignored += 1
            continue
        if row_index in predictions_by_row:
            die(f"{pred_path}: duplicate row_index {row_index} for a labelled {split} test row")
        predictions_by_row[row_index] = prediction
    missing = [row_index for row_index in yby if row_index not in predictions_by_row]
    if missing:
        examples = ", ".join(str(row_index) for row_index in missing[:5])
        die(f"{pred_path}: missing predictions for {len(missing)} labelled {split} test rows (row_index: {examples})")
    if ignored:
        print(f"  warning: {ignored} row_index values are not labelled {split} test rows; ignored")
else:
    if len(P) != len(test): die(f"{pred_path} has {len(P)} rows but {pack} {split} has {len(test)} labelled test rows; add row_index or fix the order")
    predictions_by_row = dict(zip(yby, P))

y = list(yby.values())
res = []
for c in pcols:
    yh = []
    for row_index in yby:
        value = fnum(predictions_by_row[row_index].get(c))
        if value is None:
            die(f"{pred_path}: expected a finite prediction in column {c!r} for row_index {row_index}")
        yh.append(value)
    if len(y) >= 2:
        res.append(metrics(y, yh))

print(f"{pack} {split} target={tcol} ({len(test)} labelled test rows)")
if res:
    for c, m in zip(pcols, res):
        print(f"  yours [{c}] n={len(y)}: R2={m['R2']:.3f} RMSE={m['RMSE']:.4g} MAE={m['MAE']:.4g}")
else:
    for c in pcols:
        print(f"  {c}: fewer than 2 usable predictions, skipped")
if len(res) > 1:
    print(f"  yours mean +- sd over {len(res)} columns: " + "  ".join(
        f"{k}={mean([r[k] for r in res]):.3f} +- {sd([r[k] for r in res]):.3f}" for k in ("R2", "RMSE", "MAE")))
b5 = os.path.join(H, "BASELINES_5seed.csv"); shown = 0
if os.path.isfile(b5):
    for b in csv.DictReader(open(b5)):
        if b["pack"] == pack and b["split"] == split and b["target"] == tcol:
            shown += 1
            print(f"  baseline {b['model']} ({b['n_seeds']} RF seeds): R2={float(b['R2_mean']):.3f} +- {float(b['R2_sd']):.3f} "
                  f"RMSE={float(b['RMSE_mean']):.4g} +- {float(b['RMSE_sd']):.2g} (n_test={b['n_test']})")
if not shown:  # fall back to the single-seed table
    for b in csv.DictReader(open(os.path.join(H, "BASELINES.csv"))):
        if b["pack"] == pack and b["split"] == split and b["target"] == tcol:
            print(f"  baseline {b['model']} (1 seed): R2={b['R2']} RMSE={b['RMSE']} (n_test={b['n_test']})")
