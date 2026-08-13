# %% [markdown]
# # Step 4 -- the minimal decisive experiment
#
# Train ResNet-50 on **A-full** (annotations intact). Test on **A-only**
# (anatomy destroyed, annotations kept). If A-only clears chance on two
# datasets, the shortcut exists and the project proceeds. If it does not, the
# central hypothesis is refuted for about an hour of GPU time and the proposal's
# own recommendation is to redirect to D2.
#
# Runs on **Kaggle**, Colab, or any local GPU box -- the environment is detected
# below. Paste these cells into a Kaggle notebook, or convert with
# `jupytext --to ipynb notebooks/step4_gate.py`.
#
# ## Kaggle setup, once
#
# 1. Run Stages A and B locally (`scripts/run_stage_ab.sh`). They are CPU-only.
# 2. `python scripts/prepare_kaggle_upload.py` and follow the printed commands.
#    That creates two private Kaggle datasets: one with the packaged variant
#    shards, one with `src/` + `configs/`.
# 3. New notebook -> *Add Input* -> both datasets.
# 4. **Accelerator: GPU T4 x2** (or P100).
# 5. **Internet: ON.** Off by default, and torchvision's ImageNet weights are
#    downloaded at model-build time. If your account cannot enable internet, see
#    the offline-weights cell below.
# 6. Run interactively to check the sanity cell, then *Save Version -> Save &
#    Run All (Commit)* for the real run -- interactive sessions die on idle,
#    committed ones get the full 12 hours.

# %%
# --- environment detection ------------------------------------------------
import os
import sys
from pathlib import Path


def detect_environment() -> str:
    if Path("/kaggle/input").exists():
        return "kaggle"
    if "google.colab" in sys.modules or Path("/content").exists():
        return "colab"
    return "local"


ENV = detect_environment()
print("environment:", ENV)


def show_inputs(max_depth: int = 5) -> None:
    """Print what is actually mounted. Called on failure, because 'not found'
    without the directory listing is the least useful error there is.

    Kaggle does not guarantee a mount depth: datasets have been seen at
    /kaggle/input/<slug>/ and at /kaggle/input/datasets/<user>/<slug>/, so the
    listing has to recurse rather than assume.
    """
    root = Path("/kaggle/input")
    if not root.exists():
        print(f"{root} does not exist")
        return

    def walk(path: Path, depth: int, prefix: str) -> None:
        if depth > max_depth:
            return
        children = sorted(path.glob("*"))[:12]
        for child in children:
            marker = "/" if child.is_dir() else ""
            print(f"{prefix}{child.name}{marker}")
            # stop descending once we hit the payload
            if child.is_dir() and child.name not in {"src", "configs", "tests", "asa"}:
                walk(child, depth + 1, prefix + "    ")

    print(f"{root}:")
    walk(root, 1, "  ")


def find_repo() -> Path:
    """Locate the directory containing src/asa, wherever it was mounted.

    Searches recursively rather than at a fixed depth, because Kaggle's layout
    depends on how the dataset was uploaded, and unpacks a zipped src/ if that
    is what the dataset contains (`--dir-mode zip` can leave archives rather
    than directories).
    """
    candidates = [Path.cwd(), *Path.cwd().parents]
    if ENV == "kaggle":
        candidates += sorted(Path("/kaggle/input").glob("*"))
    if ENV == "colab":
        candidates += [Path("/content/drive/MyDrive/annotation-shortcut-audit")]
    for candidate in candidates:
        if (candidate / "src" / "asa" / "__init__.py").exists():
            return candidate

    # Recursive fallback: find asa/__init__.py at any depth under the inputs.
    if ENV == "kaggle":
        for marker in Path("/kaggle/input").rglob("asa/__init__.py"):
            repo = marker.parent.parent.parent  # <repo>/src/asa/__init__.py
            if (repo / "src" / "asa").exists():
                print(f"found code at a non-standard depth: {repo}")
                return repo

        # Last resort: a zipped src/ that Kaggle did not expand.
        import zipfile

        for archive in Path("/kaggle/input").rglob("src.zip"):
            target = Path("/kaggle/working/asa-code")
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target / "src")
            if (target / "src" / "asa" / "__init__.py").exists():
                print(f"extracted {archive} -> {target}")
                for extra in ("configs.zip", "tests.zip"):
                    found = next(Path("/kaggle/input").rglob(extra), None)
                    if found:
                        with zipfile.ZipFile(found) as zf:
                            zf.extractall(target / extra.removesuffix(".zip"))
                return target

    show_inputs()
    raise FileNotFoundError(
        "could not find src/asa.\n"
        "On Kaggle: right panel -> Add Input -> Your Datasets -> add BOTH of\n"
        "  <your-username>/annotation-shortcut-audit-code\n"
        "  <your-username>/asa-packs-224\n"
        "then re-run this cell. The listing above shows what is currently mounted."
    )


def find_packages() -> dict[str, Path]:
    """Locate packaged variant stores, wherever they were mounted.

    The dataset name is read from index.csv's own ``dataset`` column rather than
    inferred from the directory path -- Kaggle rearranges and unzips uploads, and
    a name guessed from a path would silently mislabel a whole run.
    """
    import csv

    roots = sorted(Path("/kaggle/input").glob("*")) if ENV == "kaggle" else [REPO / "work"]
    found: dict[str, Path] = {}
    for root in roots:
        for meta in sorted(root.rglob("meta.json")):
            pack = meta.parent
            index_csv = pack / "index.csv"
            if not index_csv.exists():
                continue
            with open(index_csv, newline="") as handle:
                first = next(csv.DictReader(handle), None)
            name = (first or {}).get("dataset") or pack.parent.name
            found.setdefault(name, pack)
    return found


REPO = find_repo()
sys.path.insert(0, str(REPO / "src"))
CONFIG_ROOT = REPO / "configs"

OUT_ROOT = Path("/kaggle/working/runs") if ENV == "kaggle" else REPO / "runs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PACKS = find_packages()
print("repo:   ", REPO)
print("output: ", OUT_ROOT)
print("packages:")
for name, path in PACKS.items():
    print(f"   {name:16s} {path}")

# %%
# --- offline weights (only needed if Kaggle internet is OFF) --------------
# torchvision downloads ImageNet weights on first use. With internet off that
# raises a URLError at model-build time, several minutes into the run.
#
# Workaround: add a Kaggle dataset containing the checkpoint and point TORCH_HOME
# at a writable copy of it. The file torchvision wants for ResNet-50 IMAGENET1K_V2
# is `resnet50-11ad3fa6.pth`, and it belongs in $TORCH_HOME/hub/checkpoints/.
import shutil
import urllib.request

TORCH_HOME = Path("/kaggle/working/torch") if ENV == "kaggle" else Path.home() / ".cache/torch"
CHECKPOINTS = TORCH_HOME / "hub" / "checkpoints"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(TORCH_HOME)

if ENV == "kaggle":
    for source in Path("/kaggle/input").rglob("*.pth"):
        target = CHECKPOINTS / source.name
        if not target.exists():
            shutil.copy(source, target)
            print("staged offline weight:", source.name)

# A plain GET of the host root can 403 even when downloads work, which reports
# a false negative right before a run that would have succeeded. Test the thing
# we actually need: a TCP connection to the checkpoint host.
import socket

try:
    socket.create_connection(("download.pytorch.org", 443), timeout=5).close()
    print("internet: ON")
except OSError:
    print("internet: OFF --", len(list(CHECKPOINTS.glob('*.pth'))), "staged checkpoint(s)")
    if not any(CHECKPOINTS.glob("*.pth")):
        print(
            "  WARNING: no internet and no staged weights. Either enable internet in\n"
            "  the notebook settings, or add a dataset containing resnet50-11ad3fa6.pth,\n"
            "  or set model.pretrained: none in the config (which changes the experiment)."
        )

# %%
import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("NO GPU -- check the Accelerator setting; this will be unusably slow on CPU")

from asa.config import load_experiment  # noqa: E402
from asa.data.packaging import load_package  # noqa: E402
from asa.train.gate import decide, evaluate_gate, write_gate_report  # noqa: E402
from asa.train.matrix import run_matrix  # noqa: E402

# The gate needs two datasets: a shortcut seen on exactly one annotation
# convention is a quirk, not a finding.
GATE_DATASETS = [name for name in ("fetal_planes", "bus_bra") if name in PACKS]
assert len(GATE_DATASETS) >= 2, f"need two packaged datasets, found {list(PACKS)}"
print("gate datasets:", GATE_DATASETS)

# %% [markdown]
# ## 1. Sanity-check the packages before spending GPU time
#
# The variants must differ from each other in the expected direction: A-removed
# should change pixels inside the annotation mask and nowhere else, and A-only
# should change everything *except* those pixels. If that is not true the
# packaging step is broken and every downstream number is meaningless.
#
# The variants are built at native resolution, where the invariant is exact (see
# `tests/test_pipeline.py`). Here they have been through an INTER_AREA resize,
# which blends across the mask boundary, so the check is a ratio rather than an
# equality -- glyph strokes are 2-3 px and a 224 px downsample smears their
# edges by construction.

# %%
import numpy as np

for name in GATE_DATASETS:
    index, arrays, meta = load_package(PACKS[name])
    mask = np.asarray(arrays["annotation"][:64]) > 127
    full = np.asarray(arrays["a_full"][:64]).astype(np.int16)
    removed = np.asarray(arrays["a_removed"][:64]).astype(np.int16)
    only = np.asarray(arrays["a_only"][:64]).astype(np.int16)
    mask3 = np.repeat(mask[..., None], full.shape[-1], axis=-1) if full.ndim == 4 else mask

    removed_in = np.abs(full - removed)[mask3].mean()
    removed_out = np.abs(full - removed)[~mask3].mean()
    only_in = np.abs(full - only)[mask3].mean()
    only_out = np.abs(full - only)[~mask3].mean()
    print(
        f"{name}: n={meta['n']} size={meta['size']} "
        f"{'gray' if meta['grayscale'] else 'rgb'} mask={mask.mean() * 100:.3f}% of pixels\n"
        f"    A-removed  changed inside mask={removed_in:6.2f}  outside={removed_out:6.3f}\n"
        f"    A-only     changed inside mask={only_in:6.2f}  outside={only_out:6.2f}"
    )
    assert removed_in > 10 * max(removed_out, 0.01), "A-removed is not concentrated on the annotation mask"
    assert only_out > 2 * only_in, "A-only did not preserve the annotations relative to the anatomy"
    if mask.mean() < 1e-4:
        print("    WARNING: almost no annotation survives at this resolution; nothing to measure here")
print("package sanity checks passed")

# %% [markdown]
# ## 2. Run the gate
#
# 3 folds x 2 seeds x 2 datasets = 12 trainings. Roughly 1-2 hours on a T4,
# well inside a 12-hour committed session. `run_matrix` writes results after
# every trained model and skips completed cells on re-entry, so re-running this
# cell after a timeout continues rather than restarting.

# %%
verdicts = []
for name in GATE_DATASETS:
    cfg = load_experiment(CONFIG_ROOT / "experiments" / "step4_gate.yaml", CONFIG_ROOT, {"dataset": name})
    index, arrays, _ = load_package(PACKS[name])
    results = run_matrix(
        cfg,
        index,
        arrays,
        out_dir=OUT_ROOT / "step4_gate" / name,
        folds=[0, 1, 2],
        resume=True,
    )
    verdicts.append(evaluate_gate(results, name, cfg.dataset.task))
    print(verdicts[-1])

# %%
# Guard against summarising an empty run: after a session restart, `verdicts`
# is [] until the training cell above has actually executed.
assert verdicts, (
    "verdicts is empty -- the training cell has not run in this session. "
    "Re-run it before summarising; otherwise the report below would describe nothing."
)
decision = decide(verdicts, min_datasets_passing=2)
write_gate_report(decision, OUT_ROOT / "step4_gate")
print(decision["recommendation"])

import pandas as pd  # noqa: E402

pd.DataFrame([v.as_dict() for v in verdicts])[
    ["dataset", "a_full_score", "a_removed_score", "a_only_score", "chance", "shortcut_magnitude", "passed"]
]

# %% [markdown]
# ## 3. Read the result honestly
#
# * **A-only above chance on both datasets** -- the shortcut exists. Proceed to
#   Step 5 (pre-registration) *before* the full matrix, so that H1-H4 and the
#   analysis plan are fixed before the numbers are known.
# * **A-only at chance** -- the hypothesis is refuted at a cost of roughly one
#   GPU-hour. This is a publishable null: it retires a concern BUSClean stated
#   and never tested. Redirect to D2.
# * **The equal-area control drops as much as A-removed** -- the effect is pixel
#   removal, not annotations. `evaluate_gate` fails the dataset on this
#   automatically; it is reviewer attack R5 and it refutes the hypothesis
#   regardless of what A-only says.
#
# One thing the gate does *not* establish: how much of the headline A-full
# number is annotation-derived. That is the full matrix. Do not report a gate
# result as if it were the audit.
#
# On Kaggle everything under `/kaggle/working` is saved as notebook output when
# the version finishes -- download `runs/step4_gate/` from the Output tab.

# %% [markdown]
# ---
# ## Generic runner for follow-up experiments (4b, 4c, ...)
#
# Use this INSTEAD of the gate cell above for any experiment that is not the
# Step-4 gate. The gate cell calls `evaluate_gate` inside its loop, and that
# function is defined only for models trained on A-full -- so a step4b run
# (trained on A-only) raises part-way through and never reaches the datasets
# after the first. This cell has no gate logic at all.

# %%
import pandas as pd

from asa.train.matrix import run_matrix, summarise_matrix


def run_experiment(experiment: str, datasets, tag: str, folds=(0, 1, 2), overrides=None):
    """Train + evaluate one experiment across datasets, and summarise it.

    Deliberately returns the raw frame as well as the summary: the summary is
    for reading, the frame is what gets analysed and reported.
    """
    frames = []
    for name in datasets:
        cfg = load_experiment(
            CONFIG_ROOT / "experiments" / f"{experiment}.yaml",
            CONFIG_ROOT,
            {"dataset": name, **(overrides or {})},
        )
        if name not in PACKS:
            print(f"SKIP {name}: not in the mounted packages")
            continue
        index, arrays, _ = load_package(PACKS[name])
        frame = run_matrix(
            cfg, index, arrays,
            out_dir=OUT_ROOT / tag / name,
            folds=list(folds),
            resume=True,
        )
        frames.append(frame)

        real = frame[~frame.get("is_contrast", pd.Series(False, index=frame.index)).fillna(False).astype(bool)]
        print(f"\n=== {name} / {experiment} ===")
        print(
            real.groupby("test_variant")
            .agg(bal_acc=("balanced_accuracy", "mean"),
                 sd=("balanced_accuracy", "std"),
                 chance=("chance_balanced_accuracy", "mean"),
                 n=("balanced_accuracy", "size"))
            .round(4)
            .to_string()
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --- 4b: is the annotation signal learnable at all? -----------------------
# results_4b = run_experiment("step4b_available_vs_taken", ["fetal_planes", "bus_bra"], tag="step4b")

# --- 4c: positive control, full strength first ----------------------------
# Read `a_full` minus `a_full__nomarker`: that is what removing a shortcut the
# model DOES use costs, and it calibrates the real annotation-removal drop of
# -0.0002. Then read `a_only` against the real 0.176.
# results_4c = run_experiment(
#     "step4c_positive_control", ["fetal_planes"], tag="step4c/strength_1.0",
#     overrides={"synthetic": {"enabled": True, "strength": 1.0}},
# )
