"""
GPA figures for automated semilandmark detection.

Generates three figures from the per-method results CSVs:
  1. landmarks_by_method.png        - aligned landmarks, manual vs each method
  2. tps_deformation_grids.png      - thin-plate-spline deformation grids
  3. error_vs_biological_signal.png - error per specimen vs biological spread
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") #make PNG directly
import matplotlib.pyplot as plt

##CONFIG
DATA_DIR = "C:/Users/korbi/Desktop/A_Master_Thesis/results_comparison" # folder containing <method>_results.csv
OUT_DIR  = "C:/Users/korbi/Desktop/A_Master_Thesis/GPA/plots"

# Method names. Each must have a file named f"{method}_results.csv" in DATA_DIR.
METHODS = ["rembg", "knn", "mlp", "vit", "cnn", "gnn"]

# Mrembg landmarks are stored in REVERSED index, remove if upstream transformation may be fixed.
REVERSED_METHODS = ["rembg"]

P = 10                         # semilandmarks per specimen
MAG = 3                        # TPS grid displacement magnification (visual only)
DPI = 160
SORT_BY_ERROR = True           # order panels best -> worst automatically

# Column names in the CSV files.
COL_IMG, COL_IDX = "img_name", "index"
COL_NEW = ("X_new", "Y_new")   # predicted
COL_OLD = ("X_old", "Y_old")   # manual ground truth


os.makedirs(OUT_DIR, exist_ok=True)

frames = {}
for m in METHODS:
    path = os.path.join(DATA_DIR, f"{m}_results.csv")
    d = pd.read_csv(path) #load CSV files
    if m in REVERSED_METHODS: # do the magic for rembg
        d[COL_IDX] = (P - 1) - d[COL_IDX]
    frames[m] = d.sort_values([COL_IMG, COL_IDX]).reset_index(drop=True)

imgs = sorted(frames[METHODS[0]][COL_IMG].unique())
print(f"Loaded {len(METHODS)} methods x {len(imgs)} specimens x {P} landmarks")

# Correspondence check: ground truth must agree across methods once reversal is fixed.
# AI added this block and I think it is neat - why not. It does not hurt to have a sanity check.
ref = frames[METHODS[0]][list(COL_OLD)].values.astype(float)
for m in METHODS:
    v = frames[m][list(COL_OLD)].values.astype(float)
    if not np.allclose(v, ref):
        raise ValueError(
            f"Ground truth in '{m}' does not match '{METHODS[0]}'.\n"
            f"Landmark correspondence is broken - check REVERSED_METHODS.")
print("OK: ground truth consistent across all methods.")

# Give a method and an image, return the landmark coordinates as a numpy array.
def cfg(m, img, cols):
    g = frames[m]
    g = g[g[COL_IMG] == img].sort_values(COL_IDX)
    return g[list(cols)].values.astype(float)


labels, mats = [], []
for img in imgs:
    labels.append((img, "manual")); mats.append(cfg(METHODS[0], img, COL_OLD))
    for m in METHODS:
        labels.append((img, m)); mats.append(cfg(m, img, COL_NEW))
A = np.stack(mats)
lab = pd.DataFrame(labels, columns=["img", "method"])

##GPA
def center(X): return X - X.mean(0)
def csize(X): return np.sqrt(((X - X.mean(0)) ** 2).sum()) #centroid size

# procrustes rotation of X to Y, preserving scale and translation
def rot_to(X, Y):
    U_, _, Vt = np.linalg.svd(X.T @ Y)
    R = U_ @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U_ @ Vt
    return X @ R

# Preprocessing of every shape: do the unit centroid size scaling
B = np.stack([center(X) / csize(X) for X in A])
mean = B[0].copy()

for it in range(100): #arbitrary max iterations
    B = np.stack([rot_to(X, mean) for X in B])
    nm = B.mean(0)
    nm = center(nm) / csize(nm)
    if np.sqrt(((nm - mean) ** 2).sum()) < 1e-12: #arbitrary threshold for convergence
        mean = nm
        break
    mean = nm
print(f"GPA converged in {it + 1} iterations")


def get(img, m):
    return B[lab.index[(lab.img == img) & (lab.method == m)][0]]


manual_all = np.stack([get(i, "manual") for i in imgs])
mcons = manual_all.mean(0)
# KEY BENCHMARK: biological spread = mean distance of each manual specimen to the consensus
bio = float(np.mean([np.sqrt(((x - mcons) ** 2).sum()) for x in manual_all]))

cons = {m: np.stack([get(i, m) for i in imgs]).mean(0) for m in METHODS} # consensus per method
errs = {m: np.array([np.sqrt(((get(i, m) - get(i, "manual")) ** 2).sum()) for i in imgs])
        for m in METHODS}
err = {m: errs[m].mean() for m in METHODS}

# automatic ordering
order = sorted(METHODS, key=lambda m: err[m]) if SORT_BY_ERROR else list(METHODS)

print(f"\nBiological spread (manual, mean dist to consensus): {bio:.4f}")
for m in order:
    print(f"  {m:6s} ProcD = {err[m]:.4f}  ({100 * err[m] / bio:.0f}% of biological spread)")

GREEN, RED, ORANGE, DARK, GREY = "#3B6D11", "#A32D2D", "#D85A30", "#3d3d3a", "#D3D1C7"


def grid_shape(n):
    ncol = 3 if n >= 3 else n
    return int(np.ceil(n / ncol)), ncol


# ============ FIGURE 1: landmark grid ============
nrow, ncol = grid_shape(len(order))
fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.25 * nrow), squeeze=False)
for ax, m in zip(axes.ravel(), order):
    for i in imgs:
        ax.plot(*get(i, m).T, "-", color=GREY, lw=0.5, alpha=0.5, zorder=1)
    ax.plot(*mcons.T, "-o", color=DARK, lw=2, ms=7, zorder=3,
            label="manual (consensus)", mfc="white", mew=1.5)
    ax.plot(*cons[m].T, "-o", color=ORANGE, lw=1.5, ms=5, zorder=4, label=f"{m} (consensus)")
    span = mcons[:, 1].max() - mcons[:, 1].min()
    for k in range(P):
        ax.annotate("", xy=cons[m][k], xytext=mcons[k],
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2, alpha=0.9), zorder=5)
        ax.text(mcons[k][0], mcons[k][1] - 0.08 * span, str(k),
                ha="center", fontsize=7, color="#73726c")
    ratio = 100 * err[m] / bio
    ax.set_title(f"{m}   ProcD={err[m]:.4f}  ({ratio:.0f}% of biological spread)",
                 fontsize=10, color=GREEN if ratio < 100 else RED)
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
for ax in axes.ravel()[len(order):]:
    ax.axis("off")
fig.suptitle("Aligned semilandmarks: each method vs manual ground truth\n"
             "grey = all specimens for that method | arrows = mean displacement from manual",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT_DIR, "landmarks_by_method.png"), dpi=DPI)
plt.close(fig)
print("\nwrote landmarks_by_method.png")


# ============ FIGURE 2: TPS deformation grids ============
def tps_warp(src, dst, pts):
    p = src.shape[0]
    d2 = ((src[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    Kk = np.where(d2 > 0, d2 * np.log(np.where(d2 > 0, d2, 1)), 0)
    Pm = np.hstack([np.ones((p, 1)), src])
    L = np.zeros((p + 3, p + 3))
    L[:p, :p] = Kk; L[:p, p:] = Pm; L[p:, :p] = Pm.T
    W = np.linalg.solve(L + np.eye(p + 3) * 1e-9, np.vstack([dst, np.zeros((3, 2))]))
    dg = ((pts[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    Kg = np.where(dg > 0, dg * np.log(np.where(dg > 0, dg, 1)), 0)
    return Kg @ W[:p] + np.hstack([np.ones((len(pts), 1)), pts]) @ W[p:]


rx = mcons[:, 0].max() - mcons[:, 0].min()
ry = mcons[:, 1].max() - mcons[:, 1].min()
pad = 0.25 * max(rx, ry)
xs = np.linspace(mcons[:, 0].min() - pad, mcons[:, 0].max() + pad, 22)
ys = np.linspace(mcons[:, 1].min() - pad, mcons[:, 1].max() + pad, 14)

fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.25 * nrow), squeeze=False)
for ax, m in zip(axes.ravel(), order):
    tgt = mcons + (cons[m] - mcons) * MAG
    for y in ys:
        ax.plot(*tps_warp(mcons, tgt, np.column_stack([xs, np.full_like(xs, y)])).T,
                color="#B4B2A9", lw=0.6)
    for x in xs:
        ax.plot(*tps_warp(mcons, tgt, np.column_stack([np.full_like(ys, x), ys])).T,
                color="#B4B2A9", lw=0.6)
    ax.plot(*mcons.T, "-o", color=DARK, lw=1.5, ms=6, mfc="white", mew=1.5, zorder=3)
    ax.plot(*tgt.T, "o", color=ORANGE, ms=5, zorder=4)
    ratio = 100 * err[m] / bio
    ax.set_title(f"{m}   ({ratio:.0f}% of biological spread)",
                 fontsize=10, color=GREEN if ratio < 100 else RED)
    ax.set_aspect("equal"); ax.axis("off")
for ax in axes.ravel()[len(order):]:
    ax.axis("off")
fig.suptitle(f"TPS deformation grids: manual consensus warped to each method "
             f"(displacement magnified {MAG}x)\n"
             "a flat, even grid = the method preserves shape | bending = systematic distortion",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT_DIR, "tps_deformation_grids.png"), dpi=DPI)
plt.close(fig)
print("wrote tps_deformation_grids.png")


# ============ FIGURE 3: error vs biological signal ============
fig, ax = plt.subplots(figsize=(1.5 * len(order) + 2.5, 5))
vals = [errs[m] for m in order]
try:
    bp = ax.boxplot(vals, tick_labels=order, patch_artist=True, widths=0.6)
except TypeError:                       # matplotlib < 3.9 uses `labels`
    bp = ax.boxplot(vals, labels=order, patch_artist=True, widths=0.6)
for patch, m in zip(bp["boxes"], order):
    patch.set_facecolor("#C0DD97" if err[m] < bio else "#F7C1C1")
    patch.set_edgecolor("#5F5E5A")
for med in bp["medians"]:
    med.set_color(DARK)
ax.axhline(bio, color="#185FA5", ls="--", lw=1.8)
ax.annotate(f"biological\nspread\n({bio:.4f})",              # computed, not hardcoded
            xy=(1.01, bio), xycoords=("axes fraction", "data"),
            color="#185FA5", va="center", fontsize=9)
ax.set_ylabel("Procrustes distance from manual ground truth")
ax.set_title("Digitization error per specimen, against the biological signal\n"
             "green = error below the signal (usable) | red = error exceeds it", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "error_vs_biological_signal.png"), dpi=DPI)
plt.close(fig)
print("wrote error_vs_biological_signal.png")


# ============ FIGURE 4: true local bending (non-affine component only) ============
# A TPS deformation splits into:
#   affine part     - uniform tilt / shear / stretch of the whole plane
#   non-affine part - genuine LOCAL bending
# Figure 2 shows both summed, which is why those grids look tilted: with landmarks
# this close to collinear (~13:1 aspect), the affine part dominates the picture and
# mostly reflects extrapolation away from the data. This figure strips it out and
# shows only real local distortion.

GRID_PAD = 0.15   # tighter than fig 2: the TPS kernel extrapolates poorly far from
                  # the landmarks, so keep the grid near where the data actually is.


def tps_components(src, dst, pts):
    """Return (non_affine_map, bending_energy).

    non_affine_map : pts displaced by the kernel term only, affine held at identity.
    bending_energy : standard non-affine magnitude. 0 = the change is purely affine.
                     Scales with MAG**2, so always evaluate it unmagnified.
    """
    p = src.shape[0]
    d2 = ((src[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    K = np.where(d2 > 0, d2 * np.log(np.where(d2 > 0, d2, 1)), 0)
    Pm = np.hstack([np.ones((p, 1)), src])
    L = np.zeros((p + 3, p + 3))
    L[:p, :p] = K; L[:p, p:] = Pm; L[p:, :p] = Pm.T
    W = np.linalg.solve(L + np.eye(p + 3) * 1e-9, np.vstack([dst, np.zeros((3, 2))]))
    Wk = W[:p]
    dg = ((pts[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    Kg = np.where(dg > 0, dg * np.log(np.where(dg > 0, dg, 1)), 0)
    be = float(np.trace(Wk.T @ K @ Wk))
    return pts + Kg @ Wk, be


# Method bending, unmagnified.
bend = {m: tps_components(mcons, cons[m], mcons)[1] for m in METHODS}

# BENCHMARK, same logic as FIGURE 3: how much local bending separates real specimens?
# A method is acceptable if it bends the shape less than biology itself does.
bio_bend = float(np.mean([tps_components(mcons, x, mcons)[1] for x in manual_all]))

print(f"\nBiological bending energy (consensus -> each real specimen): {bio_bend:.6f}")
for m in sorted(METHODS, key=lambda m: bend[m]):
    pct = 100 * bend[m] / bio_bend
    print(f"  {m:6s} bending = {bend[m]:.6f}  ({pct:6.1f}% of biological bending)"
          f"{'' if bend[m] < bio_bend else '   <-- EXCEEDS'}")

pad4 = GRID_PAD * max(rx, ry)
xs4 = np.linspace(mcons[:, 0].min() - pad4, mcons[:, 0].max() + pad4, 24)
ys4 = np.linspace(mcons[:, 1].min() - pad4, mcons[:, 1].max() + pad4, 16)

order4 = sorted(METHODS, key=lambda m: bend[m]) if SORT_BY_ERROR else list(METHODS)

fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.25 * nrow), squeeze=False)
for ax, m in zip(axes.ravel(), order4):
    tgt = mcons + (cons[m] - mcons) * MAG          # same magnification as fig 2
    for y in ys4:
        w, _ = tps_components(mcons, tgt, np.column_stack([xs4, np.full_like(xs4, y)]))
        ax.plot(*w.T, color="#B4B2A9", lw=0.6)
    for x in xs4:
        w, _ = tps_components(mcons, tgt, np.column_stack([np.full_like(ys4, x), ys4]))
        ax.plot(*w.T, color="#B4B2A9", lw=0.6)
    # Landmarks under the NON-AFFINE part only. Plotting the full target here would
    # mix the affine component back into a figure whose whole point is to exclude it.
    lm_na, _ = tps_components(mcons, tgt, mcons)
    ax.plot(*mcons.T, "-o", color=DARK, lw=1.5, ms=6, mfc="white", mew=1.5, zorder=3,
            label="manual (consensus)")
    ax.plot(*lm_na.T, "o", color=ORANGE, ms=5, zorder=4, label=f"{m}, non-affine only")
    pct = 100 * bend[m] / bio_bend
    ax.set_title(f"{m}   bending = {bend[m]:.5f}   ({pct:.0f}% of biological bending)",
                 fontsize=10, color=GREEN if bend[m] < bio_bend else RED)
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
for ax in axes.ravel()[len(order4):]:
    ax.axis("off")
fig.suptitle(f"True local bending: non-affine component only (magnified {MAG}x)\n"
             "affine tilt/shear removed | flat square grid = no local distortion | "
             "green = bends less than biology does",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT_DIR, "local_bending_by_method.png"), dpi=DPI)
plt.close(fig)
print("wrote local_bending_by_method.png")

print(f"\nAll figures written to: {os.path.abspath(OUT_DIR)}")
