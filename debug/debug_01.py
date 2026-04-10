#debug01
import numpy as np
import matplotlib.pyplot as plt
from pipeline_helpers.config import PADDING, CROP_LENGTH, CROP_HEIGHT, SEMILANDMARK_CURVE

def run_debug_01(
        img_name,
        semilandmarks_new_full,
        gt_semi_full,
        contour_full,
        crop_box,
        contour_pts_crop,
        idx_first,
        idx_last,
        subset_crop,
        contours_crop,
        output_dir,
        stem,
    ):
    # ── DIAGNOSTIC: inspect all point pairs ──────────────────────────────────────
    print(f"\n--- Diagnostic for {img_name} ---")
    print(f"  crop_box : {crop_box}")
    print(f"  GT  points ({len(gt_semi_full)}):   {gt_semi_full}")
    print(f"  PRED points ({len(semilandmarks_new_full)}): {semilandmarks_new_full.tolist()}")
    print()
    print(f"  {'i':>3}  {'GT_x':>8} {'GT_y':>8}  {'PRED_x':>8} {'PRED_y':>8}  {'Dist':>8}")
    print(f"  {'-'*3}  {'-'*8} {'-'*8}  {'-'*8} {'-'*8}  {'-'*8}")
    for i, pt_pred in enumerate(semilandmarks_new_full):
        if i < len(gt_semi_full):
            pt_gt = np.array(gt_semi_full[i], dtype=np.float32)
            dist  = np.sqrt(np.sum((pt_pred - pt_gt) ** 2))
            print(f"  {i:>3}  {pt_gt[0]:>8.1f} {pt_gt[1]:>8.1f}  {pt_pred[0]:>8.1f} {pt_pred[1]:>8.1f}  {dist:>8.2f}  <-- OUTLIER" if dist > 20 else
                f"  {i:>3}  {pt_gt[0]:>8.1f} {pt_gt[1]:>8.1f}  {pt_pred[0]:>8.1f} {pt_pred[1]:>8.1f}  {dist:>8.2f}")

    # ── DIAGNOSTIC: Dist_to_Contour outlier investigation ────────────────────────
    print(f"\n--- Contour Distance Diagnostic for {img_name} ---")
    print(f"  crop_box: {crop_box}")
    print(f"  contour_full range: x=[{contour_full[:,0].min():.1f}, {contour_full[:,0].max():.1f}]  "
        f"y=[{contour_full[:,1].min():.1f}, {contour_full[:,1].max():.1f}]")
    print(f"  semilandmarks_new_full range: x=[{semilandmarks_new_full[:,0].min():.1f}, {semilandmarks_new_full[:,0].max():.1f}]  "
        f"y=[{semilandmarks_new_full[:,1].min():.1f}, {semilandmarks_new_full[:,1].max():.1f}]")
    print()
    print(f"  {'i':>3}  {'PRED_x':>8} {'PRED_y':>8}  {'NEAR_x':>8} {'NEAR_y':>8}  {'Dist':>8}")
    print(f"  {'-'*3}  {'-'*8} {'-'*8}  {'-'*8} {'-'*8}  {'-'*8}")
    for i, pt_pred in enumerate(semilandmarks_new_full):
        dists     = np.sqrt(np.sum((contour_full - pt_pred) ** 2, axis=1))
        nearest_i = np.argmin(dists)
        nearest   = contour_full[nearest_i]
        dist      = dists[nearest_i]
        flag      = "  <-- OUTLIER" if dist > 10 else ""
        print(f"  {i:>3}  {pt_pred[0]:>8.1f} {pt_pred[1]:>8.1f}  "
            f"{nearest[0]:>8.1f} {nearest[1]:>8.1f}  {dist:>8.2f}{flag}")


    # ── DIAGNOSTIC: plot contour points around the outlier region ─────────────────

    # Focus on the x range around the outlier
    x_data = semilandmarks_new_full[:, 0]
    y_data = semilandmarks_new_full[:, 1]
    padding = 100
    x_min_plot, x_max_plot = x_data.min() - padding, x_data.max() + padding
    y_min_plot, y_max_plot = y_data.min() - padding, y_data.max() + padding

    # 2. Filter contour points to this region for the 'region' scatter
    mask = (
        (contour_full[:, 0] > x_min_plot) & (contour_full[:, 0] < x_max_plot) &
        (contour_full[:, 1] > y_min_plot) & (contour_full[:, 1] < y_max_plot)
    )
    region = contour_full[mask]

    fig, ax = plt.subplots(figsize=(12, 4))

    # All contour points in region
    ax.scatter(contour_full[:, 0], contour_full[:, 1], s=1, c='gray', label='Full contour')
    ax.scatter(region[:, 0], region[:, 1], s=10, c='blue', label='Contour in gap region')

    # Pred points 6, 7, 8
    for i in [6, 7, 8]:
        ax.scatter(*semilandmarks_new_full[i], s=80, zorder=5, label=f'pred_{i}')
        ax.annotate(f'pred_{i}', semilandmarks_new_full[i], textcoords="offset points", xytext=(0,8))

    #ax.set_xlim(2580, 2750)
    ax.set_xlim(x_min_plot, x_max_plot)
    ax.set_ylim(y_min_plot, y_max_plot)
    ax.invert_yaxis()  # match image coordinates
    ax.legend()
    ax.set_title(f'Contour gap investigation - {img_name}')
    plt.tight_layout()
    plt.savefig(str(output_dir / f"{stem}_gap_diagnostic.png"), dpi=150)
    plt.show()
    print(f"\nContour x-values sorted in gap region:\n{sorted(region[:, 0])}")

    print(f"  original contour points {len(contours_crop)}")
    print(f"  Total contour points : {len(contour_pts_crop)}")
    print(f"  idx_first: {idx_first}, idx_last: {idx_last}")
    print(f"  Subset length: {len(subset_crop)}")
    print(f"  Subset x range: {subset_crop[:,0].min():.0f} – {subset_crop[:,0].max():.0f}")
    print(f"  Subset y range: {subset_crop[:,1].min():.0f} – {subset_crop[:,1].max():.0f}")

    # Plot the full contour traversal order to see which way it goes
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(contour_pts_crop[:, 0], contour_pts_crop[:, 1], c=np.arange(len(contour_pts_crop)), 
            cmap='rainbow', s=2)
    ax.scatter(*contour_pts_crop[idx_first], s=100, c='red',   zorder=5, label=f'idx_first ({idx_first})')
    ax.scatter(*contour_pts_crop[idx_last],  s=100, c='blue',  zorder=5, label=f'idx_last ({idx_last})')
    plt.colorbar(ax.collections[0], ax=ax, label='contour traversal order')
    ax.invert_yaxis()
    ax.legend()
    ax.set_title('Contour traversal order — rainbow = index 0 to N')
    plt.tight_layout()
    plt.savefig(str(output_dir / f"{stem}_contour_traversal.png"), dpi=150)
    plt.show()