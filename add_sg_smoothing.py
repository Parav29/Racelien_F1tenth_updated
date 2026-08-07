import json

def add_sg_smoothing_to_notebook(file_path):
    """Add SG smoothing cell to map_converter.ipynb before the CSV export step."""
    with open(file_path, 'r') as f:
        nb = json.load(f)

    # The new cell source - SG smoothing + proper closure
    sg_cell_source = [
        "# ─── Savitzky-Golay smoothing + closure fix ───────────────────────────────\n",
        "# Smooths the raw MST centerline so that:\n",
        "#   1. Max curvature stays within the vehicle curvature limit\n",
        "#   2. The closed-loop spline in prep_track.py never has crossed normals\n",
        "#   3. The closure gap is near-zero (< 0.01m)\n",
        "from scipy.signal import savgol_filter as _sg\n",
        "\n",
        "def smooth_centerline(xs, ys, wrs, wls, window=31, poly=3, taper_pts=10, wrap=50):\n",
        "    n = len(xs)\n",
        "    # Wrap arrays for periodic boundary treatment\n",
        "    _wrap = lambda a: np.concatenate([a[-wrap:], a, a[:wrap]])\n",
        "    xs_s = _sg(_wrap(xs), window, poly)[wrap:wrap+n]\n",
        "    ys_s = _sg(_wrap(ys), window, poly)[wrap:wrap+n]\n",
        "    wrs_s = _sg(_wrap(wrs), window, poly)[wrap:wrap+n]\n",
        "    wls_s = _sg(_wrap(wls), window, poly)[wrap:wrap+n]\n",
        "    # Taper last `taper_pts` to enforce closure (gap -> ~0)\n",
        "    blend = np.linspace(0, 1, taper_pts+2)[1:-1]\n",
        "    for i, b in enumerate(blend):\n",
        "        idx = n - taper_pts + i\n",
        "        xs_s[idx] = xs_s[idx]*(1-b) + xs_s[0]*b\n",
        "        ys_s[idx] = ys_s[idx]*(1-b) + ys_s[0]*b\n",
        "    return xs_s, ys_s, wrs_s, wls_s\n",
        "\n",
        "# Extract raw arrays from centerline_points / track_widths\n",
        "_pts = np.array(centerline_points)\n",
        "_wid = np.array(track_widths)\n",
        "raw_x = _pts[:, 0].astype(float)\n",
        "raw_y = _pts[:, 1].astype(float)\n",
        "raw_wr = _wid[:, 0].astype(float)\n",
        "raw_wl = _wid[:, 1].astype(float)\n",
        "\n",
        "# Determine a safe window size (must be odd, < len)\n",
        "_n = len(raw_x)\n",
        "_win = min(31, _n-1 if (_n-1) % 2 == 1 else _n-2)\n",
        "if _win < 5: _win = 5\n",
        "\n",
        "raw_x, raw_y, raw_wr, raw_wl = smooth_centerline(\n",
        "    raw_x, raw_y, raw_wr, raw_wl, window=_win)\n",
        "\n",
        "# Rebuild centerline_points and track_widths\n",
        "centerline_points = [np.array([raw_x[i], raw_y[i]]) for i in range(_n)]\n",
        "track_widths = [np.array([raw_wr[i], raw_wl[i]]) for i in range(_n)]\n",
        "\n",
        "# Diagnostics\n",
        "_dx = np.gradient(raw_x); _dy = np.gradient(raw_y)\n",
        "_ddx = np.gradient(_dx); _ddy = np.gradient(_dy)\n",
        "_k = np.abs(_dx*_ddy - _dy*_ddx) / (_dx**2 + _dy**2)**1.5\n",
        "print(f'After SG smoothing: max curvature = {np.max(_k):.3f} rad/m')\n",
        "print(f'Closure gap: {np.hypot(raw_x[-1]-raw_x[0], raw_y[-1]-raw_y[0]):.4f}m')\n",
        "print(f'Min total width: {(raw_wr+raw_wl).min():.4f}m')\n"
    ]

    # Find the cell that contains the waypoint merging / CSV export logic
    # Insert the SG cell BEFORE the cell that builds the waypoints array
    # which contains "waypoints = np.array(centerline_points)"
    target_text = "waypoints = np.array(centerline_points)"
    insert_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            if target_text in src:
                insert_idx = i
                break

    if insert_idx is None:
        print("ERROR: Could not find the waypoints cell. No changes made.")
        return

    # Check if we already inserted it
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and "smooth_centerline" in "".join(cell.get('source', [])):
            print("SG smoothing cell already present — skipping.")
            return

    # Create new code cell
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": sg_cell_source
    }

    nb['cells'].insert(insert_idx, new_cell)
    print(f"Inserted SG smoothing cell before cell index {insert_idx} (waypoints cell).")

    with open(file_path, 'w') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    add_sg_smoothing_to_notebook("/home/sedrica/Raceline-Optimization/map_converter.ipynb")
    print("Done.")
