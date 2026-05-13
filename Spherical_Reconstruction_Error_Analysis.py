r"""
Spherical_Reconstruction_Error_Analysis
=======================================

Estimates the numerical error introduced by the pixel-based midpoint
Jacobian integration used in `Spherical_Reconstruction_2.py`.

Approach
--------
Replicate the integration scheme from `Spherical_Reconstruction_2.py`:

    A_recon = sum over pixels (i,j) inside the integration domain of
              detJ(R, x_i, z_j) * (pixel area)

with

    detJ(R, x, z) = R / sqrt(R^2 - x^2 - z^2)

Two integration domains are evaluated for each grid resolution N:

  * Full disk     -> reconstructs the full half unit sphere
                     (analytical area = 2*pi)
  * Cubed-sphere  -> reconstructs one of the six cubed-sphere tiles
    tile             that together cover the full sphere
                     (analytical area = 4*pi / 6)
                     Tile region (from `Cubed_Sphere_Tile_Boundary`):
                       x^2 + 2 z^2 <= R^2  AND  2 x^2 + z^2 <= R^2

For an N x N grid spanning a square whose side equals the diameter of
the unit circle (R = 1, side L = 2 R = 2), the pixel size is h = L / N
and pixel centers are at x_i = -R + h * (i + 1/2). Pixels whose centers
lie inside the chosen domain contribute to the sum, mirroring how
`Spherical_Reconstruction_2.detJ` is summed over the cell mask pixels.

The signed relative error is

    err(N) = A_recon(N) / A_analytical - 1

The script sweeps a log-spaced range of N and writes a single figure
(linear ratio + log-log |error|) plus a CSV of the raw values, with
both the full-disk and tile-only series shown together.
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


# Project plot convention: LaTeX rendering with Computer Modern.
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb,amsfonts}'


def detJ(R, x, z):
    """Same Jacobian determinant used in Spherical_Reconstruction_2."""
    return R / np.sqrt(R ** 2 - x ** 2 - z ** 2)


def _grid(N, R):
    """Build the N x N grid of pixel-center coordinates spanning [-R, R]^2."""
    L = 2.0 * R
    h = L / N
    edges = np.linspace(-R, R, N + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])  # midpoint of each pixel
    X, Z = np.meshgrid(centers, centers, indexing='xy')
    return X, Z, h


def reconstructed_areas(N, R=1.0):
    """Compute reconstructed area for both the full disk and one CST tile.

    Returns
    -------
    A_disk : float
        Sum of detJ * h^2 over pixel centers inside the full unit disk
        (x^2 + z^2 < R^2).
    A_tile : float
        Sum of detJ * h^2 over pixel centers inside one cubed-sphere tile
        (x^2 + 2 z^2 <= R^2 AND 2 x^2 + z^2 <= R^2).
    """
    X, Z, h = _grid(N, R)
    x2 = X ** 2
    z2 = Z ** 2
    r2 = x2 + z2

    inside_disk = r2 < R ** 2
    # CST tile: bounded by ellipses x^2 + 2 z^2 = R^2 (N/S) and 2 x^2 + z^2 = R^2 (E/W)
    inside_tile = inside_disk & (x2 + 2.0 * z2 <= R ** 2) & (2.0 * x2 + z2 <= R ** 2)

    detJ_full = np.zeros_like(X)
    detJ_full[inside_disk] = R / np.sqrt(R ** 2 - r2[inside_disk])

    A_disk = detJ_full[inside_disk].sum() * h ** 2
    A_tile = detJ_full[inside_tile].sum() * h ** 2
    return A_disk, A_tile


def sweep_resolutions(N_min=8, N_max=4096, n_points=40):
    """Log-spaced unique integer resolutions between N_min and N_max."""
    Ns = np.logspace(np.log10(N_min), np.log10(N_max), n_points)
    return np.unique(np.round(Ns).astype(int))


def run_error_analysis(output_dir, N_min=8, N_max=4096, n_points=40, R=1.0,
                       verbose=True):
    os.makedirs(output_dir, exist_ok=True)

    A_full_analytical = 2.0 * np.pi * R ** 2          # half-sphere
    A_tile_analytical = (4.0 * np.pi * R ** 2) / 6.0  # one of six cubed-sphere tiles

    Ns = sweep_resolutions(N_min=N_min, N_max=N_max, n_points=n_points)

    A_full = np.empty(Ns.size, dtype=float)
    A_tile = np.empty(Ns.size, dtype=float)
    for k, N in enumerate(Ns):
        A_full[k], A_tile[k] = reconstructed_areas(int(N), R=R)
        if verbose:
            r_full = A_full[k] / A_full_analytical
            r_tile = A_tile[k] / A_tile_analytical
            print(f"  N = {int(N):5d}   "
                  f"full ratio = {r_full:.6f} (err {r_full-1:+.2e})   "
                  f"tile ratio = {r_tile:.6f} (err {r_tile-1:+.2e})")

    ratio_full = A_full / A_full_analytical
    ratio_tile = A_tile / A_tile_analytical
    err_full = ratio_full - 1.0
    err_tile = ratio_tile - 1.0

    csv_path = os.path.join(output_dir,
                            'SR2_integration_error_vs_resolution.csv')
    np.savetxt(
        csv_path,
        np.column_stack([Ns, A_full, ratio_full, err_full,
                              A_tile, ratio_tile, err_tile]),
        header=('N,A_full_recon,A_full_ratio,A_full_relerr,'
                'A_tile_recon,A_tile_ratio,A_tile_relerr'),
        delimiter=',', comments='',
    )

    png_path = os.path.join(output_dir,
                            'SR2_integration_error_vs_resolution.png')
    svg_path = os.path.join(output_dir,
                            'SR2_integration_error_vs_resolution.svg')
    _plot_results(Ns, ratio_full, err_full, ratio_tile, err_tile,
                  png_path=png_path, svg_path=svg_path)

    return {
        'Ns': Ns,
        'A_full': A_full,
        'A_tile': A_tile,
        'ratio_full': ratio_full,
        'ratio_tile': ratio_tile,
        'err_full': err_full,
        'err_tile': err_tile,
        'csv_path': csv_path,
        'png_path': png_path,
        'svg_path': svg_path,
    }


def _plot_results(Ns, ratio_full, err_full, ratio_tile, err_tile,
                  png_path, svg_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: ratio vs resolution (linear y, log x)
    ax = axes[0]
    ax.plot(Ns, ratio_full, 'o-', color='C0', linewidth=1.5, markersize=5,
            label=r'full half-sphere (disk)')
    ax.plot(Ns, ratio_tile, 's-', color='C2', linewidth=1.5, markersize=5,
            label=r'one CST tile')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5,
               label=r'analytical (=1)')
    ax.set_xscale('log')
    ax.set_xlabel(r'Grid resolution $N$ (pixels per side)')
    ax.set_ylabel(r'$A_{\mathrm{recon}}/A_{\mathrm{analytical}}$')
    ax.set_title(r'Reconstructed area / analytical area')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    # Right: |relative error| on log-log
    ax = axes[1]
    abs_full = np.abs(err_full)
    abs_tile = np.abs(err_tile)
    ax.loglog(Ns, abs_full, 'o-', color='C0', linewidth=1.5, markersize=5,
              label=r'full half-sphere (disk)')
    ax.loglog(Ns, abs_tile, 's-', color='C2', linewidth=1.5, markersize=5,
              label=r'one CST tile')
    Nref = np.array([Ns[0], Ns[-1]], dtype=float)
    anchor = max(abs_full[0], abs_tile[0])
    ax.loglog(Nref, anchor * (Nref / Ns[0]) ** -1.0, 'k:', alpha=0.6,
              label=r'$\propto N^{-1}$')
    ax.loglog(Nref, anchor * (Nref / Ns[0]) ** -2.0, 'k--', alpha=0.6,
              label=r'$\propto N^{-2}$')
    ax.set_xlabel(r'Grid resolution $N$ (pixels per side)')
    ax.set_ylabel(r'$|A_{\mathrm{recon}}/A_{\mathrm{analytical}}-1|$')
    ax.set_title(r'Numerical integration error')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    fig.suptitle(
        r'Spherical reconstruction: numerical integration error '
        r'(midpoint Jacobian rule on full disk vs.\ one cubed-sphere tile)',
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-cell error analysis: random circular cells in the disk
# ---------------------------------------------------------------------------

def reconstructed_cell_area(N_full, R, r_c, x_c=0.0, z_c=0.0):
    """SR2-style integration of detJ over a circular cell.

    Builds the disk grid at resolution N_full, takes the pixel-staircase
    representation of the cell (pixels whose centres lie inside the smooth
    cell disk), and sums `detJ * h^2` over those pixels.
    """
    h = 2.0 * R / N_full
    edges = np.linspace(-R, R, N_full + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    X, Z = np.meshgrid(centers, centers, indexing='xy')
    inside_cell = (X - x_c) ** 2 + (Z - z_c) ** 2 < r_c ** 2
    inside_disk = X ** 2 + Z ** 2 < R ** 2
    mask = inside_cell & inside_disk
    if not mask.any():
        return 0.0
    detJ_vals = R / np.sqrt(R ** 2 - X[mask] ** 2 - Z[mask] ** 2)
    return detJ_vals.sum() * h ** 2


def analytical_cell_area(R, r_c, x_c=0.0, z_c=0.0,
                         M_rho=64, M_psi_base=600):
    """Reference value of the detJ integral over a smooth circular cell.

    Closed form for a cell centred at the origin (spherical-cap area).

    For an off-centre cell, switch to polar coordinates (rho, psi) centred
    on the cell. The inner integral over psi is over a smooth 2*pi-periodic
    function, so the trapezoidal rule converges exponentially in M_psi
    (Euler--Maclaurin); the outer integral over rho is smooth on [0, r_c]
    and is handled by Gauss--Legendre.

    `M_psi` is bumped automatically when the cell approaches the disk
    silhouette (d + r_c -> R), where the integrand peaks sharply at psi = 0.
    """
    if x_c == 0.0 and z_c == 0.0:
        return 2.0 * np.pi * R * (R - np.sqrt(R ** 2 - r_c ** 2))

    d = float(np.hypot(x_c, z_c))
    if d + r_c >= R:
        raise ValueError(f"Cell extends past disk boundary: d + r_c = "
                         f"{d + r_c:.6g} >= R = {R:.6g}")

    proximity = (d + r_c) / R
    if proximity > 0.99:
        M_psi = max(M_psi_base, 20000)
    elif proximity > 0.95:
        M_psi = max(M_psi_base, 4000)
    elif proximity > 0.85:
        M_psi = max(M_psi_base, 1500)
    else:
        M_psi = M_psi_base

    rho_nodes, rho_weights = np.polynomial.legendre.leggauss(M_rho)
    rho = 0.5 * r_c * (rho_nodes + 1.0)
    rho_w = 0.5 * r_c * rho_weights

    psi = np.linspace(0.0, 2.0 * np.pi, M_psi, endpoint=False)
    cos_psi = np.cos(psi)

    rho_g = rho[:, None]
    r2 = d ** 2 + rho_g ** 2 + 2.0 * rho_g * d * cos_psi[None, :]
    arg = R ** 2 - r2
    detJ_grid = R / np.sqrt(np.maximum(arg, 0.0) + 1e-300)
    inner = (2.0 * np.pi / M_psi) * detJ_grid.sum(axis=1)
    return float(np.sum(rho_w * rho * inner))


def _cell_in_tile(x_c, z_c, r_c, R, n_check=180):
    """True iff the entire circular cell lies inside the cubed-sphere tile.

    Tile = {(x, z) : x^2 + 2 z^2 <= R^2 AND 2 x^2 + z^2 <= R^2}.
    Tested by sampling n_check equispaced points around the cell perimeter.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_check, endpoint=False)
    x_p = x_c + r_c * np.cos(theta)
    z_p = z_c + r_c * np.sin(theta)
    return bool(np.all((x_p ** 2 + 2.0 * z_p ** 2 <= R ** 2)
                       & (2.0 * x_p ** 2 + z_p ** 2 <= R ** 2)))


def random_cells(K, R, size_frac_min=1.0 / 20.0, size_frac_max=1.0 / 5.0,
                 seed=0, domain='disk'):
    """Draw K random circular cells fully contained in the chosen domain.

    Cell *radius* is uniform in [size_frac_min, size_frac_max] * R (so cell
    *diameter* is uniform in the same fractions of the disk diameter 2R).

    domain : 'disk' or 'tile'
        'disk' uses analytical uniform-by-area sampling inside the shrunk
        disk of radius (R - r_c).
        'tile' uses rejection sampling inside the tile bounding box
        [-R/sqrt(2), R/sqrt(2)]^2, keeping only cells whose entire perimeter
        lies inside both tile-bounding ellipses; the accepted distribution
        is therefore uniform-by-area inside the shrunk tile.
    """
    rng = np.random.default_rng(seed)
    if domain == 'disk':
        r_c = rng.uniform(size_frac_min, size_frac_max, size=K) * R
        max_d = R - r_c
        u = rng.uniform(0.0, 1.0, size=K)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=K)
        radial = np.sqrt(u) * max_d
        x_c = radial * np.cos(theta)
        z_c = radial * np.sin(theta)
        return np.column_stack([r_c, x_c, z_c])

    if domain == 'tile':
        bbox = R / np.sqrt(2.0)
        out = np.empty((K, 3), dtype=float)
        accepted = 0
        attempts = 0
        max_attempts = max(50_000, K * 200)
        while accepted < K and attempts < max_attempts:
            r_c = rng.uniform(size_frac_min, size_frac_max) * R
            x_c = rng.uniform(-bbox, bbox)
            z_c = rng.uniform(-bbox, bbox)
            attempts += 1
            if _cell_in_tile(x_c, z_c, r_c, R):
                out[accepted] = (r_c, x_c, z_c)
                accepted += 1
        if accepted < K:
            raise RuntimeError(
                f"random_cells(domain='tile'): only accepted {accepted}/{K} "
                f"cells after {attempts} attempts."
            )
        return out

    raise ValueError(f"Unknown domain={domain!r}; expected 'disk' or 'tile'.")


def run_per_cell_error_analysis(output_dir, R=1.0,
                                N_full_values=(128, 256, 512, 1024, 2048),
                                K=300, size_frac_min=1.0 / 20.0,
                                size_frac_max=1.0 / 5.0, seed=0,
                                domain='disk', verbose=True):
    """Per-cell error analysis over K random cells across a sweep of N_full."""
    os.makedirs(output_dir, exist_ok=True)
    cells = random_cells(K, R, size_frac_min, size_frac_max, seed, domain=domain)
    r_c_arr = cells[:, 0]
    x_c_arr = cells[:, 1]
    z_c_arr = cells[:, 2]
    radial_pos = np.sqrt(x_c_arr ** 2 + z_c_arr ** 2)

    if verbose:
        print(f"\nPer-cell analysis (domain='{domain}'): K = {K} random cells, "
              f"r_c in [{size_frac_min:.3f}, {size_frac_max:.3f}] * R")

    # Analytical reference areas (independent of N_full)
    A_true = np.empty(K, dtype=float)
    for j in range(K):
        A_true[j] = analytical_cell_area(R, r_c_arr[j], x_c_arr[j], z_c_arr[j])

    # Reconstructed area for each (cell, resolution)
    n_N = len(N_full_values)
    rel_err = np.empty((n_N, K), dtype=float)
    for k, N_full in enumerate(N_full_values):
        for j in range(K):
            A_recon = reconstructed_cell_area(N_full, R, r_c_arr[j],
                                              x_c_arr[j], z_c_arr[j])
            rel_err[k, j] = A_recon / A_true[j] - 1.0
        if verbose:
            ae = np.abs(rel_err[k])
            print(f"  N_full = {N_full:5d}   median |err| = {np.median(ae):.3e}"
                  f"   p95 = {np.percentile(ae, 95):.3e}"
                  f"   max = {ae.max():.3e}")

    # CSV: cell metadata + relative error per resolution
    csv_path = os.path.join(output_dir, f'SR2_per_cell_error_{domain}.csv')
    cols = [np.arange(K), r_c_arr, x_c_arr, z_c_arr, radial_pos, A_true]
    cols += [rel_err[k] for k in range(n_N)]
    header = (
        'cell_id,r_c,x_c,z_c,radial_pos,A_true,'
        + ','.join([f'rel_err_N{int(N)}' for N in N_full_values])
    )
    np.savetxt(csv_path, np.column_stack(cols), header=header,
               delimiter=',', comments='')

    png_path = os.path.join(output_dir, f'SR2_per_cell_error_{domain}.png')
    svg_path = os.path.join(output_dir, f'SR2_per_cell_error_{domain}.svg')
    _plot_per_cell(N_full_values, rel_err, r_c_arr, radial_pos, R,
                   size_frac_min, size_frac_max, K, domain,
                   png_path, svg_path)

    return {
        'cells': cells,
        'A_true': A_true,
        'rel_err': rel_err,
        'N_full_values': np.asarray(N_full_values),
        'csv_path': csv_path,
        'png_path': png_path,
        'svg_path': svg_path,
    }


def _plot_per_cell(N_full_values, rel_err, r_c_all, radial_pos, R,
                   size_frac_min, size_frac_max, K, domain,
                   png_path, svg_path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    Ns = np.asarray(N_full_values, dtype=float)
    abs_err = np.abs(rel_err)

    # Panel 1: median + percentile band of |rel_err| vs N_full
    ax = axes[0]
    median = np.median(abs_err, axis=1)
    p05 = np.percentile(abs_err, 5, axis=1)
    p95 = np.percentile(abs_err, 95, axis=1)
    p99 = np.percentile(abs_err, 99, axis=1)
    ax.fill_between(Ns, p05, p95, alpha=0.25, color='C0',
                    label=r'5\%--95\% band')
    ax.plot(Ns, median, 'o-', color='C0', label=r'median')
    ax.plot(Ns, p99, 'x:', color='C3', label=r'99\% percentile')
    ax.set_xscale('log')
    ax.set_yscale('log')
    Nref = np.array([Ns[0], Ns[-1]], dtype=float)
    anchor = median[0]
    ax.loglog(Nref, anchor * (Nref / Ns[0]) ** -1.0, 'k:', alpha=0.5,
              label=r'$\propto N^{-1}$')
    ax.loglog(Nref, anchor * (Nref / Ns[0]) ** -2.0, 'k--', alpha=0.5,
              label=r'$\propto N^{-2}$')
    ax.set_xlabel(r'Disk grid resolution $N$ (pixels per side)')
    ax.set_ylabel(r'$|A_{\mathrm{recon}}/A_{\mathrm{analytical}} - 1|$ per cell')
    ax.set_title(r'Per-cell error vs.\ disk resolution')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 2: scatter rel_err vs radial position at largest N_full
    last = -1
    ax = axes[1]
    sc = ax.scatter(radial_pos / R, rel_err[last], c=r_c_all / R,
                    cmap='viridis', s=22, alpha=0.85)
    plt.colorbar(sc, ax=ax, label=r'cell radius $r_c/R$')
    ax.axhline(0.0, color='k', linewidth=0.5)
    ax.set_xlabel(r'Cell radial position $|c|/R$')
    ax.set_ylabel(r'$A_{\mathrm{recon}}/A_{\mathrm{analytical}} - 1$')
    ax.set_title(rf'Per-cell error at $N={int(N_full_values[last])}$')
    ax.grid(True, alpha=0.3)

    # Panel 3: scatter rel_err vs cell size at largest N_full
    ax = axes[2]
    sc = ax.scatter(r_c_all / R, rel_err[last], c=radial_pos / R,
                    cmap='plasma', s=22, alpha=0.85)
    plt.colorbar(sc, ax=ax, label=r'cell radial position $|c|/R$')
    ax.axhline(0.0, color='k', linewidth=0.5)
    ax.set_xlabel(r'Cell radius $r_c/R$')
    ax.set_ylabel(r'$A_{\mathrm{recon}}/A_{\mathrm{analytical}} - 1$')
    ax.set_title(rf'Per-cell error at $N={int(N_full_values[last])}$')
    ax.grid(True, alpha=0.3)

    domain_label = {
        'disk': 'uniform-by-area inside disk',
        'tile': 'uniform-by-area inside cubed-sphere tile',
    }.get(domain, f"domain={domain}")
    fig.suptitle(
        rf'Per-cell SR2 integration error: $K={K}$ random circular cells, '
        rf'diameter $\in [{size_frac_min:.3f}, {size_frac_max:.3f}]\times 2R$, '
        rf'{domain_label}',
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)


def _default_output_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, 'SR2_error_analysis')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--output_dir', type=str,
                        default=_default_output_dir(),
                        help='Output directory for plots and CSV. '
                             'Defaults to ./SR2_error_analysis next to this script.')
    parser.add_argument('--N_min', type=int, default=8,
                        help='Minimum grid resolution to test.')
    parser.add_argument('--N_max', type=int, default=4096,
                        help='Maximum grid resolution to test.')
    parser.add_argument('--n_points', type=int, default=40,
                        help='Number of resolutions sampled (log-spaced).')
    parser.add_argument('--R', type=float, default=1.0,
                        help='Sphere radius (default 1).')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-resolution print output.')
    parser.add_argument('--per_cell', action='store_true', default=True,
                        help='Also run the per-cell error analysis '
                             '(default: enabled).')
    parser.add_argument('--no_per_cell', dest='per_cell', action='store_false',
                        help='Disable the per-cell analysis.')
    parser.add_argument('--K_cells', type=int, default=300,
                        help='Number of random cells for per-cell analysis.')
    parser.add_argument('--cell_size_min', type=float, default=1.0 / 20.0,
                        help='Minimum cell-diameter / disk-diameter ratio.')
    parser.add_argument('--cell_size_max', type=float, default=1.0 / 5.0,
                        help='Maximum cell-diameter / disk-diameter ratio.')
    parser.add_argument('--per_cell_seed', type=int, default=0,
                        help='Random seed for cell sampling.')
    parser.add_argument('--cell_domain', type=str, nargs='+',
                        default=['disk', 'tile'],
                        choices=['disk', 'tile'],
                        help="Which domain(s) to place test cells in. "
                             "Multiple values run multiple sweeps. "
                             "Default: both 'disk' and 'tile'.")
    args = parser.parse_args()

    print(f"Output directory: {args.output_dir}")
    out = run_error_analysis(
        output_dir=args.output_dir,
        N_min=args.N_min,
        N_max=args.N_max,
        n_points=args.n_points,
        R=args.R,
        verbose=not args.quiet,
    )
    print(f"\nSaved CSV: {out['csv_path']}")
    print(f"Saved PNG: {out['png_path']}")
    print(f"Saved SVG: {out['svg_path']}")

    if args.per_cell:
        N_full_values = sorted(set(int(n) for n in
                                   sweep_resolutions(N_min=max(args.N_min, 64),
                                                     N_max=args.N_max,
                                                     n_points=6)))
        for domain in args.cell_domain:
            cell_out = run_per_cell_error_analysis(
                output_dir=args.output_dir,
                R=args.R,
                N_full_values=tuple(N_full_values),
                K=args.K_cells,
                size_frac_min=args.cell_size_min,
                size_frac_max=args.cell_size_max,
                seed=args.per_cell_seed,
                domain=domain,
                verbose=not args.quiet,
            )
            print(f"\nSaved per-cell ({domain}) CSV: {cell_out['csv_path']}")
            print(f"Saved per-cell ({domain}) PNG: {cell_out['png_path']}")
            print(f"Saved per-cell ({domain}) SVG: {cell_out['svg_path']}")


if __name__ == '__main__':
    main()
