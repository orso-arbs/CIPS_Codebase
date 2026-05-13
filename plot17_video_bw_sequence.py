"""
plot17_video_bw_sequence.py
----------------------------
Create an annotated MP4 video from a sequence of BW PNG frames.
Each frame is labelled with its time instant (or frame number) in the
bottom-left corner using a LaTeX / Computer Modern font.

Usage:
    Run as a script (edit the parameters at the bottom of the file), or
    import and call make_bw_video() directly.

Dependencies (MastersThesis_Env2_py39):
    matplotlib, opencv-python (cv2), Pillow (PIL), numpy
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import pandas as pd

# ---------------------------------------------------------------------------
# LaTeX / Computer Modern font settings  (matches codebase convention)
# ---------------------------------------------------------------------------
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amssymb,amsfonts}"


def make_bw_video(
    input_dir,
    fps=10,
    time_per_frame_ms=None,
    csv_path=None,
    time_label_prefix=r"t/\tau",
    font_size=14,

    output_filename=None,
    output_dir_manual="",
    output_dir_comment="",
    dpi=150,
):
    """
    Build an annotated MP4 video from bw_NNNN.png frames.

    Parameters
    ----------
    input_dir : str or Path
        Folder containing bw_NNNN.png images.
    fps : int or float
        Frames per second for the output video.
    time_per_frame_ms : float or None
        Physical time step per frame in milliseconds.
        If None (and csv_path is also None), the frame index is shown.
        If given, a LaTeX-formatted time label is shown, e.g. $t = 2.5\\,\\mathrm{ms}$.
        Ignored when csv_path is provided.
    csv_path : str or Path or None
        Path to Analysis_A11_final_df.csv (tab-separated).  When provided, the
        Time_VisIt value for each frame is looked up via image_number (= the
        4-digit number in the bw_NNNN.png filename) and displayed as a label.
        Takes precedence over time_per_frame_ms.
    time_label_prefix : str
        LaTeX string used as the left-hand side of the time label when csv_path
        is given, e.g. r"t/\\tau" → label becomes $t/\\tau = 0.05$.
        Default: r"t/\\tau".
    font_size : int
        Font size for the timestamp label (matplotlib points).
    label_color : str
        Colour of the label text ('white' or 'black' work well on BW images).
    output_filename : str or None
        MP4 filename (no path).  Defaults to 'video_fps{fps}.mp4'.
    output_dir_manual : str
        If non-empty, override the auto output directory.
    output_dir_comment : str
        Appended to the auto output directory name.
    dpi : int
        Rendering DPI.  Higher = sharper but slower.  150 gives full resolution
        for typical high-speed camera crops.
    """
    input_dir = Path(input_dir)

    # -----------------------------------------------------------------------
    # Collect frames
    # -----------------------------------------------------------------------
    frame_paths = sorted(input_dir.glob("bw_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No bw_*.png files found in {input_dir}")
    print(f"Found {len(frame_paths)} frames in:\n  {input_dir}")

    # -----------------------------------------------------------------------
    # Load time lookup from CSV  (image_number → Time_VisIt)
    # -----------------------------------------------------------------------
    visit_time_map = {}
    if csv_path is not None:
        # The CSV contains embedded multi-line numpy arrays in some columns
        # (outlines, masks).  usecols limits parsing to the two needed columns;
        # on_bad_lines='skip' discards the continuation lines of those arrays.
        # Each image_number appears once per cell, so enough valid rows survive.
        df_csv = pd.read_csv(
            csv_path, sep="\t",
            usecols=["image_number", "Time_VisIt"],
            on_bad_lines="skip",
            low_memory=False,
        )
        df_csv = df_csv.dropna(subset=["image_number", "Time_VisIt"])
        visit_time_map = (
            df_csv.groupby("image_number")["Time_VisIt"].first()
            .astype(float)
            .to_dict()
        )
        # keys may be floats (e.g. 1.0) — re-key as int
        visit_time_map = {int(k): v for k, v in visit_time_map.items()}
        print(f"Loaded {len(visit_time_map)} Time_VisIt entries from:\n  {csv_path}")

    # -----------------------------------------------------------------------
    # Output directory & filename
    # -----------------------------------------------------------------------
    if output_dir_manual:
        out_dir = Path(output_dir_manual)
    else:
        suffix = f"_{output_dir_comment}" if output_dir_comment else ""
        out_dir = input_dir / f"video_output{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = f"video_fps{fps}.mp4"
    out_path = out_dir / output_filename
    print(f"Output → {out_path}")

    # -----------------------------------------------------------------------
    # Determine output frame size from first image
    # -----------------------------------------------------------------------
    first_img = np.array(Image.open(frame_paths[0]).convert("L"))
    img_h, img_w = first_img.shape

    # Figure size in inches that exactly reproduces pixel dimensions at `dpi`
    fig_w_in = img_w / dpi
    fig_h_in = img_h / dpi

    # -----------------------------------------------------------------------
    # OpenCV VideoWriter  (fourcc = mp4v for broad compatibility)
    # -----------------------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (img_w, img_h))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV VideoWriter failed to open: {out_path}")

    # -----------------------------------------------------------------------
    # Render frames
    # -----------------------------------------------------------------------
    for i, fp in enumerate(frame_paths):
        img_gray = np.array(Image.open(fp).convert("L"))

        # --- matplotlib figure with exact pixel size -----------------------
        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])          # fill entire figure
        ax.imshow(img_gray, cmap="gray", interpolation="none",
                  vmin=0, vmax=255, aspect="equal")
        ax.set_xlim(0, img_w - 1)
        ax.set_ylim(img_h - 1, 0)
        ax.axis("off")

        # --- timestamp label -----------------------------------------------
        if visit_time_map:
            img_num = int(fp.stem.split("_")[1])
            tv = visit_time_map.get(img_num, float("nan"))
            label = rf"${time_label_prefix} = {tv:.2f}$"
        elif time_per_frame_ms is not None:
            t_val = i * time_per_frame_ms
            label = rf"$t = {t_val:.1f}\,\mathrm{{ms}}$"
        else:
            label = rf"Frame $\,{i + 1}$"

        ax.text(
            0.02, 0.04,
            label,
            transform=ax.transAxes,
            fontsize=font_size * 2,
            color="black",
            verticalalignment="bottom",
            horizontalalignment="left",
        )

        # --- capture to numpy (RGBA) then convert to BGR for cv2 -----------
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())          # shape: (H, W, 4)
        rgb = rgba[:, :, :3]                              # drop alpha

        # Resize to exact target size in case of sub-pixel rounding
        if rgb.shape[0] != img_h or rgb.shape[1] != img_w:
            rgb = cv2.resize(rgb, (img_w, img_h), interpolation=cv2.INTER_AREA)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
        plt.close(fig)

        if (i + 1) % 20 == 0 or (i + 1) == len(frame_paths):
            print(f"  {i + 1}/{len(frame_paths)} frames rendered …")

    writer.release()
    print(f"\nDone.  Video saved to:\n  {out_path}")
    return str(out_path)


# ===========================================================================
# Entry point — edit parameters here and run the file directly
# ===========================================================================
if __name__ == "__main__":
    INPUT_DIR = (
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis"
        r"\FB images\Visit_projections_initial_test\BW 134 ball flame - Crop"
    )

    # ------------------------------------------------------------------
    # Core parameters
    # ------------------------------------------------------------------
    FPS = 10                    # ← change me; try e.g. 5, 10, 15, 25
    FONT_SIZE = 14              # label font size (matplotlib points)
    DPI = 150                   # render DPI — keep at original image resolution

    # Path to the tab-separated analysis CSV that contains image_number and
    # Time_VisIt columns.  Set to None to fall back to frame-index labels.
    CSV_PATH = (
        r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis"
        r"\CIPS_Pipe_Default_dir\20250625_1528537\20250625_1528554"
        r"\20250625_1626096\20250626_1700136\20250628_2007187"
        r"\Analysis_A11_final_df.csv"
    )

    # ------------------------------------------------------------------
    # Run for a single fps (or loop over several values)
    # ------------------------------------------------------------------
    make_bw_video(
        input_dir=INPUT_DIR,
        fps=FPS,
        csv_path=CSV_PATH,
        time_label_prefix=r"t/\tau",   # LaTeX LHS of label, e.g. t/τ = 0.05
        font_size=FONT_SIZE,
        dpi=DPI,
        output_dir_comment="",   # optional tag appended to the output folder
    )

    # ------------------------------------------------------------------
    # Uncomment to quickly batch-generate videos at multiple fps values:
    # ------------------------------------------------------------------
    # for fps in [5, 10, 15, 25]:
    #     make_bw_video(input_dir=INPUT_DIR, fps=fps,
    #                   csv_path=CSV_PATH, dpi=DPI)
