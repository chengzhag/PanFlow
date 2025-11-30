import numpy as np
import cv2


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology
        for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Vectorized color mapping.
    Args:
        u, v : float32 arrays of shape (B, H, W)
    Returns:
        uint8 RGB (or BGR) images, shape (B, H, W, 3)
    """
    B, H, W = u.shape
    colorwheel = make_colorwheel() / 255.0        # (55, 3)
    ncols = colorwheel.shape[0]

    rad = np.sqrt(u ** 2 + v ** 2)                # magnitude
    a   = np.arctan2(-v, -u) / np.pi              # range [-1, 1]
    fk  = (a + 1) / 2 * (ncols - 1)               # map angle to [0, 54]
    k0  = np.floor(fk).astype(np.int32)
    k1  = (k0 + 1) % ncols
    f   = fk - k0                                 # interpolation factor

    col0 = colorwheel[k0]                         # (B, H, W, 3)
    col1 = colorwheel[k1]
    col  = (1 - f)[..., None] * col0 + f[..., None] * col1

    # Saturation adjustment: inside/outside unit circle
    mask = (rad <= 1)[..., None]
    col  = np.where(mask, 1 - rad[..., None] * (1 - col), col * 0.75)

    col = (col * 255.0).astype(np.uint8)
    if convert_to_bgr:
        col = col[..., ::-1]                      # RGB → BGR
    return col


def _add_batch_dim(arr):
    """
    Ensure the input has a batch dimension.

    Returns
    -------
    batched : ndarray
        If `arr` was (H,W,C) it becomes (1,H,W,C); if already (B,H,W,C) it’s
        returned unchanged.
    squeeze_back : bool
        True if a leading batch dim was added and should be removed on return.
    """
    if arr is None:
        return None, False
    if arr.ndim == 3:         # (H, W, C)
        return arr[None, ...], True
    if arr.ndim == 4:         # (B, H, W, C)
        return arr, False
    raise ValueError("Expected 3- or 4-D array.")


def better_flow_to_image(flow_uv,
                         alpha=0.1,
                         max_flow=None,
                         clip_flow=None,
                         convert_to_bgr=False):
    """
    Batch-friendly, vectorized optical-flow visualization.
    Args:
        flow_uv : (H, W, 2) or (B, H, W, 2) array
        alpha   : exponent for magnitude compression (default 0.1)
        max_flow: normalizing constant; if None, uses per-batch max(|flow|)
        clip_flow: optional clipping threshold applied before visualization
        convert_to_bgr: if True, output is BGR instead of RGB
    Returns:
        uint8 image(s): (H, W, 3) or (B, H, W, 3)
    """
    # -------- dimension handling (reuse helper) -----------------------------
    flow_uv, squeeze_out = _add_batch_dim(flow_uv)      # (B,H,W,2)
    flow_uv = flow_uv.astype(np.float32)

    # -------- optional clipping --------------------------------------------
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, -clip_flow, clip_flow)

    u, v = flow_uv[..., 0], flow_uv[..., 1]             # (B,H,W)

    # ---------------- α-scaling ------------------------------------------------
    rad     = np.sqrt(u ** 2 + v ** 2)
    eps     = 1e-5
    rad_max = max_flow if max_flow is not None else (rad.max() + eps)
    scale   = np.power(rad / (rad_max + eps), alpha)
    u_scaled = scale * u / (rad_max + eps)
    v_scaled = scale * v / (rad_max + eps)

    # ---------------- Color mapping -------------------------------------------
    img = flow_uv_to_colors(u_scaled, v_scaled, convert_to_bgr)

    return img[0] if squeeze_out else img


def flow_to_arrows(
    flow_uv,
    step: int = 16,
    scale: float = 1.0,
    canvas=(0, 0, 0),          # RGB tuple or image tensor
    color=(255, 255, 0),       # RGB arrow colour
    thickness: int = 1,
    tipLength: float = 0.1,
):
    """
    Visualise optical flow as sparse arrows.

    Parameters
    ----------
    flow_uv : ndarray
        Flow tensor (H,W,2) or (B,H,W,2) in px / frame.
    step : int
        Grid stride in pixels between arrows.
    scale : float
        Multiplier applied to flow vectors for visual length.
    canvas : tuple[int,int,int] or ndarray
        * RGB tuple  -> create a solid-colour background of that colour.
        * ndarray    -> RGB image(s) with shape matching `flow_uv` (B,H,W,3).
    color : tuple[int,int,int]
        Arrow colour (RGB).

    Returns
    -------
    ndarray
        RGB image(s) with arrows. Shape matches `flow_uv` except channels = 3.
    """
    # -------------------- verify arrow colour --------------------------------
    if not (isinstance(color, (tuple, list)) and len(color) == 3):
        raise ValueError("`color` must be an RGB tuple/list of length 3.")
    bgr_arrow = tuple(int(c) for c in color[::-1])  # OpenCV wants BGR

    # -------------------- batch handling for flow ----------------------------
    flow_uv, squeeze_out = _add_batch_dim(flow_uv)          # (B, H, W, 2)
    B, H, W, _ = flow_uv.shape

    # -------------------- prepare background ---------------------------------
    if isinstance(canvas, (tuple, list)):                   # solid colour
        if len(canvas) != 3:
            raise ValueError("`canvas` RGB tuple must have length 3.")
        rgb_fill = np.array(canvas, dtype=np.uint8, copy=False)
        bg = np.tile(rgb_fill, (B, H, W, 1))                # (B,H,W,3)
    elif isinstance(canvas, np.ndarray):                    # image tensor
        bg, _ = _add_batch_dim(canvas)
        if bg.shape[:3] != (B, H, W):
            raise ValueError("`canvas` image size / batch mismatch.")
        bg = bg.copy()                                      # avoid in-place edit
    else:
        raise TypeError("`canvas` must be an RGB tuple or an ndarray.")

    # -------------------- arrow grid -----------------------------------------
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step].astype(int)

    # -------------------- draw arrows batch-wise -----------------------------
    for b in range(B):
        img_bgr = cv2.cvtColor(bg[b], cv2.COLOR_RGB2BGR)
        dx, dy = (flow_uv[b, ys, xs] * scale).transpose(2, 0, 1)  # (dx, dy)

        for x0, y0, ddx, ddy in zip(xs.flat, ys.flat, dx.flat, dy.flat):
            cv2.arrowedLine(
                img_bgr,
                (int(x0), int(y0)),
                (int(round(x0 + ddx)), int(round(y0 + ddy))),
                bgr_arrow,
                thickness=thickness,
                tipLength=tipLength,
                line_type=cv2.LINE_AA,
            )

        bg[b] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return bg[0] if squeeze_out else bg
