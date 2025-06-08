def remove_red(img_bgr):
    """Pone en negro los píxeles predominantemente rojos de una imagen BGR."""
    out = img_bgr.copy()

    r = out[..., 0]
    g = out[..., 1]
    b = out[..., 2]

    # mask definitions
    red_only = (r > 0) & (g == 0) & (b == 0)
    green_only = (g > 0) & (r == 0) & (b == 0)
    blue_only = (b > 0) & (r == 0) & (g == 0)

    # apply transformations
    out[red_only] = [255, 255, 255]
    out[green_only] = [255, 255, 255]
    out[blue_only] = [0, 0, 0]

    return out
