"""
Generate realistic synthetic chest X-ray demo images for ENGRAM medical training app.

Creates 512x512 grayscale PNG images for 11 pathology categories (3 per category).
Each category has visually distinct features that simulate the corresponding pathology.

Uses PIL/Pillow only -- no external dependencies.
"""

import math
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIZE = 512
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "demo")
CATEGORIES = [
    "Cardiomegaly",
    "Pneumothorax",
    "Pleural Effusion",
    "Lung Opacity",
    "Consolidation",
    "Atelectasis",
    "Edema",
    "Pneumonia",
    "No Finding",
    "Support Devices",
    "Fracture",
]
IMAGES_PER_CATEGORY = 3

# Seed for reproducibility
random.seed(42)

# ---------------------------------------------------------------------------
# Helper: Noise texture via random pixel blocks
# ---------------------------------------------------------------------------


def _add_noise(img: Image.Image, intensity: int = 12) -> Image.Image:
    """Add subtle film-grain noise to the image."""
    pixels = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            offset = random.randint(-intensity, intensity)
            v = max(0, min(255, pixels[x, y] + offset))
            pixels[x, y] = v
    return img


def _radial_gradient(cx: float, cy: float, rx: float, ry: float,
                     brightness: int, size: int = SIZE) -> Image.Image:
    """Create an elliptical radial gradient centered at (cx, cy)."""
    grad = Image.new("L", (size, size), 0)
    pixels = grad.load()
    for y in range(size):
        for x in range(size):
            dx = (x - cx) / max(rx, 1)
            dy = (y - cy) / max(ry, 1)
            d = math.sqrt(dx * dx + dy * dy)
            v = max(0, int(brightness * max(0, 1.0 - d)))
            pixels[x, y] = v
    return grad


def _blend_add(base: Image.Image, overlay: Image.Image, alpha: float = 1.0) -> Image.Image:
    """Additive blend of overlay onto base."""
    from PIL import ImageChops
    if alpha < 1.0:
        overlay = overlay.point(lambda p: int(p * alpha))
    return ImageChops.add(base, overlay)


def _blend_screen(base: Image.Image, overlay: Image.Image) -> Image.Image:
    """Screen blend (lightens)."""
    from PIL import ImageChops
    inv_base = ImageChops.invert(base)
    inv_overlay = ImageChops.invert(overlay)
    mult = ImageChops.multiply(inv_base, inv_overlay)
    return ImageChops.invert(mult)


# ---------------------------------------------------------------------------
# Base chest X-ray (PA view)
# ---------------------------------------------------------------------------


def create_base_xray(variation: int = 0) -> Image.Image:
    """
    Generates a base chest X-ray appearance:
      - Dark background (air-filled lungs)
      - Lighter central mediastinum / spine
      - Rib cage arcs
      - Diaphragm dome at bottom
      - Shoulder/clavicle structures at top
    """
    img = Image.new("L", (SIZE, SIZE), 10)
    draw = ImageDraw.Draw(img)

    cx, cy = SIZE // 2, SIZE // 2
    # Small positional variation per image
    x_shift = random.randint(-8, 8)
    y_shift = random.randint(-5, 5)
    cx += x_shift
    cy += y_shift

    # --- Thorax outline (elliptical body contour) ---
    body_rx, body_ry = 210 + random.randint(-10, 10), 230 + random.randint(-5, 5)
    body_box = [cx - body_rx, cy - body_ry - 10, cx + body_rx, cy + body_ry + 30]
    draw.ellipse(body_box, fill=18)

    # --- Lung fields (darker ellipses, left and right) ---
    lung_offset_x = 90 + random.randint(-5, 5)
    lung_rx = 80 + random.randint(-5, 5)
    lung_ry = 130 + random.randint(-5, 5)
    lung_cy = cy - 15
    # Right lung
    r_lung = [cx + lung_offset_x - lung_rx, lung_cy - lung_ry,
              cx + lung_offset_x + lung_rx, lung_cy + lung_ry]
    draw.ellipse(r_lung, fill=8)
    # Left lung
    l_lung = [cx - lung_offset_x - lung_rx, lung_cy - lung_ry,
              cx - lung_offset_x + lung_rx, lung_cy + lung_ry]
    draw.ellipse(l_lung, fill=8)

    # --- Mediastinum (central bright region: heart + spine + vessels) ---
    med_grad = _radial_gradient(cx, cy + 20, 70, 120, 55)
    from PIL import ImageChops
    img = ImageChops.add(img, med_grad)
    draw = ImageDraw.Draw(img)

    # --- Spine (vertical bright line) ---
    spine_w = 14 + random.randint(-2, 2)
    spine_top = cy - 180
    spine_bot = cy + 200
    draw.rectangle([cx - spine_w // 2, spine_top, cx + spine_w // 2, spine_bot], fill=55)
    # soften spine
    spine_grad = _radial_gradient(cx, cy + 10, spine_w * 2, 210, 25)
    img = ImageChops.add(img, spine_grad)
    draw = ImageDraw.Draw(img)

    # --- Heart silhouette (left-shifted bright mass) ---
    heart_cx = cx - 15 + random.randint(-5, 5)
    heart_cy = cy + 55 + random.randint(-5, 5)
    heart_rx = 60 + random.randint(-5, 5)
    heart_ry = 55 + random.randint(-3, 3)
    heart_grad = _radial_gradient(heart_cx, heart_cy, heart_rx, heart_ry, 50)
    img = ImageChops.add(img, heart_grad)
    draw = ImageDraw.Draw(img)

    # --- Diaphragm domes ---
    # Right dome (slightly higher)
    r_dia_cy = cy + 135 + random.randint(-5, 5)
    l_dia_cy = cy + 145 + random.randint(-5, 5)
    # Draw bright arcs for diaphragm
    for i in range(4):
        draw.arc([cx + 15, r_dia_cy - 50 + i, cx + 195, r_dia_cy + 50 + i], 180, 360,
                 fill=45 + random.randint(-5, 5), width=2)
        draw.arc([cx - 195, l_dia_cy - 45 + i, cx - 15, l_dia_cy + 55 + i], 180, 360,
                 fill=45 + random.randint(-5, 5), width=2)

    # Below diaphragm: brighter (abdomen)
    draw.rectangle([cx - 210, r_dia_cy + 20, cx + 210, SIZE], fill=35)

    # --- Rib cage ---
    num_ribs = 10
    rib_spacing = 26 + random.randint(-1, 1)
    rib_start_y = cy - 155
    for i in range(num_ribs):
        rib_y = rib_start_y + i * rib_spacing
        rib_brightness = 32 + random.randint(-4, 4)
        rib_width = 2 if i < 7 else 1
        # Angle: ribs slope downward from spine
        angle_offset = i * 3 + random.randint(-1, 1)
        # Right ribs
        r_start = (cx + 10, rib_y)
        r_end = (cx + 170 + random.randint(-10, 10), rib_y + angle_offset)
        draw.line([r_start, r_end], fill=rib_brightness, width=rib_width)
        # Left ribs
        l_start = (cx - 10, rib_y)
        l_end = (cx - 170 + random.randint(-10, 10), rib_y + angle_offset)
        draw.line([l_start, l_end], fill=rib_brightness, width=rib_width)

    # --- Clavicles (bright diagonal lines at top) ---
    clav_y = cy - 165 + random.randint(-5, 5)
    clav_brightness = 55 + random.randint(-5, 5)
    # Right clavicle
    draw.line([(cx + 5, clav_y), (cx + 160, clav_y - 30 + random.randint(-5, 5))],
              fill=clav_brightness, width=3)
    # Left clavicle
    draw.line([(cx - 5, clav_y), (cx - 160, clav_y - 30 + random.randint(-5, 5))],
              fill=clav_brightness, width=3)

    # --- Scapulae (faint lateral bright regions) ---
    scap_grad_r = _radial_gradient(cx + 175, cy - 40, 45, 90, 20)
    scap_grad_l = _radial_gradient(cx - 175, cy - 40, 45, 90, 20)
    img = ImageChops.add(img, scap_grad_r)
    img = ImageChops.add(img, scap_grad_l)

    # --- Soft tissue shoulders ---
    shoulder_y = cy - 190
    draw = ImageDraw.Draw(img)
    draw.arc([cx + 100, shoulder_y - 40, cx + 260, shoulder_y + 60], 200, 340,
             fill=30, width=3)
    draw.arc([cx - 260, shoulder_y - 40, cx - 100, shoulder_y + 60], 200, 340,
             fill=30, width=3)

    # --- Trachea (faint dark line above carina) ---
    trachea_top = cy - 175
    trachea_bot = cy - 100
    draw.line([(cx, trachea_top), (cx, trachea_bot)], fill=12, width=6)

    # Slight Gaussian blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

    # Add film grain
    _add_noise(img, intensity=8)

    return img


# ---------------------------------------------------------------------------
# Pathology overlays
# ---------------------------------------------------------------------------


def apply_cardiomegaly(base: Image.Image, variant: int) -> Image.Image:
    """Enlarged heart: larger, brighter central cardiac silhouette."""
    img = base.copy()
    from PIL import ImageChops
    cx, cy = SIZE // 2, SIZE // 2
    # Progressively larger heart per variant
    scale = 1.3 + variant * 0.15
    heart_rx = int(75 * scale) + random.randint(-5, 5)
    heart_ry = int(70 * scale) + random.randint(-3, 3)
    heart_cx = cx - 10 + random.randint(-8, 8)
    heart_cy = cy + 50 + random.randint(-5, 5)
    heart = _radial_gradient(heart_cx, heart_cy, heart_rx, heart_ry, 65)
    img = ImageChops.add(img, heart)
    # Additional left-sided bulge
    bulge = _radial_gradient(heart_cx - 30, heart_cy + 10, heart_rx - 10, heart_ry - 10, 30)
    img = ImageChops.add(img, bulge)
    return img


def apply_pneumothorax(base: Image.Image, variant: int) -> Image.Image:
    """Dark region at lung apex with visceral pleural line."""
    img = base.copy()
    draw = ImageDraw.Draw(img)
    from PIL import ImageChops
    cx, cy = SIZE // 2, SIZE // 2

    # Choose side: variant 0,2 = right; 1 = left
    side = 1 if variant == 1 else -1  # 1 = right, -1 = left
    apex_cx = cx + side * 95
    apex_cy = cy - 120 - variant * 15

    # Dark region (collapsed lung / air)
    dark_region = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(dark_region)
    ptx_rx = 65 + variant * 10
    ptx_ry = 80 + variant * 15
    d.ellipse([apex_cx - ptx_rx, apex_cy - ptx_ry, apex_cx + ptx_rx, apex_cy + ptx_ry],
              fill=30)
    dark_region = dark_region.filter(ImageFilter.GaussianBlur(radius=8))
    # Subtract darkness from base
    inv_dark = ImageChops.invert(dark_region)
    img = ImageChops.multiply(img, inv_dark)

    # Visceral pleural line (bright thin arc)
    draw = ImageDraw.Draw(img)
    line_rx = ptx_rx - 10
    line_ry = ptx_ry - 10
    draw.arc([apex_cx - line_rx, apex_cy - line_ry, apex_cx + line_rx, apex_cy + line_ry],
             160, 380, fill=70 + random.randint(-5, 5), width=2)

    return img


def apply_pleural_effusion(base: Image.Image, variant: int) -> Image.Image:
    """White gradient at the base of the lung (meniscus sign)."""
    img = base.copy()
    from PIL import ImageChops
    cx = SIZE // 2

    # Side: variant 0 = bilateral, 1 = right, 2 = left
    effusion_height = 80 + variant * 25 + random.randint(-10, 10)

    overlay = Image.new("L", (SIZE, SIZE), 0)
    pixels = overlay.load()
    for y in range(SIZE):
        for x in range(SIZE):
            bottom_dist = SIZE - y
            if bottom_dist < effusion_height:
                # Meniscus: brighter at periphery and bottom
                frac = 1.0 - (bottom_dist / effusion_height)
                # Lateral meniscus curve (higher at edges)
                lateral_dist = abs(x - cx) / (SIZE / 2)
                meniscus_boost = lateral_dist * 0.4
                brightness = int(min(1.0, frac + meniscus_boost) * 70)

                if variant == 1 and x < cx - 30:
                    brightness = 0  # right-only
                elif variant == 2 and x > cx + 30:
                    brightness = 0  # left-only

                pixels[x, y] = brightness

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=6))
    img = ImageChops.add(img, overlay)
    return img


def apply_lung_opacity(base: Image.Image, variant: int) -> Image.Image:
    """Patchy lighter areas scattered in lung fields (ground-glass-like)."""
    img = base.copy()
    from PIL import ImageChops
    cx, cy = SIZE // 2, SIZE // 2

    num_patches = 3 + variant
    for _ in range(num_patches):
        # Random position within lung fields
        side = random.choice([-1, 1])
        px = cx + side * random.randint(40, 140)
        py = cy + random.randint(-100, 80)
        prx = random.randint(25, 55)
        pry = random.randint(20, 45)
        brightness = random.randint(25, 45)
        patch = _radial_gradient(px, py, prx, pry, brightness)
        img = ImageChops.add(img, patch)

    # Slight overall haze
    haze = _radial_gradient(cx, cy - 20, 160, 140, 10)
    img = ImageChops.add(img, haze)
    return img


def apply_consolidation(base: Image.Image, variant: int) -> Image.Image:
    """Dense white region in one lobe (air bronchograms)."""
    img = base.copy()
    from PIL import ImageChops
    cx, cy = SIZE // 2, SIZE // 2

    # Location varies: RLL, LLL, RML
    positions = [
        (cx + 80, cy + 60),   # Right lower lobe
        (cx - 85, cy + 65),   # Left lower lobe
        (cx + 70, cy - 10),   # Right middle lobe
    ]
    px, py = positions[variant % len(positions)]
    px += random.randint(-10, 10)
    py += random.randint(-10, 10)

    # Dense consolidation
    cons_rx = 55 + random.randint(-5, 10)
    cons_ry = 45 + random.randint(-5, 10)
    consolidation = _radial_gradient(px, py, cons_rx, cons_ry, 80)
    img = ImageChops.add(img, consolidation)

    # Air bronchograms: thin dark lines through the consolidation
    draw = ImageDraw.Draw(img)
    for i in range(3):
        bx = px - 20 + i * 15
        by_start = py - 25 + random.randint(-5, 5)
        by_end = py + 25 + random.randint(-5, 5)
        draw.line([(bx, by_start), (bx + random.randint(-8, 8), by_end)],
                  fill=15, width=1)

    return img


def apply_atelectasis(base: Image.Image, variant: int) -> Image.Image:
    """Linear bright bands with volume loss / asymmetry."""
    img = base.copy()
    draw = ImageDraw.Draw(img)
    from PIL import ImageChops
    cx, cy = SIZE // 2, SIZE // 2

    # Side: 0=right, 1=left, 2=bilateral
    side = variant % 3

    if side == 0 or side == 2:
        # Right-sided plate atelectasis
        for i in range(2 + variant):
            band_y = cy + 20 + i * 25 + random.randint(-5, 5)
            band_x1 = cx + 30
            band_x2 = cx + 160 + random.randint(-15, 15)
            draw.line([(band_x1, band_y), (band_x2, band_y + random.randint(-3, 3))],
                      fill=60 + random.randint(-5, 5), width=3)
        # Volume loss: slight mediastinal shift
        shift = _radial_gradient(cx + 15, cy, 50, 100, 15)
        img = ImageChops.add(img, shift)

    if side == 1 or side == 2:
        # Left-sided
        for i in range(2 + variant):
            band_y = cy + 30 + i * 25 + random.randint(-5, 5)
            band_x1 = cx - 30
            band_x2 = cx - 155 + random.randint(-15, 15)
            draw.line([(band_x1, band_y), (band_x2, band_y + random.randint(-3, 3))],
                      fill=58 + random.randint(-5, 5), width=3)

    # Elevated hemidiaphragm on affected side
    diaphragm_overlay = _radial_gradient(
        cx + (70 if side != 1 else -70),
        cy + 115, 80, 30, 25
    )
    img = ImageChops.add(img, diaphragm_overlay)

    return img


def apply_edema(base: Image.Image, variant: int) -> Image.Image:
    """Bilateral perihilar haze (bat-wing pattern), Kerley B lines."""
    img = base.copy()
    from PIL import ImageChops
    cx, cy = SIZE // 2, SIZE // 2

    # Perihilar haze (butterfly / bat-wing)
    severity = 0.6 + variant * 0.15
    r_haze = _radial_gradient(cx + 50, cy - 10, int(110 * severity), int(100 * severity),
                              int(50 * severity))
    l_haze = _radial_gradient(cx - 50, cy - 10, int(110 * severity), int(100 * severity),
                              int(50 * severity))
    img = ImageChops.add(img, r_haze)
    img = ImageChops.add(img, l_haze)

    # Central haze
    central = _radial_gradient(cx, cy + 10, int(80 * severity), int(70 * severity),
                               int(35 * severity))
    img = ImageChops.add(img, central)

    # Kerley B lines at bases (short horizontal lines at periphery)
    draw = ImageDraw.Draw(img)
    for side in [-1, 1]:
        base_x = cx + side * 140
        for i in range(4 + variant):
            ly = cy + 80 + i * 12 + random.randint(-3, 3)
            lx1 = base_x + random.randint(-10, 10)
            lx2 = lx1 + side * random.randint(15, 30)
            draw.line([(lx1, ly), (lx2, ly)], fill=50, width=1)

    # Cephalization (upper lobe vessel prominence)
    upper_haze = _radial_gradient(cx, cy - 90, 100, 60, int(20 * severity))
    img = ImageChops.add(img, upper_haze)

    return img


def apply_pneumonia(base: Image.Image, variant: int) -> Image.Image:
    """Focal bright patch in lower lobe with air bronchograms."""
    img = base.copy()
    from PIL import ImageChops
    cx, cy = SIZE // 2, SIZE // 2

    # Positions: RLL, LLL, lingula
    positions = [
        (cx + 85, cy + 70),
        (cx - 80, cy + 75),
        (cx - 60, cy + 20),
    ]
    px, py = positions[variant % len(positions)]
    px += random.randint(-8, 8)
    py += random.randint(-8, 8)

    # Patchy infiltrate (less homogeneous than consolidation)
    for _ in range(3):
        sub_px = px + random.randint(-20, 20)
        sub_py = py + random.randint(-15, 15)
        sub_rx = random.randint(20, 40)
        sub_ry = random.randint(18, 35)
        patch = _radial_gradient(sub_px, sub_py, sub_rx, sub_ry,
                                 random.randint(35, 55))
        img = ImageChops.add(img, patch)

    # Silhouette sign: obscured diaphragm or heart border
    silhouette = _radial_gradient(px, py + 30, 35, 25, 20)
    img = ImageChops.add(img, silhouette)

    return img


def apply_no_finding(base: Image.Image, variant: int) -> Image.Image:
    """Clean baseline -- just return the base with minor variation."""
    # Already clean; add very slight random variation
    img = base.copy()
    from PIL import ImageChops
    cx, cy = SIZE // 2, SIZE // 2
    # Tiny vascular markings
    draw = ImageDraw.Draw(img)
    for _ in range(6):
        side = random.choice([-1, 1])
        vx = cx + side * random.randint(30, 150)
        vy = cy + random.randint(-120, 80)
        length = random.randint(15, 40)
        angle = random.uniform(-0.5, 0.5)
        vx2 = vx + int(length * math.cos(angle))
        vy2 = vy + int(length * math.sin(angle))
        draw.line([(vx, vy), (vx2, vy2)], fill=30 + random.randint(-5, 5), width=1)
    return img


def apply_support_devices(base: Image.Image, variant: int) -> Image.Image:
    """Bright lines representing tubes/lines (ETT, central line, chest tube)."""
    img = base.copy()
    draw = ImageDraw.Draw(img)
    cx, cy = SIZE // 2, SIZE // 2

    line_brightness = 85 + random.randint(-5, 10)

    if variant == 0 or variant == 2:
        # Endotracheal tube (ETT): vertical line down trachea
        ett_x = cx + random.randint(-5, 5)
        draw.line([(ett_x, cy - 190), (ett_x + random.randint(-3, 3), cy - 80)],
                  fill=line_brightness, width=3)
        # Tip marker
        draw.ellipse([ett_x - 4, cy - 84, ett_x + 4, cy - 76], fill=line_brightness)

    if variant == 0 or variant == 1:
        # Central venous catheter (right IJ to SVC)
        points = [
            (cx + 80 + random.randint(-5, 5), cy - 175),
            (cx + 50, cy - 140),
            (cx + 20, cy - 110),
            (cx + 5, cy - 70),
        ]
        draw.line(points, fill=line_brightness, width=2)
        # Tip
        draw.ellipse([points[-1][0] - 3, points[-1][1] - 3,
                      points[-1][0] + 3, points[-1][1] + 3], fill=line_brightness)

    if variant == 1 or variant == 2:
        # Chest tube (lateral, curving inferiorly)
        ct_side = 1 if variant == 1 else -1
        ct_points = [
            (cx + ct_side * 170, cy + 30),
            (cx + ct_side * 140, cy + 60),
            (cx + ct_side * 100, cy + 90),
            (cx + ct_side * 80, cy + 110),
        ]
        draw.line(ct_points, fill=line_brightness - 10, width=3)

    if variant == 2:
        # NG tube
        ng_x = cx + random.randint(-8, 8)
        ng_points = [
            (ng_x, cy - 185),
            (ng_x - 5, cy - 100),
            (ng_x - 10, cy + 30),
            (ng_x - 15, cy + 130),
        ]
        draw.line(ng_points, fill=line_brightness - 15, width=2)

    return img


def apply_fracture(base: Image.Image, variant: int) -> Image.Image:
    """Small bright discontinuity along rib with cortical step-off."""
    img = base.copy()
    draw = ImageDraw.Draw(img)
    from PIL import ImageChops
    cx, cy = SIZE // 2, SIZE // 2

    # Fracture locations along different ribs
    fracture_positions = [
        (cx + 120, cy - 50),   # Right lateral mid ribs
        (cx - 115, cy - 20),   # Left lateral
        (cx + 100, cy + 30),   # Right lower lateral
    ]
    fx, fy = fracture_positions[variant % len(fracture_positions)]
    fx += random.randint(-10, 10)
    fy += random.randint(-5, 5)

    # Cortical disruption: bright spot with offset
    draw.line([(fx - 12, fy - 2), (fx - 2, fy)], fill=70, width=3)
    draw.line([(fx + 2, fy + 3), (fx + 12, fy + 1)], fill=70, width=3)
    # Gap between fragments
    draw.rectangle([fx - 2, fy - 2, fx + 2, fy + 4], fill=15)

    # Subtle callus / periosteal reaction (hazy bright area around fracture)
    callus = _radial_gradient(fx, fy, 18, 12, 25)
    img = ImageChops.add(img, callus)

    # Second fracture on variant 2 (multiple rib fractures)
    if variant == 2:
        fx2 = fx + random.randint(-20, 20)
        fy2 = fy + 26 + random.randint(-3, 3)
        draw = ImageDraw.Draw(img)
        draw.line([(fx2 - 10, fy2), (fx2 - 2, fy2 + 1)], fill=68, width=2)
        draw.line([(fx2 + 2, fy2 + 2), (fx2 + 10, fy2 + 1)], fill=68, width=2)
        draw.rectangle([fx2 - 2, fy2 - 1, fx2 + 2, fy2 + 3], fill=15)

    return img


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

PATHOLOGY_FN = {
    "Cardiomegaly": apply_cardiomegaly,
    "Pneumothorax": apply_pneumothorax,
    "Pleural Effusion": apply_pleural_effusion,
    "Lung Opacity": apply_lung_opacity,
    "Consolidation": apply_consolidation,
    "Atelectasis": apply_atelectasis,
    "Edema": apply_edema,
    "Pneumonia": apply_pneumonia,
    "No Finding": apply_no_finding,
    "Support Devices": apply_support_devices,
    "Fracture": apply_fracture,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    total = 0
    for category in CATEGORIES:
        cat_dir = os.path.join(OUTPUT_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)

        apply_fn = PATHOLOGY_FN[category]

        for i in range(IMAGES_PER_CATEGORY):
            base = create_base_xray(variation=i)
            img = apply_fn(base, variant=i)

            # Final post-processing: slight contrast adjustment and blur
            img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

            # Ensure pixel values are in valid range
            img = img.point(lambda p: min(255, max(0, p)))

            filename = f"case_{i + 1:03d}.png"
            filepath = os.path.join(cat_dir, filename)
            img.save(filepath, "PNG")
            total += 1
            print(f"  [{category}] {filename}")

    print(f"\nGenerated {total} images across {len(CATEGORIES)} categories.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
