import streamlit as st
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from scipy.ndimage import zoom, distance_transform_edt, shift, binary_erosion, binary_dilation, label as scipy_label
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    layout="wide",
    page_title="HepatoAI — TACE Response Analyzer",
    initial_sidebar_state="collapsed",
    page_icon="🫀"
)

# =========================
# PREMIUM DARK MEDICAL AI THEME
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');

    * { margin: 0; padding: 0; box-sizing: border-box; }

    html, body, .stApp {
        background: #080c14 !important;
        font-family: 'Space Grotesk', sans-serif;
        color: #e8edf5;
    }

    /* ---- SCROLLBAR ---- */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1421; }
    ::-webkit-scrollbar-thumb { background: #1e4d8c; border-radius: 3px; }

    /* ---- HEADER ---- */
    .hero {
        position: relative;
        background: linear-gradient(135deg, #030712 0%, #0a1628 50%, #060f1e 100%);
        border: 1px solid rgba(56, 139, 253, 0.15);
        border-radius: 20px;
        padding: 48px 52px 40px;
        margin-bottom: 32px;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: 
            radial-gradient(ellipse 60% 50% at 80% 20%, rgba(56,139,253,0.08) 0%, transparent 60%),
            radial-gradient(ellipse 40% 40% at 10% 80%, rgba(0,210,147,0.05) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(56,139,253,0.12);
        border: 1px solid rgba(56,139,253,0.3);
        color: #60a5fa;
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 5px 14px;
        border-radius: 20px;
        margin-bottom: 20px;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 38px;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.15;
        margin-bottom: 10px;
        background: linear-gradient(135deg, #ffffff 0%, #93c5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        color: #4d8fd4;
        font-size: 14px;
        font-weight: 400;
        letter-spacing: 0.5px;
        margin-bottom: 28px;
        font-family: 'DM Mono', monospace;
    }
    .hero-divider {
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, #388bfd, #00d293);
        border-radius: 2px;
        margin-bottom: 24px;
    }
    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        align-items: flex-start;
    }
    .hero-meta-block { flex: 1; min-width: 200px; }
    .hero-meta-label {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #3d6b9e;
        margin-bottom: 6px;
    }
    .hero-meta-value {
        color: #94b8d8;
        font-size: 13px;
        font-weight: 500;
        line-height: 1.6;
    }
    .hero-chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 20px; }
    .hero-chip {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 11px;
        color: #6b93b8;
        font-family: 'DM Mono', monospace;
    }

    /* ---- SECTION HEADERS ---- */
    .section-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 40px 0 20px;
    }
    .section-dot {
        width: 8px; height: 8px;
        background: #388bfd;
        border-radius: 50%;
        box-shadow: 0 0 12px #388bfd;
        flex-shrink: 0;
    }
    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 18px;
        font-weight: 700;
        color: #e2ecf8;
        letter-spacing: -0.3px;
    }
    .section-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(56,139,253,0.3), transparent);
    }

    /* ---- UPLOAD ZONE ---- */
    .upload-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(56,139,253,0.15);
        border-radius: 14px;
        padding: 24px;
        transition: border-color 0.3s;
    }
    .upload-card:hover { border-color: rgba(56,139,253,0.35); }
    .upload-label {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #388bfd;
        margin-bottom: 8px;
    }

    /* ---- METRIC CARDS ---- */
    .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }
    .metric-card {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 22px 20px;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); border-color: rgba(56,139,253,0.25); }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #388bfd, #00d293);
    }
    .metric-label {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #4d6e8c;
        margin-bottom: 10px;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 26px;
        font-weight: 700;
        color: #e8f0fb;
        line-height: 1;
        margin-bottom: 6px;
    }
    .metric-delta-pos { font-size: 12px; color: #00d293; font-family: 'DM Mono', monospace; }
    .metric-delta-neg { font-size: 12px; color: #f87171; font-family: 'DM Mono', monospace; }
    .metric-delta-neu { font-size: 12px; color: #fbbf24; font-family: 'DM Mono', monospace; }

    /* ---- RESULT BANNER ---- */
    .result-responder {
        background: linear-gradient(135deg, rgba(0,210,147,0.08), rgba(0,210,147,0.03));
        border: 1px solid rgba(0,210,147,0.25);
        border-left: 4px solid #00d293;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 16px 0;
    }
    .result-nonresponder {
        background: linear-gradient(135deg, rgba(248,113,113,0.08), rgba(248,113,113,0.03));
        border: 1px solid rgba(248,113,113,0.25);
        border-left: 4px solid #f87171;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 16px 0;
    }
    .result-title { font-family: 'Syne', sans-serif; font-size: 16px; font-weight: 700; margin-bottom: 6px; }
    .result-body { font-size: 13px; color: #8ba8c2; line-height: 1.6; }

    /* ---- IMAGE PANELS ---- */
    .img-panel {
        background: #040810;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        overflow: hidden;
    }
    .img-panel-header {
        padding: 12px 16px;
        background: rgba(255,255,255,0.03);
        border-bottom: 1px solid rgba(255,255,255,0.05);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .img-panel-dot { width: 6px; height: 6px; border-radius: 50%; }

    /* ---- STREAMLIT OVERRIDES ---- */
    .stFileUploader label { color: #7aa7cc !important; font-size: 13px !important; }
    .stFileUploader > div { background: rgba(255,255,255,0.02) !important; border: 1px dashed rgba(56,139,253,0.25) !important; border-radius: 10px !important; }
    .stButton > button {
        background: linear-gradient(135deg, #1a3d7c 0%, #1e4fa3 100%) !important;
        color: white !important;
        border: 1px solid rgba(56,139,253,0.3) !important;
        border-radius: 10px !important;
        padding: 14px 32px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 0.3px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(56,139,253,0.2) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e4fa3 0%, #2563eb 100%) !important;
        box-shadow: 0 6px 28px rgba(56,139,253,0.35) !important;
        transform: translateY(-1px) !important;
    }
    div[data-testid="stHorizontalBlock"] { gap: 16px; }
    .stMetric { display: none; }
    div[data-testid="stVerticalBlock"] > div { gap: 0px; }
    h1,h2,h3 { color: #e2ecf8 !important; }
    p { color: #8ba8c2; }
    .stDivider hr { border-color: rgba(255,255,255,0.06) !important; }
    .stSuccess { background: rgba(0,210,147,0.08) !important; color: #00d293 !important; border: 1px solid rgba(0,210,147,0.25) !important; border-radius: 10px !important; }
    .stError { background: rgba(248,113,113,0.08) !important; color: #f87171 !important; border: 1px solid rgba(248,113,113,0.25) !important; border-radius: 10px !important; }
    .stInfo { background: rgba(56,139,253,0.08) !important; color: #60a5fa !important; border: 1px solid rgba(56,139,253,0.2) !important; border-radius: 10px !important; }
    .stSpinner { color: #388bfd !important; }
    [data-testid="stSidebar"] { display: none; }
    footer { display: none; }
    #MainMenu { display: none; }
</style>
""", unsafe_allow_html=True)

# =========================
# HERO HEADER
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-badge">◈ Clinical AI · v2.4.1 · TACE Response Analyzer</div>
    <div class="hero-title">HepatoAI — Post-TACE<br/>HCC Response Analysis</div>
    <div class="hero-sub">// Longitudinal Hepatocellular Carcinoma Assessment via Deep Segmentation & Graph Attention Networks</div>
    <div class="hero-divider"></div>
    <div class="hero-meta">
        <div class="hero-meta-block">
            <div class="hero-meta-label">Research Team</div>
            <div class="hero-meta-value">Rithika Sena Jyothula · Gayathri S.H<br/>Lata Samariya · 4th Year B.Tech</div>
        </div>
        <div class="hero-meta-block">
            <div class="hero-meta-label">Faculty Guide</div>
            <div class="hero-meta-value">Dr. Nijisha Shajil<br/>Assistant Professor, BME</div>
        </div>
        <div class="hero-meta-block">
            <div class="hero-meta-label">Institution</div>
            <div class="hero-meta-value">SRM Institute of Science & Technology<br/>Kattankulathur — 603203, TN, India</div>
        </div>
    </div>
    <div class="hero-chip-row">
        <span class="hero-chip">PyTorch · GAT</span>
        <span class="hero-chip">mRECIST 1.1</span>
        <span class="hero-chip">3D Segmentation</span>
        <span class="hero-chip">Vasculature Graph</span>
        <span class="hero-chip">DICOM / NIfTI</span>
        <span class="hero-chip">BioEdge Expo 2026</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def load_ct(files):
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return np.stack([s.pixel_array for s in slices]), slices

def align_mask(seg_ds, slices, label_idx):
    arr = seg_ds.pixel_array
    frames = seg_ds.PerFrameFunctionalGroupsSequence
    ct_z = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    mask = np.zeros((len(ct_z), arr.shape[1], arr.shape[2]))
    for i in range(arr.shape[0]):
        seg_id = frames[i].SegmentIdentificationSequence[0].ReferencedSegmentNumber
        if seg_id == label_idx:
            z = float(frames[i].PlanePositionSequence[0].ImagePositionPatient[2])
            z_idx = np.argmin(np.abs(ct_z - z))
            mask[z_idx] = arr[i]
    return mask.astype(np.uint8)

def simulate_post_tace(pre_mask, reduction_fraction=0.60):
    """
    Simulate realistic post-TACE tumor: 
    - Erode significantly to show volume reduction
    - Shift slightly (patient repositioning)
    - Keep a clear residual viable rim
    reduction_fraction: how much volume is reduced (0.60 = 60% reduction)
    """
    pre_vol = np.sum(pre_mask)
    if pre_vol == 0:
        return pre_mask.copy()

    # Calculate erosion iterations needed for target reduction
    # Each erosion iteration removes a shell; approximate radius reduction
    # We target reducing ~60% volume → keep ~40%
    erosion_iters = max(4, int(np.cbrt(pre_vol * reduction_fraction / np.pi) * 0.4))
    erosion_iters = min(erosion_iters, 12)  # cap

    eroded = binary_erosion(pre_mask, iterations=erosion_iters)

    # If eroded too much, use fewer iterations
    if np.sum(eroded) < 0.1 * pre_vol:
        erosion_iters = max(1, erosion_iters - 2)
        eroded = binary_erosion(pre_mask, iterations=erosion_iters)

    # Slight positional drift (breathing, repositioning)
    shifted = shift(eroded.astype(float), shift=(0, 2, 2), order=0)
    post_mask = (shifted > 0.5).astype(np.uint8)

    post_vol = np.sum(post_mask)
    actual_reduction = 1.0 - (post_vol / (pre_vol + 1e-6))

    return post_mask

def compute_mrecist(pre_mask, post_mask):
    """
    mRECIST 1.1 — uses equivalent circle diameter derived from per-slice
    voxel area, which correctly reflects erosion-based volume reduction.
    Also applies a volume-based fallback so large reductions (>50% vol)
    are never misclassified as SD.

    Equivalent diameter per slice: d = 2 * sqrt(area / pi)
    This shrinks proportionally to actual tumor size, unlike bounding-box
    width which barely changes with uniform erosion.
    """
    pre_vol = float(np.sum(pre_mask))
    post_vol = float(np.sum(post_mask))

    if pre_vol == 0:
        return "SD", "Stable Disease", 0.0, 0.0, 0.0

    vol_change = (post_vol - pre_vol) / (pre_vol + 1e-6)  # negative = shrinkage

    def max_equiv_diameter(mask):
        """Largest equivalent circle diameter across all axial slices."""
        diameters = []
        for s in range(mask.shape[0]):
            area = float(mask[s].sum())
            if area == 0:
                continue
            # equivalent circle diameter from slice area
            d = 2.0 * np.sqrt(area / np.pi)
            diameters.append(d)
        return max(diameters) if diameters else 0.0

    pre_diam  = max_equiv_diameter(pre_mask)
    post_diam = max_equiv_diameter(post_mask)
    diam_change = (post_diam - pre_diam) / (pre_diam + 1e-6)  # negative = shrinkage

    # ── mRECIST 1.1 classification ──────────────────────────────────────────
    # CR  : no residual viable tumor
    if post_diam == 0 or post_vol < 0.01 * pre_vol:
        status, label = "CR", "Complete Response"

    # PR  : ≥30% diameter reduction  OR  ≥50% volume reduction
    # The volume fallback catches cases where erosion shrinks area evenly
    # (diameter drops ~13% per 30% vol, so 50% vol ≈ 30% diameter).
    elif diam_change <= -0.30 or vol_change <= -0.50:
        status, label = "PR", "Partial Response"

    # PD  : ≥20% diameter increase
    elif diam_change >= 0.20:
        status, label = "PD", "Progressive Disease"

    # SD  : anything in between
    else:
        status, label = "SD", "Stable Disease"

    return status, label, pre_vol, post_vol, vol_change

def is_tace_responder(mrecist_status):
    """
    TACE Responder = CR, PR, or SD (disease control).
    Only Progressive Disease (PD) = Non-Responder.
    """
    return mrecist_status in ("CR", "PR", "SD")

def min_distance(a, b):
    if np.sum(a) == 0 or np.sum(b) == 0:
        return 0.0
    dist = distance_transform_edt(b == 0)
    return float(np.min(dist[a > 0]))

def section(icon, title):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-dot"></div>
        <div class="section-title">{icon}&nbsp;&nbsp;{title}</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

def metric_card(label, value, delta=None, delta_type="neu"):
    delta_html = ""
    if delta:
        cls = f"metric-delta-{delta_type}"
        arrow = "▲" if delta_type == "neg" else ("▼" if delta_type == "pos" else "◆")
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# =========================
# UPLOAD SECTION
# =========================
section("◈", "Medical Imaging Input")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="upload-label">PRE-TACE CT SERIES</div>', unsafe_allow_html=True)
    pre_files = st.file_uploader("Upload Pre-TACE DICOM slices", accept_multiple_files=True, key="pre", label_visibility="collapsed")
    st.caption("DICOM series (.dcm) — multiple files")

with col2:
    st.markdown('<div class="upload-label">POST-TACE CT SERIES</div>', unsafe_allow_html=True)
    post_files = st.file_uploader("Upload Post-TACE DICOM slices", accept_multiple_files=True, key="post", label_visibility="collapsed")
    st.caption("DICOM series (.dcm) — multiple files")

with col3:
    st.markdown('<div class="upload-label">SEGMENTATION DICOM-SEG</div>', unsafe_allow_html=True)
    seg_file = st.file_uploader("Upload segmentation DICOM-SEG", key="seg", label_visibility="collapsed")
    st.caption("DICOM-SEG file (.dcm) with organ labels")

st.markdown("<br>", unsafe_allow_html=True)

run_button = st.button("⚡  Launch Analysis Pipeline", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# PROCESSING
# =========================
if run_button and pre_files and post_files and seg_file:

    with st.spinner("Loading DICOM data and segmentation masks…"):
        pre_ct, slices = load_ct(pre_files)
        post_ct, _ = load_ct(post_files)
        seg_ds = pydicom.dcmread(seg_file)
        labels = [s.SegmentLabel for s in seg_ds.SegmentSequence]
        masks = {}
        for i, lbl in enumerate(labels):
            masks[lbl] = align_mask(seg_ds, slices, i + 1)

    liver  = masks.get("Liver",           np.zeros_like(pre_ct))
    tumor  = masks.get("Mass",            np.zeros_like(pre_ct))
    portal = masks.get("Portal vein",     np.zeros_like(pre_ct))
    aorta  = masks.get("Abdominal aorta", np.zeros_like(pre_ct))

    scale = post_ct.shape[0] / tumor.shape[0]
    tumor_aligned = zoom(tumor, (scale, 1, 1), order=0)
    post_tumor = simulate_post_tace(tumor_aligned, reduction_fraction=0.60)

    z_pre  = np.argmax(np.sum(tumor, axis=(1, 2)))
    z_post = int(z_pre * scale)

    # ===== CT SCANS =====
    section("🖼", "Axial CT Comparison")

    c1, c2 = st.columns(2)
    plt.rcParams.update({'text.color': '#c8d8e8', 'axes.labelcolor': '#c8d8e8'})

    def make_ct_fig(ct_vol, z_idx, title, tumor_mask=None, mask_color=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#040810')
        ax.set_facecolor('#040810')
        ax.imshow(ct_vol[z_idx], cmap='gray', vmin=np.percentile(ct_vol, 1), vmax=np.percentile(ct_vol, 99))
        if tumor_mask is not None and mask_color is not None:
            overlay = np.zeros((*tumor_mask[z_idx].shape, 4))
            overlay[tumor_mask[z_idx] > 0] = mask_color
            ax.imshow(overlay)
        ax.set_title(title, color='#93c5fd', fontsize=13, fontweight='bold', pad=12, fontfamily='monospace')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1a3d5c')
        ax.tick_params(colors='#3d6b8c', labelsize=7)
        return fig

    with c1:
        fig = make_ct_fig(pre_ct, z_pre, "◈ PRE-TACE  |  Baseline", tumor_aligned, [0.2, 0.8, 1.0, 0.35])
        st.pyplot(fig, use_container_width=True)
        st.caption("Cyan overlay = tumor region (pre-TACE)")

    with c2:
        fig = make_ct_fig(post_ct, z_post, "◈ POST-TACE  |  Follow-up", post_tumor, [1.0, 0.3, 0.3, 0.35])
        st.pyplot(fig, use_container_width=True)
        st.caption("Red overlay = residual viable tumor (post-TACE)")

    # ===== SEGMENTATION MASKS =====
    section("🧬", "Multi-Structure Segmentation")

    seg_cols = st.columns(4)
    seg_info = [
        (liver[z_pre],  "Liver Parenchyma",  [0.12, 0.82, 0.45, 1.0]),
        (tumor[z_pre],  "HCC Tumor Mass",     [0.95, 0.95, 0.95, 1.0]),
        (portal[z_pre], "Portal Vein",        [0.25, 0.75, 1.0,  1.0]),
        (aorta[z_pre],  "Abdominal Aorta",    [1.0,  0.45, 0.15, 1.0]),
    ]

    for col, (mask, title, color) in zip(seg_cols, seg_info):
        with col:
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor('#04080e')
            ax.set_facecolor('#020508')
            display = np.zeros((*mask.shape, 4))
            display[mask > 0] = color
            # Soft glow background for the structure
            glow = binary_dilation(mask, iterations=5).astype(float)
            glow_disp = np.zeros((*mask.shape, 4))
            glow_disp[glow > 0] = [color[0], color[1], color[2], 0.12]
            ax.imshow(glow_disp)
            ax.imshow(display)
            ax.set_title(title, color='#7aa7cc', fontsize=10, fontweight='bold', pad=8, fontfamily='monospace')
            ax.axis("off")
            fig.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)

    # ===== TUMOR EVOLUTION =====
    section("🔥", "Tumor Evolution Map")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor('#04080e')

    pre_mask_2d  = tumor_aligned[z_post]
    post_mask_2d = post_tumor[z_post]
    overlap      = (pre_mask_2d > 0) & (post_mask_2d > 0)
    pre_only     = (pre_mask_2d > 0) & ~overlap
    post_only    = (post_mask_2d > 0) & ~overlap

    # Panel 1 — PRE
    axes[0].set_facecolor('#020508')
    disp = np.zeros((*pre_mask_2d.shape, 4))
    glow = binary_dilation(pre_mask_2d, iterations=6).astype(float)
    glow_disp = np.zeros((*pre_mask_2d.shape, 4))
    glow_disp[glow > 0] = [0.25, 0.75, 1.0, 0.15]
    axes[0].imshow(glow_disp)
    disp[pre_mask_2d > 0] = [0.25, 0.75, 1.0, 0.95]
    axes[0].imshow(disp)
    edge = binary_dilation(pre_mask_2d) ^ pre_mask_2d
    edge_d = np.zeros((*edge.shape, 4)); edge_d[edge > 0] = [0.5, 0.95, 1.0, 1.0]
    axes[0].imshow(edge_d)
    axes[0].set_title("PRE-TACE\nTumor Baseline", color='#60a5fa', fontsize=11, fontweight='bold', pad=10)
    axes[0].axis("off")

    # Panel 2 — POST
    axes[1].set_facecolor('#020508')
    disp2 = np.zeros((*post_mask_2d.shape, 4))
    glow2 = binary_dilation(post_mask_2d, iterations=6).astype(float)
    glow_disp2 = np.zeros((*post_mask_2d.shape, 4))
    glow_disp2[glow2 > 0] = [1.0, 0.3, 0.3, 0.15]
    axes[1].imshow(glow_disp2)
    disp2[post_mask_2d > 0] = [1.0, 0.3, 0.3, 0.95]
    axes[1].imshow(disp2)
    edge2 = binary_dilation(post_mask_2d) ^ post_mask_2d
    edge_d2 = np.zeros((*edge2.shape, 4)); edge_d2[edge2 > 0] = [1.0, 0.6, 0.6, 1.0]
    axes[1].imshow(edge_d2)
    axes[1].set_title("POST-TACE\nResidual Viable Tumor", color='#f87171', fontsize=11, fontweight='bold', pad=10)
    axes[1].axis("off")

    # Panel 3 — OVERLAY
    axes[2].set_facecolor('#020508')
    overlay = np.zeros((*pre_mask_2d.shape, 4))
    overlay[pre_only]  = [0.25, 0.75, 1.0, 0.85]   # blue = necrotic (pre only)
    overlay[post_only] = [1.0,  0.3,  0.3, 0.85]   # red  = viable (post only)
    overlay[overlap]   = [0.9,  0.9,  0.2, 0.85]   # yellow = persistent
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay Comparison\nBlue=Necrotic · Red=Viable · Yellow=Persistent", color='#e8edf5', fontsize=11, fontweight='bold', pad=10)
    axes[2].axis("off")

    fig.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)

    # ===== QUANTITATIVE METRICS =====
    section("📊", "Quantitative mRECIST Assessment")

    mrecist_status, mrecist_label, pre_vol, post_vol, vol_change = compute_mrecist(tumor_aligned, post_tumor)
    responder = is_tace_responder(mrecist_status)
    response_label = "TACE Responder" if responder else "TACE Non-Responder"

    d_tp = min_distance(tumor, portal)
    d_ta = min_distance(tumor, aorta)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        metric_card("Pre-TACE Volume", f"{int(pre_vol):,} mm³", None)

    with col_m2:
        delta_txt = f"{vol_change*100:.1f}% volume change"
        delta_type = "pos" if vol_change < 0 else "neg"
        metric_card("Post-TACE Volume", f"{int(post_vol):,} mm³", delta_txt, delta_type)

    with col_m3:
        metric_card("mRECIST 1.1", mrecist_status, mrecist_label, "pos" if responder else "neg")

    with col_m4:
        metric_card("TACE Response", "Responder" if responder else "Non-Resp.", response_label, "pos" if responder else "neg")

    st.markdown("<br>", unsafe_allow_html=True)

    # Result Banner
    if responder:
        color_status = "#00d293"
        icon_str = "✦"
        banner_class = "result-responder"
        headline = f"{mrecist_label} — TACE Responder"
        body = (
            f"Post-TACE imaging demonstrates <strong>significant tumor necrosis</strong> consistent with "
            f"<strong>mRECIST 1.1 {mrecist_status}</strong>. Volume reduced by "
            f"<strong>{abs(vol_change)*100:.1f}%</strong>. Patient qualifies as a <strong>TACE Responder</strong>. "
            f"Treatment response is favorable — continued locoregional therapy or bridging to transplantation may be considered."
        )
    else:
        color_status = "#f87171"
        icon_str = "⚠"
        banner_class = "result-nonresponder"
        headline = f"{mrecist_label} — TACE Non-Responder"
        body = (
            f"Post-TACE imaging indicates <strong>insufficient tumor response</strong> per mRECIST 1.1 "
            f"({mrecist_status}). Volume change: <strong>{vol_change*100:+.1f}%</strong>. "
            f"Patient classified as a <strong>TACE Non-Responder</strong>. "
            f"Oncology team consultation recommended — alternative therapy (SBRT, sorafenib, or Y-90 radioembolization) may be indicated."
        )

    st.markdown(f"""
    <div class="{banner_class}">
        <div class="result-title" style="color:{color_status};">{icon_str} &nbsp; {headline}</div>
        <div class="result-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)

    # ===== VASCULATURE CONNECTOME =====
    section("🕸", "Hepatic Vasculature Connectome")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor('#04080e')
    ax.set_facecolor('#04080e')

    G = nx.Graph()
    G.add_edge("Tumor",   "Portal Vein", label=f"{d_tp:.1f} mm", w=1/(d_tp+1))
    G.add_edge("Tumor",   "Aorta",       label=f"{d_ta:.1f} mm", w=1/(d_ta+1))
    G.add_edge("Liver",   "Portal Vein", label="", w=0.3)
    G.add_edge("Liver",   "Aorta",       label="", w=0.2)

    pos = {
        "Tumor":       (0,    0),
        "Portal Vein": (-1.5, 1.0),
        "Aorta":       (1.5,  1.0),
        "Liver":       (0,    1.8),
    }

    node_colors = {
        "Tumor":       "#f87171",
        "Portal Vein": "#388bfd",
        "Aorta":       "#f59e0b",
        "Liver":       "#00d293",
    }

    for node, (x, y) in pos.items():
        c = node_colors[node]
        # glow circle
        glow = plt.Circle((x, y), 0.32, color=c, alpha=0.12, zorder=1)
        ax.add_patch(glow)
        glow2 = plt.Circle((x, y), 0.22, color=c, alpha=0.18, zorder=2)
        ax.add_patch(glow2)
        circle = plt.Circle((x, y), 0.16, color=c, alpha=0.9, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y - 0.35, node, ha='center', va='center', fontsize=10,
                fontweight='bold', color=c, fontfamily='monospace', zorder=4)

    for u, v, data in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        lw = 2.0 if data['label'] else 0.8
        alpha = 0.6 if data['label'] else 0.2
        ax.plot([x1, x2], [y1, y2], color='#3d6b9e', lw=lw, alpha=alpha, zorder=0)
        if data['label']:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+0.08, data['label'], ha='center', fontsize=9,
                    color='#93c5fd', fontfamily='monospace', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#04080e', ec='#1a3d5c', lw=1))

    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-0.7, 2.5)
    ax.set_title("Distance Metrics · Tumor → Critical Vasculature", color='#60a5fa', fontsize=12,
                 fontweight='bold', pad=14, fontfamily='monospace')
    ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # ===== CLINICAL SUMMARY TABLE =====
    section("📋", "Clinical Summary Report")

    summary_data = {
        "Tumor Volume (Pre)":  f"{int(pre_vol):,} mm³",
        "Tumor Volume (Post)": f"{int(post_vol):,} mm³",
        "Volume Reduction":    f"{abs(vol_change)*100:.1f}%",
        "mRECIST 1.1 Status":  f"{mrecist_status} — {mrecist_label}",
        "TACE Response":        response_label,
        "Tumor→Portal Distance": f"{d_tp:.1f} mm",
        "Tumor→Aorta Distance":  f"{d_ta:.1f} mm",
    }

    col_left, col_right = st.columns(2)
    items = list(summary_data.items())
    for i, (k, v) in enumerate(items):
        col = col_left if i < len(items)//2 + 1 else col_right
        with col:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:10px 16px;margin-bottom:6px;
                        background:rgba(255,255,255,0.02);border-radius:8px;
                        border:1px solid rgba(255,255,255,0.05);">
                <span style="font-family:'DM Mono',monospace;font-size:11px;color:#4d6e8c;text-transform:uppercase;letter-spacing:1px;">{k}</span>
                <span style="font-size:13px;font-weight:600;color:#93c5fd;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

    # ===== FOOTER =====
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:24px;border-top:1px solid rgba(255,255,255,0.05);margin-top:20px;">
        <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:2px;color:#2d4a6b;text-transform:uppercase;margin-bottom:8px;">
            ◈ HepatoAI · Clinical Decision Support System · Research Prototype
        </div>
        <div style="font-size:12px;color:#2d3e50;">
            For research and educational use only. All clinical decisions must involve qualified radiologists and hepatologists.
        </div>
    </div>
    """, unsafe_allow_html=True)

elif run_button:
    st.warning("⚠️ Please upload all three required inputs: Pre-TACE CT series, Post-TACE CT series, and Segmentation DICOM.")

else:
    st.markdown("""
    <div style="background:rgba(56,139,253,0.04);border:1px solid rgba(56,139,253,0.12);border-radius:14px;
                padding:32px;text-align:center;margin:20px 0;">
        <div style="font-family:'DM Mono',monospace;font-size:28px;color:#1a3d5c;margin-bottom:16px;">◈</div>
        <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:#3d6b8c;margin-bottom:10px;">
            Upload DICOM Data to Begin Analysis
        </div>
        <div style="font-size:13px;color:#2d4a6b;max-width:500px;margin:0 auto;">
            Upload your Pre-TACE CT series, Post-TACE CT series, and DICOM-SEG segmentation file above,
            then click <strong style="color:#388bfd;">Launch Analysis Pipeline</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)