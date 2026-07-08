import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# Import synthetic generators
from generators import (
    generate_brain_mri_3d,
    generate_lung_ct_3d,
    generate_pet_ct_3d,
    simulate_pipeline_stages,
    create_ellipsoid_mask
)

# Page Configuration
st.set_page_config(
    page_title="MedInsight 3D - Medical Imaging AI Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM STYLE SHEET (CSS) -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Main font and styling override */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Background dark mode styling */
.stApp {
    background-color: #0b0f19;
    background-image: 
        radial-gradient(at 0% 0%, rgba(30, 58, 138, 0.15) 0, transparent 50%), 
        radial-gradient(at 50% 0%, rgba(88, 28, 135, 0.12) 0, transparent 50%), 
        radial-gradient(at 100% 0%, rgba(30, 58, 138, 0.15) 0, transparent 50%);
    background-attachment: fixed;
}

/* Glassmorphism card utility */
.glass-card {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(0, 210, 255, 0.2);
    transform: translateY(-2px);
    box-shadow: 0 12px 40px 0 rgba(0, 210, 255, 0.08);
}

/* Premium gradient text */
.gradient-text {
    background: linear-gradient(135deg, #00D2FF 0%, #3B82F6 50%, #8B5CF6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #080c14;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Custom indicators */
.badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-right: 5px;
}

.badge-blue { background: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.3); }
.badge-purple { background: rgba(139, 92, 246, 0.2); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.3); }
.badge-green { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
.badge-orange { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }

/* Horizontal steps indicator */
.steps-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 20px 0 30px 0;
    position: relative;
}

.steps-container::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 0;
    width: 100%;
    height: 2px;
    background: rgba(255, 255, 255, 0.1);
    z-index: 1;
}

.step-node {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #0f172a;
    border: 2px solid rgba(255, 255, 255, 0.2);
    z-index: 2;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: bold;
    color: rgba(255, 255, 255, 0.6);
    transition: all 0.3s ease;
}

.step-node.active {
    background: #3b82f6;
    border-color: #00d2ff;
    color: white;
    box-shadow: 0 0 15px rgba(0, 210, 255, 0.5);
}

.step-node.completed {
    background: #10b981;
    border-color: #34d399;
    color: white;
}

.step-label {
    font-size: 11px;
    font-weight: 500;
    margin-top: 6px;
    color: rgba(255, 255, 255, 0.6);
    text-align: center;
    position: absolute;
    width: 100px;
    transform: translateX(-34%);
}

.step-wrapper {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)


# ----------------- CACHED DATA GENERATORS -----------------
@st.cache_data(show_spinner="Generating 3D MRI Volume...")
def get_cached_mri():
    return generate_brain_mri_3d()

@st.cache_data(show_spinner="Generating 3D CT Volume...")
def get_cached_ct():
    return generate_lung_ct_3d()

@st.cache_data(show_spinner="Generating 3D PET-CT Volume...")
def get_cached_pet_ct():
    return generate_pet_ct_3d()


# ----------------- HELPER VISUALIZATION FUNCTIONS -----------------
def create_slice_with_overlay(slice_img, mask_img, modality="Brain MRI"):
    """
    Overlays multi-class segmentation mask on a scan slice with custom colors and transparency.
    """
    # Normalize scan slice to [0, 1] for RGB display
    s_min, s_max = slice_img.min(), slice_img.max()
    if s_max > s_min:
        slice_norm = (slice_img - s_min) / (s_max - s_min)
    else:
        slice_norm = np.zeros_like(slice_img)
        
    # Stack to create 3-channel RGB image
    rgb_img = np.stack([slice_norm]*3, axis=-1)
    
    # Class-specific colors
    if modality == "Brain MRI":
        colors = {
            1: [0.9, 0.1, 0.1],  # Red for Necrotic Core
            2: [0.9, 0.9, 0.1],  # Yellow for Enhancing Tumor
            3: [0.1, 0.7, 0.9]   # Cyan for Edema
        }
    elif modality == "Lung CT":
        colors = {
            1: [0.9, 0.4, 0.0]   # Bright Orange for Nodule
        }
    else: # PET-CT primary tumor
        colors = {
            1: [0.8, 0.0, 0.9]   # Fuchsia for Tumor
        }
        
    alpha = 0.45
    for label, color in colors.items():
        region = (mask_img == label)
        if np.any(region):
            for c in range(3):
                rgb_img[region, c] = (1.0 - alpha) * rgb_img[region, c] + alpha * color[c]
                
    return rgb_img

def create_pet_ct_fusion(ct_slice, pet_slice, alpha=0.5):
    """
    Blends anatomical CT slice (grayscale) with functional PET activity slice (hot map).
    """
    # 1. CT normal windowing (-1000 to +400 HU)
    ct_min, ct_max = -1000.0, 400.0
    ct_clipped = np.clip(ct_slice, ct_min, ct_max)
    ct_norm = (ct_clipped - ct_min) / (ct_max - ct_min)
    ct_rgb = np.stack([ct_norm]*3, axis=-1)
    
    # 2. PET normalization and custom colormap
    pet_max = 12.0 # SUV limit
    pet_norm = np.clip(pet_slice / pet_max, 0.0, 1.0)
    
    # Build pseudo-color map (Black -> Red -> Yellow -> White)
    pet_r = pet_norm * 1.3
    pet_g = np.clip((pet_norm - 0.25) * 1.5, 0.0, 1.0)
    pet_b = np.clip((pet_norm - 0.65) * 3.0, 0.0, 1.0)
    pet_rgb = np.stack([pet_r, pet_g, pet_b], axis=-1)
    pet_rgb = np.clip(pet_rgb, 0.0, 1.0)
    
    # 3. Alpha blending where PET has metabolic activity (SUV > 1.0)
    fusion = ct_rgb.copy()
    active_pet = pet_slice > 1.0
    
    for c in range(3):
        blend = alpha * (pet_slice[active_pet] / pet_max)
        blend = np.clip(blend, 0.1, 0.9)
        fusion[active_pet, c] = (1.0 - blend) * ct_rgb[active_pet, c] + blend * pet_rgb[active_pet, c]
        
    return np.clip(fusion, 0.0, 1.0)


# ----------------- SIDEBAR HEADER & NAVIGATION -----------------
with st.sidebar:
    # App Logo / Header Image
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.markdown("<h2 class='gradient-text'>MedInsight 3D</h2>", unsafe_allow_html=True)
        
    st.markdown("<div style='text-align: center; color: rgba(255,255,255,0.5); font-size: 13px; margin-top:-10px; margin-bottom: 20px;'>Clinical Imaging AI Dashboard</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation Selection
    page = st.radio(
        "Navigation",
        [
            "🏥 Clinical Dashboard Overview",
            "🔍 Interactive 3D Scan Explorer",
            "⚙️ Image Analysis Pipeline",
            "📊 Models & Performance Metrics",
            "💡 Clinical Insights & Analytics"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("<div style='font-size:11px; color:rgba(255,255,255,0.4); text-align:center;'>Built with Streamlit & Plotly<br>© 2026 MedInsight 3D</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: CLINICAL DASHBOARD OVERVIEW
# ==============================================================================
if page == "🏥 Clinical Dashboard Overview":
    st.markdown("""
    <div style="margin-bottom: 25px;">
        <h1 class="gradient-text" style="margin: 0; font-size: 38px; font-weight: 800; line-height: 1.1;">MedInsight 3D</h1>
        <p style="margin: 5px 0 0 0; font-size: 14px; color: rgba(255, 255, 255, 0.6); font-weight: 500; text-transform: uppercase; letter-spacing: 1.5px;">Enterprise Clinical AI Analytics Platform</p>
    </div>
    
    <div style="display: flex; gap: 15px; margin-bottom: 30px; flex-wrap: wrap;">
        <div class="glass-card" style="flex: 1; min-width: 170px; padding: 15px; border-left: 4px solid #3b82f6; margin-bottom: 0;">
            <h4 style="margin: 0; color: #3b82f6; font-size: 24px; font-weight: 700;">1,482</h4>
            <span style="font-size: 11px; color: rgba(255,255,255,0.6); font-weight: 500; text-transform: uppercase; display: block; margin-top: 5px; line-height:1.2;">Total Scans Processed</span>
        </div>
        <div class="glass-card" style="flex: 1; min-width: 170px; padding: 15px; border-left: 4px solid #10b981; margin-bottom: 0;">
            <h4 style="margin: 0; color: #10b981; font-size: 24px; font-weight: 700;">91.2%</h4>
            <span style="font-size: 11px; color: rgba(255,255,255,0.6); font-weight: 500; text-transform: uppercase; display: block; margin-top: 5px; line-height:1.2;">AI Pipeline Mean Dice</span>
        </div>
        <div class="glass-card" style="flex: 1; min-width: 170px; padding: 15px; border-left: 4px solid #ef4444; margin-bottom: 0;">
            <h4 style="margin: 0; color: #ef4444; font-size: 24px; font-weight: 700;">14</h4>
            <span style="font-size: 11px; color: rgba(255,255,255,0.6); font-weight: 500; text-transform: uppercase; display: block; margin-top: 5px; line-height:1.2;">High Risk Clinical Alerts</span>
        </div>
        <div class="glass-card" style="flex: 1; min-width: 170px; padding: 15px; border-left: 4px solid #f59e0b; margin-bottom: 0;">
            <h4 style="margin: 0; color: #f59e0b; font-size: 24px; font-weight: 700;">3</h4>
            <span style="font-size: 11px; color: rgba(255,255,255,0.6); font-weight: 500; text-transform: uppercase; display: block; margin-top: 5px; line-height:1.2;">Active Queued Scans</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
        
    # Main sections: Active Triage Queue & System Status
    col1, col2 = st.columns([3, 2], gap="large")
    
    with col1:
        st.markdown("#### 🚨 AI Clinical Triage & Scanning Queue")
        st.markdown("<p style='font-size: 13px; color: rgba(255,255,255,0.6); margin-top:-10px;'>Scans processed in the last 24 hours. High-priority cases are flagged based on tumor volumetric growth or high malignancy features.</p>", unsafe_allow_html=True)
        
        # Patients queue table
        triage_data = {
            "Patient ID": ["P-8402", "P-6910", "P-7214", "P-3912", "P-1048"],
            "Modality": ["Brain MRI (BraTS)", "Lung CT (LIDC)", "PET-CT (Hecktor)", "Brain MRI (BraTS)", "Lung CT (LIDC)"],
            "AI Findings": ["Glioblastoma (Enhancing Core)", "Solitary Nodule (12mm)", "Metastatic Squamous Cell", "Infarction / No Tumor", "Benign Nodule (3mm)"],
            "AI Confidence": ["98.4%", "89.2%", "94.6%", "99.1%", "95.8%"],
            "Triage Priority": ["🔴 CRITICAL", "🟡 MEDIUM", "🔴 CRITICAL", "🟢 LOW", "🟢 LOW"]
        }
        df_triage = pd.DataFrame(triage_data)
        st.dataframe(df_triage, hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div class="glass-card" style="margin-top: 15px; padding: 15px; background: rgba(239, 68, 68, 0.05); border-color: rgba(239, 68, 68, 0.2);">
            <h6 style="color: #ef4444; margin-top:0; display:flex; align-items:center;">
                <span style="font-size:18px; margin-right:8px;">⚠️</span> Action Required: Immediate Review
            </h6>
            <p style="font-size: 12px; color: rgba(255,255,255,0.75); margin-bottom:0; line-height: 1.5;">
                Patient <b>P-8402</b> (Brain MRI) exhibits a <b>18.4% volumetric expansion</b> in the Enhancing Tumor (ET) core relative to the previous scan (12 days ago). This indicates rapid progression. Immediate diagnostic confirmation is advised.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("#### ⚙️ Platform Engine & GPU Status")
        
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 12px;">
            <div style="display:flex; justify-content:space-between; margin-bottom: 8px;">
                <span style="font-size:13px; font-weight:500;">GPU Core (NVIDIA A100-SXM4)</span>
                <span style="font-size:13px; color:#10b981; font-weight:bold;">Active (78%)</span>
            </div>
            <div style="background:rgba(255,255,255,0.08); height:6px; border-radius:3px;">
                <div style="background:#10b981; width:78%; height:6px; border-radius:3px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:12px; font-size:11px; color:rgba(255,255,255,0.5);">
                <span>VRAM Allocated: 14.8 GB / 40.0 GB</span>
                <span>GPU Temp: 64°C</span>
            </div>
        </div>
        
        <div class="glass-card" style="margin-bottom: 12px;">
            <h5 style="margin-top:0; font-size:14px;">Active Pipeline Models</h5>
            <div style="font-size:12px; line-height:1.6;">
                <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.05); padding: 4px 0;">
                    <span>🧠 <b>3D Attention U-Net</b> (BraTS)</span>
                    <span style="color:#10b981; font-weight:500;">Loaded (v2.4.1)</span>
                </div>
                <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.05); padding: 4px 0;">
                    <span>🫁 <b>3D ResNet Nodule Classifier</b> (LIDC)</span>
                    <span style="color:#10b981; font-weight:500;">Loaded (v1.8.0)</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding: 4px 0;">
                    <span>⚡ <b>Dual-Path Fusion Segmenter</b> (Hecktor)</span>
                    <span style="color:#10b981; font-weight:500;">Loaded (v3.0.2)</span>
                </div>
            </div>
        </div>

        <div class="glass-card">
            <h5 style="margin-top:0; font-size:14px;">Clinical Integration Status</h5>
            <div style="font-size:12px; line-height:1.6;">
                <div style="display:flex; justify-content:space-between;">
                    <span>PACS Link (DICOM listener)</span>
                    <span style="color:#10b981;">Online</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span>HL7 Electronic Health Record Sync</span>
                    <span style="color:#10b981;">Online</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span>FDA-Approved Clinical Safety Check</span>
                    <span style="color:#3b82f6;">Pending Approval</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔍 Interactive 3D Scan Explorer":
    st.markdown("<h2 class='gradient-text'>Interactive 3D Clinical Scan Explorer</h2>", unsafe_allow_html=True)
    st.markdown("Inspect slices, select different modalities, toggle segmentation overlays, and explore multi-modal image fusion.")
    st.markdown("---")
    
    # Selection Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
    
    with col_ctrl1:
        modality = st.selectbox(
            "Select Clinical Dataset & Modality",
            ["Brain MRI (BraTS)", "Lung CT (LIDC-IDRI)", "PET-CT Fusion (Hecktor)"]
        )
        
    with col_ctrl2:
        plane = st.selectbox("Scan Viewing Plane", ["Axial (Z)", "Sagittal (X)", "Coronal (Y)"])
        
    with col_ctrl3:
        overlay_seg = st.checkbox("Overlay Deep Learning Segmentation Mask", value=True)
        
        # If PET-CT is selected, allow fusion parameters
        if modality == "PET-CT Fusion (Hecktor)":
            fusion_mode = st.checkbox("Enable PET-CT Fusion Mode", value=True)
            fusion_alpha = st.slider("PET Fusion Opacity (Alpha)", 0.1, 1.0, 0.5)
        else:
            fusion_mode = False
            fusion_alpha = 0.5
            
    # Load corresponding data
    grid_size = 64
    if modality == "Brain MRI (BraTS)":
        vols, mask = get_cached_mri()
        # Tabs for MRI sequences
        seq = st.tabs(["T1c (Enhancing-contrast)", "T1", "T2 (Bright Fluid)", "FLAIR (Fluid Attenuated)"])
        with seq[0]:
            active_vol = vols['T1c']
            seq_name = "T1c"
        with seq[1]:
            active_vol = vols['T1']
            seq_name = "T1"
        with seq[2]:
            active_vol = vols['T2']
            seq_name = "T2"
        with seq[3]:
            active_vol = vols['FLAIR']
            seq_name = "FLAIR"
            
    elif modality == "Lung CT (LIDC-IDRI)":
        active_vol, mask = get_cached_ct()
        seq_name = "CT"
    else: # PET-CT Fusion
        ct_vol, pet_vol, mask = get_cached_pet_ct()
        seq_name = "PET-CT"
        
    # Setup slice indexing based on plane selection
    if plane == "Axial (Z)":
        max_slices = grid_size
        slice_idx = st.slider("Axial Slice Index", 0, max_slices - 1, grid_size // 2)
    elif plane == "Sagittal (X)":
        max_slices = grid_size
        slice_idx = st.slider("Sagittal Slice Index", 0, max_slices - 1, grid_size // 2)
    else:
        max_slices = grid_size
        slice_idx = st.slider("Coronal Slice Index", 0, max_slices - 1, grid_size // 2)
        
    # Slice Extraction
    if plane == "Axial (Z)":
        if modality == "PET-CT Fusion (Hecktor)":
            ct_slice = ct_vol[slice_idx, :, :]
            pet_slice = pet_vol[slice_idx, :, :]
        else:
            slice_img = active_vol[slice_idx, :, :]
        mask_slice = mask[slice_idx, :, :]
        title_suffix = f"Slice Z = {slice_idx}"
    elif plane == "Sagittal (X)":
        if modality == "PET-CT Fusion (Hecktor)":
            ct_slice = ct_vol[:, :, slice_idx]
            pet_slice = pet_vol[:, :, slice_idx]
        else:
            slice_img = active_vol[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]
        title_suffix = f"Slice X = {slice_idx}"
    else: # Coronal
        if modality == "PET-CT Fusion (Hecktor)":
            ct_slice = ct_vol[:, slice_idx, :]
            pet_slice = pet_vol[:, slice_idx, :]
        else:
            slice_img = active_vol[:, slice_idx, :]
        mask_slice = mask[:, slice_idx, :]
        title_suffix = f"Slice Y = {slice_idx}"
        
    # Render Plotly Display
    col_viz, col_lbl = st.columns([2, 1], gap="medium")
    
    with col_viz:
        st.markdown(f"##### {modality} - {title_suffix}")
        
        # Build image to display
        if modality == "PET-CT Fusion (Hecktor)":
            if fusion_mode:
                display_img = create_pet_ct_fusion(ct_slice, pet_slice, fusion_alpha)
                # Overlay mask on top of fusion if selected
                if overlay_seg:
                    display_img = create_slice_with_overlay(display_img, mask_slice, modality)
                fig = px.imshow(display_img)
            else:
                # Show CT by default or allow checkbox toggle for single PET
                show_pet = st.toggle("Show Functional PET Scan instead of CT Scan", value=False)
                if show_pet:
                    fig = px.imshow(pet_slice, color_continuous_scale="hot", labels={"color": "SUV"})
                else:
                    fig = px.imshow(ct_slice, color_continuous_scale="gray", labels={"color": "HU"})
                if overlay_seg:
                    # Single modality overlay
                    raw_slice = pet_slice if show_pet else ct_slice
                    fig = px.imshow(create_slice_with_overlay(raw_slice, mask_slice, modality))
        else:
            if overlay_seg:
                display_img = create_slice_with_overlay(slice_img, mask_slice, modality)
                fig = px.imshow(display_img)
            else:
                fig = px.imshow(slice_img, color_continuous_scale="gray")
                
        # Aesthetic configurations for clinical scans
        fig.update_layout(
            width=550,
            height=500,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=False)
        
    with col_lbl:
        # Contextual label & legend
        st.markdown("##### Clinical Annotations")
        
        if modality == "Brain MRI (BraTS)":
            st.markdown("""
            <div class="glass-card">
                <h6>MRI Modality Contrast Physics</h6>
                <ul style="font-size:13px; color:rgba(255,255,255,0.75);">
                    <li><b>T1-contrast (T1c)</b>: Highlights blood-brain barrier breakdown (active tumor borders fluoresce).</li>
                    <li><b>T2 / FLAIR</b>: FLAIR attenuates free CSF signal, making vasogenic tumor edema stand out as bright white.</li>
                </ul>
                <hr style="margin: 10px 0; border-color: rgba(255,255,255,0.05);">
                <h6>Segmentation Color Guide:</h6>
                <div style="font-size:12px; margin-top:8px;">
                    <div style="display:flex; align-items:center; margin-bottom:5px;">
                        <div style="width:14px; height:14px; background:rgba(230,25,25,0.7); margin-right:8px; border-radius:3px;"></div>
                        <span><b>Necrotic Core (NCR)</b> - Dead cellular debris</span>
                    </div>
                    <div style="display:flex; align-items:center; margin-bottom:5px;">
                        <div style="width:14px; height:14px; background:rgba(230,230,25,0.7); margin-right:8px; border-radius:3px;"></div>
                        <span><b>Enhancing Tumor (ET)</b> - Active mitotic cell division</span>
                    </div>
                    <div style="display:flex; align-items:center;">
                        <div style="width:14px; height:14px; background:rgba(25,180,230,0.7); margin-right:8px; border-radius:3px;"></div>
                        <span><b>Edema (ED)</b> - Surrounding swelling/fluid accumulation</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        elif modality == "Lung CT (LIDC-IDRI)":
            st.markdown("""
            <div class="glass-card">
                <h6>Hounsfield Unit (HU) Calibration</h6>
                <ul style="font-size:13px; color:rgba(255,255,255,0.75);">
                    <li><b>Air</b>: -1000 HU</li>
                    <li><b>Lungs</b>: -850 to -700 HU</li>
                    <li><b>Fat / Muscle</b>: -50 to +50 HU</li>
                    <li><b>Spinal Bone</b>: +300 to +600 HU</li>
                </ul>
                <hr style="margin: 10px 0; border-color: rgba(255,255,255,0.05);">
                <h6>Target Nodule Annotation:</h6>
                <div style="font-size:12px; margin-top:8px;">
                    <div style="display:flex; align-items:center; margin-bottom:5px;">
                        <div style="width:14px; height:14px; background:rgba(230,102,0,0.7); margin-right:8px; border-radius:3px;"></div>
                        <span><b>Solitary Pulmonary Nodule (SPN)</b></span>
                    </div>
                    <p style="font-size:11px; color:rgba(255,255,255,0.5); margin-top:5px; line-height:1.4;">
                        Target nodule is located in the left lower lobe parenchyma. Border shows mild spiculation, suggesting elevated malignancy risk.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        else: # PET-CT Fusion
            st.markdown("""
            <div class="glass-card">
                <h6>Multi-modal Fusion Physics</h6>
                <ul style="font-size:13px; color:rgba(255,255,255,0.75);">
                    <li><b>CT (Structure)</b>: High-resolution anatomical structures (muscles, trachea, spine).</li>
                    <li><b>PET (Function)</b>: standard uptake value (SUV) reveals cellular glycolysis. Malignant tumors consume glucose rapidly, lighting up bright white/yellow.</li>
                </ul>
                <hr style="margin: 10px 0; border-color: rgba(255,255,255,0.05);">
                <h6>Target Lymph Node Annotation:</h6>
                <div style="font-size:12px; margin-top:8px;">
                    <div style="display:flex; align-items:center; margin-bottom:5px;">
                        <div style="width:14px; height:14px; background:rgba(204,0,230,0.7); margin-right:8px; border-radius:3px;"></div>
                        <span><b>Primary Squamous Cell Carcinoma</b></span>
                    </div>
                    <p style="font-size:11px; color:rgba(255,255,255,0.5); margin-top:5px; line-height:1.4;">
                        Metabolically hyper-active primary lesion identified in the left lateral neck region. High metabolic activity (SUV Max &gt; 10) confirms metastatic progression.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: IMAGE ANALYSIS PIPELINE
# ==============================================================================
elif page == "⚙️ Image Analysis Pipeline":
    st.markdown("<h2 class='gradient-text'>Medical Image Analysis Pipeline Simulator</h2>", unsafe_allow_html=True)
    st.markdown("Trace how raw clinical DICOM images are converted, preprocessed, aligned, segmented, and reduced to quantitative radiomic descriptors.")
    st.markdown("---")
    
    # Pipeline Modality Selector
    pipe_mod = st.selectbox("Pipeline Target Modality", ["Brain MRI", "Lung CT", "PET-CT Fusion"])
    
    # Steps Indicator (Visual HTML Steps)
    step_selected = st.slider("Execute Pipeline Step", 1, 5, 4, help="Drag the slider to step through the image processing stages.")
    
    steps = [
        {"num": 1, "name": "Raw DICOM", "desc": "Convert & format"},
        {"num": 2, "name": "Preprocessing", "desc": "Denoise & Normalize"},
        {"num": 3, "name": "Registration", "desc": "Spatial Alignment"},
        {"num": 4, "name": "Segmentation", "desc": "Deep Learning Mask"},
        {"num": 5, "name": "Radiomics", "desc": "Quantitative Metrics"}
    ]
    
    step_html = "<div class='steps-container'>"
    for step in steps:
        active_class = ""
        if step["num"] < step_selected:
            active_class = "completed"
        elif step["num"] == step_selected:
            active_class = "active"
            
        step_html += f"""
        <div class="step-wrapper">
            <div class="step-node {active_class}">{step["num"]}</div>
            <div class="step-label" style="font-weight:{'bold' if step['num'] == step_selected else 'normal'}; color:{'#00D2FF' if step['num'] == step_selected else 'rgba(255,255,255,0.5)'};">
                {step["name"]}<br><span style="font-size:9px; font-weight:normal; opacity:0.7;">{step["desc"]}</span>
            </div>
        </div>
        """
    step_html += "</div><br>"
    st.markdown(step_html, unsafe_allow_html=True)
    
    # Simulate data based on selection
    sim = simulate_pipeline_stages(pipe_mod)
    
    # Main Visualization Columns
    col_v1, col_v2 = st.columns([3, 2], gap="large")
    
    with col_v1:
        st.markdown(f"#### Step {step_selected}: {steps[step_selected-1]['name']} Visualizer")
        
        # Axial Slice representation of step
        z_idx = 32
        
        if pipe_mod == "Brain MRI":
            if step_selected == 1:
                # Raw
                fig = px.imshow(sim['Raw'][z_idx, :, :], color_continuous_scale="gray")
                st.info("⚠️ Raw MRI scan contains intensity bias field (brighter left-to-right gradient) and acquisition noise.")
            elif step_selected == 2:
                # Preprocessed
                fig = px.imshow(sim['Preprocessed'][z_idx, :, :], color_continuous_scale="gray")
                st.success("✔️ Preprocessing complete: Applied N4 bias field correction, Z-score intensity normalization, and Gaussian noise reduction.")
            elif step_selected == 3:
                # Registered
                fig = px.imshow(sim['Registered'][z_idx, :, :], color_continuous_scale="gray")
                st.success("✔️ Modality Registration: Coregistered and resampled T1c, T2, and FLAIR to isotropic 1x1x1 mm resolution in MNI152 space.")
            elif step_selected == 4:
                # Segmented
                overlay = create_slice_with_overlay(sim['Preprocessed'][z_idx, :, :], sim['Mask'][z_idx, :, :], "Brain MRI")
                fig = px.imshow(overlay)
                st.success("✔️ Deep Learning Inference: Multi-class 3D Attention U-Net segmented three distinct tumor compartments.")
            else:
                # Radiomics features page
                overlay = create_slice_with_overlay(sim['Preprocessed'][z_idx, :, :], sim['Mask'][z_idx, :, :], "Brain MRI")
                fig = px.imshow(overlay)
                st.success("✔️ Quantitative Extraction: Extracting Shape, Gray Level Co-occurrence (GLCM), and Run Length (GLRLM) features from the tumor mask.")
                
        elif pipe_mod == "Lung CT":
            if step_selected == 1:
                fig = px.imshow(sim['Raw'][z_idx, :, :], color_continuous_scale="gray")
                st.info("⚠️ Raw CT volume contains breathing motion artifacts and high background noise.")
            elif step_selected == 2:
                fig = px.imshow(sim['Preprocessed'][z_idx, :, :], color_continuous_scale="gray")
                st.success("✔️ Preprocessing complete: Clipped Hounsfield Units (-1000 to +400 HU) to isolate lung structures, followed by bilateral filtering.")
            elif step_selected == 3:
                fig = px.imshow(sim['Preprocessed'][z_idx, :, :], color_continuous_scale="gray")
                st.success("✔️ Spatial Alignment: Registered scan to lung atlas to standardize coordinate spaces.")
            elif step_selected == 4:
                overlay = create_slice_with_overlay(sim['Preprocessed'][z_idx, :, :], sim['Mask'][z_idx, :, :], "Lung CT")
                fig = px.imshow(overlay)
                st.success("✔️ Deep Learning Inference: 3D ResNet/U-Net segmented a localized solitary pulmonary nodule in the left lobe.")
            else:
                overlay = create_slice_with_overlay(sim['Preprocessed'][z_idx, :, :], sim['Mask'][z_idx, :, :], "Lung CT")
                fig = px.imshow(overlay)
                st.success("✔️ Quantitative Extraction: Computing clinical nodule characteristics (mean Hounsfield Unit density, maximum diameter, sphericity).")
                
        else: # PET-CT
            if step_selected == 1:
                # Show misaligned side-by-side or fused
                overlay_misaligned = create_pet_ct_fusion(sim['Raw_CT'][z_idx, :, :], sim['Raw_PET'][z_idx, :, :], 0.5)
                fig = px.imshow(overlay_misaligned)
                st.info("⚠️ Spatial Misalignment: Patient moved between CT scan and PET scan. Note the 3-pixel spatial shift between anatomy and metabolism.")
            elif step_selected == 2:
                # Preprocessed
                fig = px.imshow(sim['Preprocessed_CT'][z_idx, :, :], color_continuous_scale="gray")
                st.success("✔️ Preprocessing complete: Normalized CT intensity and standardized PET SUV scaling parameters.")
            elif step_selected == 3:
                # Registered
                overlay_aligned = create_pet_ct_fusion(sim['CT'][z_idx, :, :], sim['PET'][z_idx, :, :], 0.5)
                fig = px.imshow(overlay_aligned)
                st.success("✔️ Deformable Co-registration: Aligned PET and CT scans using mutual information deformable registration (ANTs). Spatial alignment matches perfectly.")
            elif step_selected == 4:
                overlay_fused = create_pet_ct_fusion(sim['CT'][z_idx, :, :], sim['PET'][z_idx, :, :], 0.5)
                overlay_seg = create_slice_with_overlay(overlay_fused, sim['Mask'][z_idx, :, :], "PET-CT Fusion")
                fig = px.imshow(overlay_seg)
                st.success("✔️ Deep Learning Inference: Dual-modal 3D CNN segmented primary neck carcinoma based on fused features.")
            else:
                overlay_fused = create_pet_ct_fusion(sim['CT'][z_idx, :, :], sim['PET'][z_idx, :, :], 0.5)
                overlay_seg = create_slice_with_overlay(overlay_fused, sim['Mask'][z_idx, :, :], "PET-CT Fusion")
                fig = px.imshow(overlay_seg)
                st.success("✔️ Quantitative Extraction: Extracting metabolic metrics including Total Lesion Glycolysis (TLG) and Metabolic Tumor Volume (MTV).")
                
        # Clean figure styling
        fig.update_layout(
            width=500,
            height=450,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=False)
        
    with col_v2:
        st.markdown("#### Pipeline State Metadata")
        
        # Display features if step 5 is active
        if step_selected == 5:
            st.markdown("""
            <div class="glass-card" style="border-color:#10b981; background:rgba(16,185,129,0.02);">
                <h5 style="color:#10b981; margin-top:0;">📊 Extracted Radiomic Features</h5>
                <p style="font-size:12px; color:rgba(255,255,255,0.6);">These numeric descriptors are utilized to predict survival, therapy response, or recurrence risks.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Format feature dict as a table
            feats = sim['Features']
            df_feat = pd.DataFrame(list(feats.items()), columns=["Feature Name", "Value"])
            st.dataframe(df_feat, hide_index=True, use_container_width=True)
        else:
            # Show processing metadata card
            if step_selected == 1:
                st.markdown("""
                <div class="glass-card">
                    <h5>📥 Acquisition & Format Conversion</h5>
                    <p style="font-size:12px; line-height:1.5;">
                        <b>Action</b>: Read patient folders containing raw DICOM slices. Convert DICOM headers and image data to isotropic NIfTI (Neuroimaging Informatics Technology Initiative) format.
                    </p>
                    <p style="font-size:12px; line-height:1.5;">
                        <b>Algorithms</b>: <code>dcm2niix</code> library conversion. Checking for slice inconsistencies, missing slices, or header corruption.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif step_selected == 2:
                st.markdown("""
                <div class="glass-card">
                    <h5>🧹 Preprocessing & Denoising</h5>
                    <p style="font-size:12px; line-height:1.5;">
                        <b>Action</b>: Standardize intensity values, filter scanning artifacts, and adjust bias fields.
                    </p>
                    <p style="font-size:12px; line-height:1.5;">
                        <b>Algorithms</b>:
                        <ul>
                            <li><b>N4 Bias Field Correction</b>: Iteratively estimates bias field and divides the image to correct RF coil variations.</li>
                            <li><b>Intensity Normalization</b>: Z-score scaling (subtracting mean, dividing by standard deviation) of brain voxels.</li>
                        </ul>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif step_selected == 3:
                st.markdown("""
                <div class="glass-card">
                    <h5>📐 Registration & Spatial Alignment</h5>
                    <p style="font-size:12px; line-height:1.5;">
                        <b>Action</b>: Align different MRI sequences (T1, T2, FLAIR) or co-register functional PET with structural CT.
                    </p>
                    <p style="font-size:12px; line-height:1.5;">
                        <b>Algorithms</b>:
                        <ul>
                            <li><b>Affine Registration</b>: 12-degrees-of-freedom transformation matching global orientation, rotation, and scale.</li>
                            <li><b>Deformable B-spline Registration</b>: Corrects patient posture variations or respiratory tissue shifting.</li>
                        </ul>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif step_selected == 4:
                st.markdown("""
                <div class="glass-card">
                    <h5>🧠 Deep Learning Segmentation</h5>
                    <p style="font-size:12px; line-height:1.5;">
                        <b>Action</b>: Execute 3D Deep Neural Networks to predict coordinate-wise labels for tissues or pathologies.
                    </p>
                    <p style="font-size:12px; line-height:1.5;">
                        <b>Algorithms</b>:
                        <ul>
                            <li><b>3D Attention U-Net</b>: Incorporates attention gates to filter out non-anatomical background regions, focusing weights on the target pathology.</li>
                            <li><b>Skip Connections</b>: Propagates fine spatial details from the encoder directly to the decoder.</li>
                        </ul>
                    </p>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: MODELS & PERFORMANCE METRICS
# ==============================================================================
elif page == "📊 Models & Performance Metrics":
    st.markdown("<h2 class='gradient-text'>Model Architectures & Validation Performance</h2>", unsafe_allow_html=True)
    st.markdown("Review the network architectures, performance curves, and validation metrics used to qualify models for clinical trials.")
    st.markdown("---")
    
    # Model Architecture Cards
    st.markdown("#### Applied Computer Vision Networks")
    
    tab_arch = st.tabs(["3D Attention U-Net (Segmentation)", "3D ResNet (Classification)", "Dual-Encoder PET-CT Network"])
    
    with tab_arch[0]:
        st.markdown("""
        <div class="glass-card">
            <h5>3D Attention U-Net Architecture</h5>
            <p style="font-size:13px; color:rgba(255,255,255,0.75);">
                Specifically optimized for multi-class Brain Tumor (BraTS) segmentation. The network uses 3D convolutions (3x3x3 kernels) to capture spatial features along the coronal, sagittal, and axial dimensions simultaneously.
            </p>
            <div style="font-size:12px; color:rgba(255,255,255,0.6);">
                <ul>
                    <li><b>Encoder</b>: 4 downsampling blocks (Double 3D Convolution + GroupNorm + GeLU + MaxPool3D).</li>
                    <li><b>Attention Gates</b>: Placed on the skip connections. They use gating coefficients from the coarser decoder level to prune low-level activation maps, removing spatial noise.</li>
                    <li><b>Decoder</b>: 4 upsampling blocks (3D Transposed Convolutions + Concatenation + 3D Attention + Double Convolutions).</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with tab_arch[1]:
        st.markdown("""
        <div class="glass-card">
            <h5>3D ResNet-34 Nodule Classifier</h5>
            <p style="font-size:13px; color:rgba(255,255,255,0.75);">
                Designed for classifying the malignancy risk of solitary pulmonary nodules extracted from Chest CT volumes.
            </p>
            <div style="font-size:12px; color:rgba(255,255,255,0.6);">
                <ul>
                    <li><b>Input</b>: Resampled 3D bounding boxes around the nodule (32x32x32 voxels).</li>
                    <li><b>Structure</b>: Standard ResNet-34 expanded to 3D. Skip connections prevent gradient vanishing during backpropagation through spatial dimensions.</li>
                    <li><b>Classifier</b>: Global Average Pooling 3D followed by a Fully Connected Layer with Sigmoid activation outputting malignancy probability (0 to 1).</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with tab_arch[2]:
        st.markdown("""
        <div class="glass-card">
            <h5>Dual-Encoder Co-registered Fusion Network</h5>
            <p style="font-size:13px; color:rgba(255,255,255,0.75);">
                Specifically built for co-registered PET-CT datasets (e.g. Hecktor head & neck cancer).
            </p>
            <div style="font-size:12px; color:rgba(255,255,255,0.6);">
                <ul>
                    <li><b>Dual-Path Encoder</b>: Two parallel encoders process CT (anatomy) and PET (glycolysis activity) separately.</li>
                    <li><b>Cross-Attention Fusion</b>: CT feature maps gate PET feature maps, and vice-versa, combining anatomical contours with metabolic SUV hotspots.</li>
                    <li><b>Output</b>: Decodes fused feature representation into a single high-accuracy tumor mask.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Interactive Metrics
    st.markdown("#### Model Validation & Evaluation Plots")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        # ROC-AUC Curve using Plotly
        st.markdown("##### Receiver Operating Characteristic (ROC) Curve")
        
        # Generate synthetic ROC data
        fpr = np.linspace(0, 1, 100)
        tpr_m1 = 1 - (1 - fpr)**2.8 # AUC ~ 0.74 (baseline)
        tpr_m2 = 1 - (1 - fpr)**5.5 # AUC ~ 0.85 (U-Net)
        tpr_m3 = 1 - (1 - fpr)**10.0 # AUC ~ 0.92 (3D Attention U-Net)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr_m3, mode='lines', name='3D Attention U-Net (AUC = 0.92)', line=dict(color='#00D2FF', width=3)))
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr_m2, mode='lines', name='Standard 3D U-Net (AUC = 0.85)', line=dict(color='#A78BFA', width=2, dash='dash')))
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr_m1, mode='lines', name='2D ResNet baseline (AUC = 0.74)', line=dict(color='rgba(255,255,255,0.4)', width=2, dash='dot')))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Guess (AUC = 0.50)', line=dict(color='red', width=1, dash='dot')))
        
        fig_roc.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.5, y=0.1, bgcolor='rgba(15,23,42,0.8)', bordercolor='rgba(255,255,255,0.05)'),
            xaxis=dict(title='False Positive Rate (1 - Specificity)', gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title='True Positive Rate (Sensitivity)', gridcolor='rgba(255,255,255,0.05)'),
            margin=dict(l=40, r=40, t=10, b=40)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with col_m2:
        # Confusion Matrix using Plotly
        st.markdown("##### Segmentation Pixel-wise Confusion Matrix")
        
        z_cm = [
            [0.982, 0.015, 0.003], # Background
            [0.021, 0.941, 0.038], # Tumor Core
            [0.008, 0.042, 0.950]  # Edema
        ]
        classes = ["Background", "Tumor Core", "Edema"]
        
        fig_cm = px.imshow(
            z_cm,
            x=classes,
            y=classes,
            text_auto=".3f",
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Label", y="True Label", color="Proportion")
        )
        
        fig_cm.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=40, t=10, b=40),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    # Table of Key Metrics
    st.markdown("##### Quantitative Metric Overview")
    metric_data = {
        "Modality / Mod": ["Brain MRI", "Brain MRI", "Brain MRI", "Lung CT", "PET-CT Fusion"],
        "Target Pathology": ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)", "Pulmonary Nodule", "Neck Cancer (GTV)"],
        "Dice Coefficient": [0.954, 0.892, 0.865, 0.884, 0.842],
        "Mean Hausdorff Dist (mm)": [2.4, 3.8, 4.1, 1.8, 4.5],
        "Sensitivity": [0.961, 0.904, 0.871, 0.912, 0.868],
        "Specificity": [0.994, 0.991, 0.995, 0.985, 0.990]
    }
    st.dataframe(pd.DataFrame(metric_data), hide_index=True, use_container_width=True)


# ==============================================================================
# PAGE 5: CLINICAL INSIGHTS & ANALYTICS
# ==============================================================================
else: # Clinical Insights
    st.markdown("<h2 class='gradient-text'>Clinical Insights & Longitudinal Analytics</h2>", unsafe_allow_html=True)
    st.markdown("Examine the real-world utility of medical imaging AI: tracking treatment response over time and stratifying patient risk via radiomic biomarkers.")
    st.markdown("---")
    
    # Longitudinal Study simulation
    st.markdown("#### 1. Longitudinal Tumor Volume Tracking")
    st.markdown("Tracking tumor volumetric shrinkage over 6 cycles of systemic therapy. This provides radiologists with automated, objective response indicators.")
    
    # Slider to simulate cycles
    cycles = st.slider("Simulate Patient Treatment Cycles", 1, 6, 6)
    
    # Generate data
    cycles_arr = np.array(range(1, cycles + 1))
    # Responders (volume goes down rapidly)
    vol_resp = 45.0 * np.exp(-0.45 * (cycles_arr - 1)) + np.random.normal(0, 1.0, len(cycles_arr))
    # Partial Responders (volume goes down then plateaus)
    vol_part = 45.0 * np.exp(-0.15 * (cycles_arr - 1)) + np.random.normal(0, 1.2, len(cycles_arr))
    # Non-Responders (volume increases or stays high)
    vol_non = 45.0 + 3.2 * (cycles_arr - 1) + np.random.normal(0, 1.5, len(cycles_arr))
    
    # Create Line plot
    fig_long = go.Figure()
    fig_long.add_trace(go.Scatter(x=cycles_arr, y=vol_resp, mode='lines+markers', name='Patient A: Responder', line=dict(color='#10B981', width=3)))
    fig_long.add_trace(go.Scatter(x=cycles_arr, y=vol_part, mode='lines+markers', name='Patient B: Partial Responder', line=dict(color='#F59E0B', width=2, dash='dash')))
    fig_long.add_trace(go.Scatter(x=cycles_arr, y=vol_non, mode='lines+markers', name='Patient C: Non-Responder', line=dict(color='#EF4444', width=3)))
    
    fig_long.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.7, y=0.9, bgcolor='rgba(15,23,42,0.8)', bordercolor='rgba(255,255,255,0.05)'),
        xaxis=dict(title='Chemotherapy / Immunotherapy Cycle', gridcolor='rgba(255,255,255,0.05)', tickmode='linear', tick0=1, dtick=1),
        yaxis=dict(title='Tumor Volume (cc)', gridcolor='rgba(255,255,255,0.05)', range=[0, 70]),
        margin=dict(l=40, r=40, t=10, b=40)
    )
    st.plotly_chart(fig_long, use_container_width=True)
    
    # Key Insight Section
    col_ins1, col_ins2 = st.columns(2, gap="large")
    
    with col_ins1:
        st.markdown("""
        <div class="glass-card" style="border-color:#3b82f6;">
            <h5 style="color:#60A5FA; margin-top:0;">💡 Key Insight: Radiomic Texture Correlation</h5>
            <p style="font-size:13px; color:rgba(255,255,255,0.8); line-height:1.6;">
                By utilizing our image analysis pipeline, we discovered that <b>Gray Level Co-occurrence Matrix (GLCM) Entropy</b> extracted from baseline (Cycle 0) PET scans is a strong predictor of patient response.
            </p>
            <p style="font-size:12px; color:rgba(255,255,255,0.65);">
                Patients exhibiting highly heterogeneous PET metabolic uptake (entropy &gt; 4.8) respond poorly to traditional chemotherapy compared to those with homogeneous uptake (hazard ratio = 2.45, <i>p</i> &lt; 0.01).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_ins2:
        # Kaplan-Meier survival curves
        st.markdown("##### 2. Kaplan-Meier Survival Analysis")
        st.markdown("<p style='font-size:11px; margin-top:-10px; color:rgba(255,255,255,0.5);'>Overall survival probability of patient cohort stratified by radiomic entropy signature.</p>", unsafe_allow_html=True)
        
        # Generate synthetic survival curve
        time_days = np.linspace(0, 1000, 100)
        prob_low = np.exp(-0.0004 * time_days) # High survival (low entropy)
        prob_high = np.exp(-0.0011 * time_days) # Low survival (high entropy)
        
        fig_km = go.Figure()
        fig_km.add_trace(go.Scatter(x=time_days, y=prob_low, mode='lines', name='Low-Risk Radiomic Signature', line=dict(color='#10B981', width=3)))
        fig_km.add_trace(go.Scatter(x=time_days, y=prob_high, mode='lines', name='High-Risk Radiomic Signature', line=dict(color='#EF4444', width=3)))
        
        fig_km.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.5, y=0.9, bgcolor='rgba(15,23,42,0.8)', bordercolor='rgba(255,255,255,0.05)'),
            xaxis=dict(title='Overall Survival Time (Days)', gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title='Survival Probability', gridcolor='rgba(255,255,255,0.05)', range=[0, 1.05]),
            margin=dict(l=40, r=40, t=10, b=40)
        )
        st.plotly_chart(fig_km, use_container_width=True)
