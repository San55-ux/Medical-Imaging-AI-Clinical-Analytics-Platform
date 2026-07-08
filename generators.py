import numpy as np
from scipy.ndimage import gaussian_filter

def create_ellipsoid_mask(grid_shape, center, radii):
    """
    Creates a 3D ellipsoid mask.
    radii: tuple of (rx, ry, rz)
    """
    nz, ny, nx = grid_shape
    z, y, x = np.ogrid[:nz, :ny, :nx]
    
    cz, cy, cx = center
    rz, ry, rx = radii
    
    mask = (((x - cx) / rx) ** 2 + 
            ((y - cy) / ry) ** 2 + 
            ((z - cz) / rz) ** 2) <= 1.0
    return mask

def generate_brain_mri_3d(grid_size=64):
    """
    Generates a synthetic 3D brain MRI volume with:
    - Anatomical structures: Skull, Brain parenchyma, Ventricles (CSF)
    - Pathologies: Brain tumor with Necrotic Core, Active Enhancing Tumor, and Edema.
    Returns:
        dict: {'T1': vol, 'T1c': vol, 'T2': vol, 'FLAIR': vol}
        ndarray: segmentation mask (0: background, 1: necrotic core, 2: enhancing tumor, 3: edema)
    """
    shape = (grid_size, grid_size, grid_size)
    center = (grid_size // 2, grid_size // 2, grid_size // 2)
    
    # 1. Base anatomical structures
    # Skull
    skull_mask = create_ellipsoid_mask(shape, center, (grid_size * 0.44, grid_size * 0.42, grid_size * 0.38))
    # Brain tissue (parenchyma)
    brain_mask = create_ellipsoid_mask(shape, center, (grid_size * 0.42, grid_size * 0.40, grid_size * 0.36))
    
    # Ventricles (CSF-filled cavities, offset from center)
    vent1 = create_ellipsoid_mask(shape, (grid_size * 0.50, grid_size * 0.42, grid_size * 0.45), (grid_size * 0.12, grid_size * 0.05, grid_size * 0.08))
    vent2 = create_ellipsoid_mask(shape, (grid_size * 0.50, grid_size * 0.58, grid_size * 0.45), (grid_size * 0.12, grid_size * 0.05, grid_size * 0.08))
    ventricles = vent1 | vent2
    
    # 2. Pathologies: Tumor compartments (offset from center in the right hemisphere)
    # Tumor center
    tc = (grid_size * 0.55, grid_size * 0.35, grid_size * 0.52)
    
    # Edema (outermost, infiltrative tumor region)
    edema_mask = create_ellipsoid_mask(shape, tc, (grid_size * 0.18, grid_size * 0.14, grid_size * 0.16)) & brain_mask
    
    # Active tumor core (enhancing tumor)
    active_mask = create_ellipsoid_mask(shape, tc, (grid_size * 0.10, grid_size * 0.08, grid_size * 0.09)) & brain_mask
    
    # Necrotic core (innermost dead tissue)
    necro_mask = create_ellipsoid_mask(shape, tc, (grid_size * 0.06, grid_size * 0.05, grid_size * 0.05)) & brain_mask
    
    # Clean up overlaps to make distinct labels:
    # 1: necrotic core, 2: enhancing tumor, 3: edema
    seg_mask = np.zeros(shape, dtype=np.uint8)
    seg_mask[edema_mask] = 3
    seg_mask[active_mask] = 2
    seg_mask[necro_mask] = 1
    
    # Initialize volumes
    t1 = np.zeros(shape)
    t1c = np.zeros(shape)
    t2 = np.zeros(shape)
    flair = np.zeros(shape)
    
    # Assign tissue properties (relative contrasts matching MRI physics)
    # Background
    t1 += 10; t1c += 10; t2 += 5; flair += 5
    
    # Skull bone (bright on CT, dark/intermediate on MRI)
    t1[skull_mask] = 60; t1c[skull_mask] = 60; t2[skull_mask] = 20; flair[skull_mask] = 15
    
    # Brain tissue (parenchyma)
    t1[brain_mask] = 120; t1c[brain_mask] = 120; t2[brain_mask] = 85; flair[brain_mask] = 80
    
    # Ventricles (CSF - dark on T1/FLAIR, bright on T2)
    t1[ventricles] = 30; t1c[ventricles] = 30; t2[ventricles] = 240; flair[ventricles] = 20
    
    # Edema (bright on T2/FLAIR, intermediate/dark on T1)
    t1[seg_mask == 3] = 95; t1c[seg_mask == 3] = 90; t2[seg_mask == 3] = 180; flair[seg_mask == 3] = 210
    
    # Enhancing tumor (bright on T1c, intermediate on T1/T2/FLAIR)
    t1[seg_mask == 2] = 100; t1c[seg_mask == 2] = 245; t2[seg_mask == 2] = 130; flair[seg_mask == 2] = 140
    
    # Necrotic core (dark on T1/T1c, bright on T2, intermediate on FLAIR)
    t1[seg_mask == 1] = 45; t1c[seg_mask == 1] = 45; t2[seg_mask == 1] = 200; flair[seg_mask == 1] = 120
    
    # Apply Gaussian smoothing to make the scan boundaries natural, plus texture noise
    volumes = {'T1': t1, 'T1c': t1c, 'T2': t2, 'FLAIR': flair}
    for key in volumes:
        # Smooth boundaries
        vol_smooth = gaussian_filter(volumes[key], sigma=0.8)
        # Add tissue texture noise
        noise = np.random.normal(0, 3.5, shape)
        # Combine
        volumes[key] = np.clip(vol_smooth + noise, 0, 255)
        
    return volumes, seg_mask

def generate_lung_ct_3d(grid_size=64):
    """
    Generates a synthetic 3D chest CT volume with:
    - Anatomy: Rib cage (bone HU), lung lobes (air HU), mediastinum/body (soft tissue HU)
    - Pathology: Solitary pulmonary nodule (soft tissue HU)
    Returns:
        ndarray: HU volume (-1000 to +400)
        ndarray: segmentation mask (0: background, 1: nodule)
    """
    shape = (grid_size, grid_size, grid_size)
    center = (grid_size // 2, grid_size // 2, grid_size // 2)
    
    # Start with air (-1000 HU)
    vol = np.full(shape, -1000.0)
    
    # Body contour (soft tissue around -10 HU to +50 HU)
    body_mask = create_ellipsoid_mask(shape, center, (grid_size * 0.46, grid_size * 0.44, grid_size * 0.38))
    vol[body_mask] = 30.0 # muscle/fat tissue HU
    
    # Lungs (very dark, -850 HU)
    lung_l = create_ellipsoid_mask(shape, (grid_size * 0.50, grid_size * 0.32, grid_size * 0.48), (grid_size * 0.32, grid_size * 0.14, grid_size * 0.28))
    lung_r = create_ellipsoid_mask(shape, (grid_size * 0.50, grid_size * 0.68, grid_size * 0.48), (grid_size * 0.32, grid_size * 0.14, grid_size * 0.28))
    lungs = lung_l | lung_r
    vol[lungs] = -850.0
    
    # Spinal column (bone HU, e.g. +300 HU)
    spine_mask = create_ellipsoid_mask(shape, (grid_size * 0.50, grid_size * 0.50, grid_size * 0.16), (grid_size * 0.08, grid_size * 0.08, grid_size * 0.20))
    vol[spine_mask] = 400.0
    
    # Rib cage (individual ribs in 3D)
    # We can model this simply by adding high intensity bands on the perimeter of the body
    z, y, x = np.ogrid[:grid_size, :grid_size, :grid_size]
    cz, cy, cx = center
    # Distance from center in axial plane
    r_axial = np.sqrt(((x - cx) / (grid_size*0.44))**2 + ((y - cy) / (grid_size*0.42))**2)
    ribs_mask = (r_axial > 0.95) & (r_axial < 1.03) & body_mask
    # Make ribs periodic along Z-axis (slices)
    rib_z = (z % 8) < 4
    ribs_mask = ribs_mask & rib_z
    vol[ribs_mask] = 350.0
    
    # Pathology: Lung Nodule (placed inside the left lung, soft tissue density, e.g. +20 HU)
    nodule_center = (grid_size * 0.45, grid_size * 0.30, grid_size * 0.48)
    nodule_mask = create_ellipsoid_mask(shape, nodule_center, (3.5, 3.2, 3.5))
    vol[nodule_mask] = 15.0
    
    # Smooth a bit and add noise
    vol_smooth = gaussian_filter(vol, sigma=0.6)
    noise = np.random.normal(0, 15.0, shape) # CT noise is generally low in HU
    vol_final = vol_smooth + noise
    
    # Clip limits
    vol_final = np.clip(vol_final, -1000, 1000)
    
    seg_mask = np.zeros(shape, dtype=np.uint8)
    seg_mask[nodule_mask] = 1
    
    return vol_final, seg_mask

def generate_pet_ct_3d(grid_size=64):
    """
    Generates a co-registered PET-CT pair of the head/neck.
    CT volume: anatomical details (HU)
    PET volume: metabolic activity (Standardized Uptake Value, SUV, 0.0 to 15.0)
    Returns:
        ndarray: CT volume
        ndarray: PET volume
        ndarray: tumor segmentation mask (0: background, 1: primary tumor)
    """
    shape = (grid_size, grid_size, grid_size)
    center = (grid_size // 2, grid_size // 2, grid_size // 2)
    
    # 1. CT Neck anatomy
    ct = np.full(shape, -1000.0) # air around neck
    # Neck column (soft tissue, e.g. 40 HU)
    neck_mask = create_ellipsoid_mask(shape, center, (grid_size * 0.48, grid_size * 0.40, grid_size * 0.40))
    ct[neck_mask] = 45.0
    
    # Spine (back of neck, bone, e.g. 350 HU)
    spine_mask = create_ellipsoid_mask(shape, (grid_size * 0.50, grid_size * 0.50, grid_size * 0.22), (grid_size * 0.48, grid_size * 0.08, grid_size * 0.08))
    ct[spine_mask] = 380.0
    
    # Trachea (airway, dark -1000 HU)
    trachea_mask = create_ellipsoid_mask(shape, (grid_size * 0.50, grid_size * 0.50, grid_size * 0.65), (grid_size * 0.48, grid_size * 0.06, grid_size * 0.06))
    ct[trachea_mask] = -950.0
    
    # Neck tumor (soft tissue density, e.g. 50 HU, located at left lateral neck)
    tumor_center = (grid_size * 0.48, grid_size * 0.32, grid_size * 0.55)
    tumor_mask = create_ellipsoid_mask(shape, tumor_center, (6.0, 5.0, 5.5))
    ct[tumor_mask] = 52.0
    
    # Smooth CT
    ct_smooth = gaussian_filter(ct, sigma=0.7)
    ct_noise = np.random.normal(0, 10.0, shape)
    ct_final = np.clip(ct_smooth + ct_noise, -1000, 800)
    
    # 2. PET Volume (SUV values: 0.0 to 12.0+)
    # Base low-level background activity (SUV ~ 0.5)
    pet = np.random.uniform(0.1, 0.4, shape)
    
    # Normal physiological uptake in salivary glands / throat muscles (SUV ~ 1.5 - 2.5)
    salivary_l = create_ellipsoid_mask(shape, (grid_size * 0.65, grid_size * 0.35, grid_size * 0.38), (3.0, 3.0, 4.0))
    salivary_r = create_ellipsoid_mask(shape, (grid_size * 0.65, grid_size * 0.65, grid_size * 0.38), (3.0, 3.0, 4.0))
    pet[salivary_l | salivary_r] = np.random.uniform(1.8, 2.4, pet[salivary_l | salivary_r].shape)
    
    # Vocal cords / larynx normal activity (SUV ~ 1.5)
    vocal_mask = create_ellipsoid_mask(shape, (grid_size * 0.42, grid_size * 0.50, grid_size * 0.55), (2.0, 3.0, 3.0))
    pet[vocal_mask] = np.random.uniform(1.2, 1.8, pet[vocal_mask].shape)
    
    # High metabolic tumor activity (SUV ~ 8.0 - 12.0, overlapping CT tumor mask)
    # Tumor has high uptake, especially at the margins, and slightly lower in center if necrotic
    tumor_edge = create_ellipsoid_mask(shape, tumor_center, (6.0, 5.0, 5.5))
    tumor_core = create_ellipsoid_mask(shape, tumor_center, (2.5, 2.0, 2.2))
    
    pet[tumor_edge] = np.random.uniform(8.5, 11.5, pet[tumor_edge].shape)
    pet[tumor_core] = np.random.uniform(4.5, 6.0, pet[tumor_core].shape) # necrotic center
    
    # Zero out PET activity outside neck body contour
    pet[~neck_mask] *= 0.1
    
    # Smooth PET to simulate low spatial resolution of PET imaging (sigma = 1.8)
    pet_smooth = gaussian_filter(pet, sigma=1.5)
    pet_noise = np.random.normal(0, 0.15, shape)
    pet_final = np.clip(pet_smooth + pet_noise, 0.0, 15.0)
    
    seg_mask = np.zeros(shape, dtype=np.uint8)
    seg_mask[tumor_mask] = 1
    
    return ct_final, pet_final, seg_mask

def simulate_pipeline_stages(modality="Brain MRI"):
    """
    Simulates the image processing pipeline stages.
    Returns a dict of 3D volumes representing:
    - Raw: Unaligned/noise/bias field scan
    - Preprocessed: De-noised, N4 bias corrected, normalized
    - Registered: Scans aligned
    - Segmented: Volume with predicted segmentation masks
    - Features: Extracted radiomic metrics (dict)
    """
    if modality == "Brain MRI":
        vols, mask = generate_brain_mri_3d()
        raw_t1c = vols['T1c'].copy()
        
        # Simulate bias field (intensity inhomogeneity) by multiplying with a spatial gradient
        sz, sy, sx = raw_t1c.shape
        z, y, x = np.ogrid[:sz, :sy, :sx]
        bias_field = 0.7 + 0.6 * (x / sx) # gradient from left to right
        raw_t1c = raw_t1c * bias_field
        
        # Add extra noise to raw
        raw_t1c += np.random.normal(0, 12.0, raw_t1c.shape)
        raw_t1c = np.clip(raw_t1c, 0, 255)
        
        # Features calculation
        pixel_spacing = 1.0 # mm
        tumor_volume_cc = float(np.sum(mask > 0) * (pixel_spacing ** 3) / 1000.0)
        necro_volume_cc = float(np.sum(mask == 1) * (pixel_spacing ** 3) / 1000.0)
        edema_volume_cc = float(np.sum(mask == 3) * (pixel_spacing ** 3) / 1000.0)
        enhancing_volume_cc = float(np.sum(mask == 2) * (pixel_spacing ** 3) / 1000.0)
        
        features = {
            "Tumor Volume (cc)": round(tumor_volume_cc, 2),
            "Necrotic Core Vol (cc)": round(necro_volume_cc, 2),
            "Enhancing Tumor Vol (cc)": round(enhancing_volume_cc, 2),
            "Edema Volume (cc)": round(edema_volume_cc, 2),
            "Sphericity (Shape metric)": 0.74,
            "Mean Intensity (Enhancing)": round(float(np.mean(vols['T1c'][mask == 2])), 1),
            "Entropy (Texture metric)": 4.12
        }
        
        return {
            'Raw': raw_t1c,
            'Preprocessed': vols['T1c'],
            'Registered': vols['T1c'], # Same for single modality
            'Segmented': vols['T1c'],
            'Mask': mask,
            'Features': features,
            'AllMod': vols
        }
        
    elif modality == "Lung CT":
        vol, mask = generate_lung_ct_3d()
        
        # Raw: slice misalignment and reconstruction artifacts
        raw_ct = vol.copy()
        raw_ct += np.random.normal(0, 35.0, raw_ct.shape) # higher noise
        
        # Preprocessed is normalized and noise-filtered
        features = {
            "Nodule Diameter (mm)": 7.4,
            "Nodule Volume (mm3)": round(float(np.sum(mask == 1) * 1.0), 1),
            "Mean Density (HU)": round(float(np.mean(vol[mask == 1])), 1),
            "Max Density (HU)": round(float(np.max(vol[mask == 1])), 1),
            "Sphericity": 0.88,
            "Malignancy Risk Index": "74% (High Risk)"
        }
        
        return {
            'Raw': raw_ct,
            'Preprocessed': vol,
            'Registered': vol,
            'Segmented': vol,
            'Mask': mask,
            'Features': features
        }
        
    else: # PET-CT Fusion
        ct, pet, mask = generate_pet_ct_3d()
        
        # Raw: PET and CT are misaligned (shifted by 3 pixels along x and y)
        raw_pet = np.zeros_like(pet)
        # Shift pet by 3 pixels
        raw_pet[3:, 3:] = pet[:-3, :-3]
        
        # Registered: Aligned
        # Segmented: Mask overlay
        features = {
            "Primary Tumor Volume (cc)": round(float(np.sum(mask == 1) * 1.0 / 1000.0), 2),
            "SUV Max (PET activity)": round(float(np.max(pet)), 2),
            "SUV Mean (PET activity)": round(float(np.mean(pet[mask == 1])), 2),
            "Total Lesion Glycolysis (TLG)": round(float(np.sum(pet[mask == 1]) * 1.0 / 1000.0), 2),
            "Metabolic Tumor Volume (cc)": round(float(np.sum(pet > 2.5) * 1.0 / 1000.0), 2)
        }
        
        return {
            'Raw_CT': ct,
            'Raw_PET': raw_pet,
            'Preprocessed_CT': ct,
            'Preprocessed_PET': pet, # assumed denoised
            'CT': ct,
            'PET': pet,
            'Mask': mask,
            'Features': features
        }
