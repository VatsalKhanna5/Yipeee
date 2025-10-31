import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import io
import hashlib

# --- 1. Model Definition (3D U-Net Architecture) ---

class DoubleConv(nn.Module):
    """(Conv3D -> GroupNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with MaxPool3d followed by DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.encoder(x)

class Up(nn.Module):
    """Upscaling with trilinear Upsample followed by DoubleConv"""
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        # Use simple trilinear upsampling
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Center crop x2 to match x1 size for concatenation (Skip Connection)
        # This handles the size mismatch that often occurs in U-Net implementations
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2, 
                         diffH // 2, diffH - diffH // 2, 
                         diffD // 2, diffD - diffD // 2])

        x = torch.cat([x2, x1], dim=1) # Concatenate skip connection
        return self.conv(x)

class Out(nn.Module):
    """Final 1x1 convolution to map to n_classes"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet3d(nn.Module):
    """The full 3D U-Net Architecture for BraTS"""
    def __init__(self, in_channels=4, n_classes=3, n_channels=24):
        super().__init__()
        # 4 Input Modalities -> 3 Output Classes (WT, TC, ET)
        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels) # Bottleneck

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4) # Bottleneck

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask

# --- 2. Caching and Utility Functions ---

def get_file_hash(uploaded_file):
    """Generates a hash of the uploaded file contents for caching."""
    uploaded_file.seek(0)
    file_hash = hashlib.sha256(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return file_hash

@st.cache_resource(hash_funcs={io.BytesIO: get_file_hash})
def load_model(model_file_io):
    """Initializes and loads the model only once per unique weight file."""
    # Use CPU if CUDA is unavailable
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = UNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
    
    # Load state dict from the BytesIO object
    model.load_state_dict(torch.load(model_file_io, map_location=device))
    model.eval()
    return model, device

def postprocess_prediction(prediction_tensor, input_volume):
    """
    Converts 5D prediction tensor to 2D montage masks and returns input montage.
    """
    # 1. Prediction post-processing
    mask = prediction_tensor.squeeze().cpu().detach().numpy()
    
    # Apply sigmoid and threshold (0.5) to get binary masks
    mask = (mask > 0.5).astype(np.float32)

    # Re-order axes from (C, D, H, W) to (C, H, W, D) for correct visualization
    # This is a common requirement when using standard 2D tools on 3D data
    mask_reordered = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

    # Extract individual tumor types
    # The montage function collapses the D-dimension (slice) into a single image.
    mask_WT = np.rot90(montage(mask_reordered[0])) # Whole Tumor
    mask_TC = np.rot90(montage(mask_reordered[1])) # Tumor Core
    mask_ET = np.rot90(montage(mask_reordered[2])) # Enhancing Tumor
    
    # 2. Input post-processing (FLAIR montage)
    # Assuming FLAIR is the first channel (index 0) of the input_volume (C, D, H, W)
    # The input volume is already D, H, W. We need to swap to D, H, W for montage.
    # The input volume is (C, D, H, W). We want (D, H, W) for FLAIR.
    flair_volume = input_volume[0] 
    flair_montage = np.rot90(montage(flair_volume))
    
    return flair_montage, mask_WT, mask_TC, mask_ET

# --- 3. Streamlit Application ---

def main():
    st.set_page_config(layout="wide", page_title="3D U-Net BraTS Segmentation")
    st.title("🧠 3D U-Net Brain Tumor Segmentation")
    st.markdown("---")

    # Sidebar for instructions
    with st.sidebar:
        st.header("Instructions & Requirements")
        st.markdown("""
        This application uses a 3D U-Net for Brain Tumor Segmentation (BraTS 2020).
        
        **Model Inputs:**
        1. **Model Weights:** A `.pth` file containing the trained PyTorch state dictionary.
        2. **MRI Volume:** A single **4D NumPy array** (`.npy` or `.npz` file) with shape `(4, D, H, W)`.
           - **4 Channels:** [FLAIR, T1, T1CE, T2] modalities.
           - **D, H, W:** Depth, Height, Width (e.g., 155, 240, 240).
           
        **Note:** Since we avoid `nibabel`, your NIfTI files **must be pre-processed** into a single, stacked, normalized NumPy array before upload.
        """)

    # --- File Upload Section ---
    st.header("1. Upload Files")
    col_weights, col_input = st.columns(2)

    with col_weights:
        model_file = st.file_uploader(
            "Upload **Model Weights (.pth)**", 
            type=["pth"], 
            help="The file containing the learned model parameters."
        )

    with col_input:
        input_file = st.file_uploader(
            "Upload **4D MRI Volume (.npy or .npz)**", 
            type=["npy", "npz"], 
            help="The stacked, normalized NumPy array (4 channels)."
        )

    # --- Prediction Execution ---
    all_files_uploaded = model_file and input_file
    
    st.markdown("---")
    if st.button("▶️ Run Segmentation Prediction", disabled=not all_files_uploaded, use_container_width=True):
        
        if not all_files_uploaded:
            st.error("Please upload both the model weights and the 4D MRI volume to proceed.")
            return

        try:
            # A. Load Model (Cached)
            with st.spinner("Loading and initializing 3D U-Net model..."):
                model, device = load_model(model_file)
                st.sidebar.info(f"Model loaded successfully on {device.upper()}.")

            # B. Load Input Data
            with st.spinner("Reading and preparing 4D input volume..."):
                # Load NumPy array from BytesIO object
                input_data = np.load(input_file).astype(np.float32)

                # Check shape (must be 4D and start with 4 channels)
                if input_data.ndim != 4 or input_data.shape[0] != 4:
                    st.error(f"Input file must be a 4D array with 4 channels. Detected shape: {input_data.shape}. Expected shape: (4, D, H, W).")
                    return

                # Convert to Tensor: (B=1, C=4, D, H, W) and move to device
                input_tensor = torch.from_numpy(input_data).unsqueeze(0).to(device)
            
            # C. Run Prediction
            with st.spinner("Executing 3D Segmentation..."):
                start_time = time.time()
                with torch.no_grad():
                    output_logits = model(input_tensor)
                    output_probs = torch.sigmoid(output_logits) # Convert logits to probabilities
                
                elapsed_time = time.time() - start_time
            st.success(f"Prediction complete in {elapsed_time:.2f} seconds!")

            # D. Post-processing and Visualization
            st.header("2. Segmentation Results (2D Montage)")
            st.markdown("Visualizing 3D results collapsed into a single 2D montage of all slices.")

            flair_montage, mask_WT, mask_TC, mask_ET = postprocess_prediction(output_probs, input_data)
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.set_title(f"Predicted Segmentation Overlay (Input Shape: {input_data.shape[1:]})", fontsize=16)
            ax.axis("off")

            # 1. Background FLAIR image
            ax.imshow(flair_montage, cmap='bone')

            # 2. Overlay Masks (Alpha blending)
            # WT (Whole Tumor) - Red/Orange
            ax.imshow(np.ma.masked_where(mask_WT == 0, mask_WT),
                      cmap='autumn', alpha=0.6) 
            # TC (Tumor Core) - Yellow/Green
            ax.imshow(np.ma.masked_where(mask_TC == 0, mask_TC),
                      cmap='spring', alpha=0.6)
            # ET (Enhancing Tumor) - Blue/Cyan
            ax.imshow(np.ma.masked_where(mask_ET == 0, mask_ET),
                      cmap='cool', alpha=0.6)
            
            # 3. Create Legend
            legend_patches = [
                plt.matplotlib.patches.Patch(color='cool', label='ET (Enhancing Tumor)'),
                plt.matplotlib.patches.Patch(color='spring', label='TC (Tumor Core)'),
                plt.matplotlib.patches.Patch(color='autumn', label='WT (Whole Tumor)')
            ]
            ax.legend(handles=legend_patches, loc='lower left', framealpha=0.8, fontsize=12)
            
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred during prediction. Please check your file formats and try again. Error: {e}")

if __name__ == "__main__":
    main()
