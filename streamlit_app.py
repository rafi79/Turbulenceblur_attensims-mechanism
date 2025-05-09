import streamlit as st
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import google.generativeai as genai
import PIL.Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from einops import rearrange
import tempfile
import os
from typing import List, Dict, Tuple, Optional, Union
import time

# Helper functions for attention
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        
        return out

class TemporalAttention(nn.Module):
    """Attention mechanism that works across frames in a video sequence"""
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x_sequence):
        """
        x_sequence: list of tensors, each with shape [b, c, h, w]
        returns: list of tensors with same shape
        """
        if len(x_sequence) == 1:
            # If only one frame, just return it
            return x_sequence
            
        # Stack along batch dimension temporarily
        seq_len = len(x_sequence)
        x = torch.cat(x_sequence, dim=0)  # [seq_len*b, c, h, w]
        b, c, h, w = x_sequence[0].shape
        
        # Project to q, k, v
        q = self.q(x)  # [seq_len*b, c, h, w]
        k = self.k(x)
        v = self.v(x)
        
        # Reshape for attention
        q = rearrange(q, '(s b) c h w -> (b h w) s c', s=seq_len, b=b)
        k = rearrange(k, '(s b) c h w -> (b h w) s c', s=seq_len, b=b)
        v = rearrange(v, '(s b) c h w -> (b h w) s c', s=seq_len, b=b)
        
        # Multi-head split
        q = rearrange(q, 'p s (head c) -> p s head c', head=self.num_heads)
        k = rearrange(k, 'p s (head c) -> p s head c', head=self.num_heads)
        v = rearrange(v, 'p s (head c) -> p s head c', head=self.num_heads)
        
        # Normalize
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        # Apply attention
        out = (attn @ v)  # [p, s, head, c]
        
        # Reshape back
        out = rearrange(out, 'p s head c -> (s b) (head c) h w', s=seq_len, b=b, h=h, w=w)
        out = self.project_out(out)
        
        # Split back to sequence
        out_sequence = torch.chunk(out, seq_len, dim=0)
        return list(out_sequence)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2.66):
        super().__init__()
        
        self.norm1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = LayerNorm(dim)
        
        hidden_features = int(dim * ffn_expansion_factor)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_features, dim, kernel_size=1, bias=True)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class VideoDeblurNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=48, num_spatial_blocks=6, num_temporal_blocks=2):
        super().__init__()
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=True)
        
        # Spatial transformer blocks
        self.spatial_transformer_blocks = nn.ModuleList([
            TransformerBlock(dim) for _ in range(num_spatial_blocks)
        ])
        
        # Temporal attention blocks
        self.temporal_attention_blocks = nn.ModuleList([
            TemporalAttention(dim) for _ in range(num_temporal_blocks)
        ])
        
        # Output projection
        self.conv_last = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=True)
        
    def forward_single(self, x):
        """Process a single frame"""
        # Feature extraction
        feat = self.conv_first(x)
        
        # Apply transformer blocks
        for block in self.spatial_transformer_blocks:
            feat = block(feat)
        
        return feat
    
    def forward_sequence(self, x_sequence):
        """Process a sequence of frames with temporal attention"""
        # First apply spatial attention to each frame independently
        features = [self.forward_single(x) for x in x_sequence]
        
        # Then apply temporal attention
        for temporal_block in self.temporal_attention_blocks:
            features = temporal_block(features)
        
        # Final projection for each frame
        outputs = [self.conv_last(feat) + x_sequence[i] for i, feat in enumerate(features)]
        outputs = [torch.clamp(out, 0, 1) for out in outputs]
        
        return outputs

class AdvancedMediaExtractor:
    def __init__(self, api_key='AIzaSyB-VpIY25J2Mo13Q8h26Au5W218SHO6dPs'):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
        
        # Initialize turbulence deblur model
        self.deblur_model = VideoDeblurNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.deblur_model.to(self.device)
        self.deblur_model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.inverse_transform = transforms.Compose([
            transforms.ToPILImage()
        ])
    
    def gentle_enhance(self, image):
        """Gentle traditional enhancement for better text readability"""
        # Convert to float32
        img_float = image.astype(np.float32) / 255.0
        
        # Mild denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            (img_float * 255).astype(np.uint8),
            None,
            h=10,  # Reduced strength
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        ).astype(np.float32) / 255.0
        
        # Very gentle sharpening
        kernel_sharp = np.array([[-0.5,-0.5,-0.5],
                               [-0.5, 5,-0.5],
                               [-0.5,-0.5,-0.5]]) * 0.5
        
        if len(image.shape) == 3:
            sharpened = np.zeros_like(denoised)
            for i in range(3):
                sharpened[..., i] = cv2.filter2D(denoised[..., i], -1, kernel_sharp)
        else:
            sharpened = cv2.filter2D(denoised, -1, kernel_sharp)
        
        # Subtle contrast enhancement
        enhanced = cv2.convertScaleAbs((sharpened * 255).astype(np.uint8), alpha=1.1, beta=0)
        
        return enhanced
    
    def attention_deblur_image(self, image):
        """Apply attention-based turbulence deblurring to a single image"""
        with torch.no_grad():
            # Convert to RGB if needed
            if image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
                
            # Convert to tensor
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Process with spatial attention only
            features = self.deblur_model.forward_single(input_tensor)
            output_tensor = self.deblur_model.conv_last(features) + input_tensor
            output_tensor = torch.clamp(output_tensor, 0, 1)
            
            # Convert back to numpy
            output_image = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_image = (output_image * 255).astype(np.uint8)
            
            # Convert back to BGR if needed
            if image.shape[2] == 3:
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
                
            return output_image
    
    def process_video_frames(self, video_path, key_frame_interval=5, max_frames=30):
        """Process video with attention mechanisms, returning key frames and their text"""
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Failed to open video file"}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frames to process
        if total_frames > max_frames * key_frame_interval:
            # If video is too long, sample evenly
            frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        else:
            # Otherwise, take every key_frame_interval frame
            frame_indices = np.arange(0, total_frames, key_frame_interval, dtype=int)
            
        # Extract frames
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            return {"error": "No frames extracted from video"}
        
        # Process frames with attention
        processed_frames = []
        frame_tensors = []
        
        for frame in frames:
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to tensor
            frame_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
            frame_tensors.append(frame_tensor)
        
        # Process with spatiotemporal attention
        with torch.no_grad():
            output_tensors = self.deblur_model.forward_sequence(frame_tensors)
            
            # Convert back to numpy
            for output_tensor in output_tensors:
                output_frame = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output_frame = (output_frame * 255).astype(np.uint8)
                processed_frames.append(output_frame)
        
        # Extract text from key frames
        text_results = []
        for i, (original, processed) in enumerate(zip(frames, processed_frames)):
            # Convert to PIL Image
            original_pil = PIL.Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            processed_pil = PIL.Image.fromarray(processed)
            
            prompt = f"""
            Analyze this video frame (frame {i+1} of {len(frames)}).
            
            1. Extract any visible text in the frame
            2. Describe the main objects and actions visible
            3. Note any important information displayed (numbers, indicators, text)
            
            If text is unclear or partially visible, indicate with [...].
            """
            
            # Generate content for both original and processed
            response_original = self.model.generate_content([prompt, original_pil])
            response_processed = self.model.generate_content([prompt, processed_pil])
            
            response_original.resolve()
            response_processed.resolve()
            
            text_results.append({
                "frame_index": frame_indices[i],
                "timestamp": frame_indices[i] / fps,
                "original_analysis": response_original.text.strip(),
                "enhanced_analysis": response_processed.text.strip()
            })
        
        # Create a summary using Gemini
        summary_prompt = f"""
        You've analyzed {len(frames)} frames from a video. Based on all the text and visual information extracted, 
        provide a comprehensive summary of what this video shows and any key information it contains.
        Focus especially on any textual information that appears consistently or changes throughout the video.
        """
        
        # Create a context with descriptions of all frames
        context = "\n\n".join([f"Frame {r['frame_index']} ({r['timestamp']:.2f}s): {r['enhanced_analysis']}" 
                              for r in text_results])
        
        summary_response = self.model.generate_content([summary_prompt, context])
        summary_response.resolve()
        
        return {
            "total_frames": total_frames,
            "frames_processed": len(frames),
            "fps": fps,
            "duration": total_frames / fps,
            "frame_results": text_results,
            "video_summary": summary_response.text.strip(),
            "processed_frames": processed_frames
        }
    
    def extract_text(self, image):
        """Extract text from image"""
        try:
            # Apply traditional enhancement
            enhanced_traditional = self.gentle_enhance(image)
            
            # Apply attention-based deblurring
            enhanced_attention = self.attention_deblur_image(image)
            
            # Convert to PIL Image
            original_pil = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            enhanced_traditional_pil = PIL.Image.fromarray(cv2.cvtColor(enhanced_traditional, cv2.COLOR_BGR2RGB))
            enhanced_attention_pil = PIL.Image.fromarray(cv2.cvtColor(enhanced_attention, cv2.COLOR_BGR2RGB))
            
            prompt = """
            Read and extract the text from this image.
            The text appears to be a quote or saying.
            Please return the exact text with:
            - Proper line breaks
            - Correct punctuation
            - Any attribution or source (like website names)
            
            Focus on accuracy rather than making corrections.
            If any part is unclear, mark it with [...].
            """
            
            # Try with all three versions
            response_original = self.model.generate_content([prompt, original_pil])
            response_traditional = self.model.generate_content([prompt, enhanced_traditional_pil])
            response_attention = self.model.generate_content([prompt, enhanced_attention_pil])
            
            response_original.resolve()
            response_traditional.resolve()
            response_attention.resolve()
            
            # Return all results
            return {
                'original': response_original.text.strip(),
                'traditional': response_traditional.text.strip(),
                'attention': response_attention.text.strip()
            }
            
        except Exception as e:
            return f"Text extraction error: {str(e)}"

def main():
    st.title("Advanced Media Processing with Attention Mechanisms")
    
    # Sidebar for API key
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        value="AIzaSyB-VpIY25J2Mo13Q8h26Au5W218SHO6dPs",
        type="password"
    )
    
    # Mode selection
    mode = st.radio("Select mode:", ["Image Processing", "Video Analysis"])
    
    # Initialize extractor
    extractor = AdvancedMediaExtractor(api_key)
    
    if mode == "Image Processing":
        # Image upload
        uploaded_file = st.file_uploader("Upload an image with text", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display original image
            st.subheader("Input Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if st.button("Process Image and Extract Text"):
                with st.spinner("Processing with attention mechanisms..."):
                    # Apply traditional enhancement
                    enhanced_traditional = extractor.gentle_enhance(image)
                    
                    # Apply attention-based deblurring
                    enhanced_attention = extractor.attention_deblur_image(image)
                    
                    # Display enhanced images
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Traditional Enhancement")
                        st.image(cv2.cvtColor(enhanced_traditional, cv2.COLOR_BGR2RGB))
                    
                    with col2:
                        st.subheader("Attention-Based Deblurring")
                        st.image(cv2.cvtColor(enhanced_attention, cv2.COLOR_BGR2RGB))
                    
                    # Extract text
                    results = extractor.extract_text(image)
                    
                    if isinstance(results, dict):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.subheader("From Original")
                            st.text_area("Original Text", results['original'], height=200)
                        
                        with col2:
                            st.subheader("From Traditional")
                            st.text_area("Traditional Enhanced", results['traditional'], height=200)
                        
                        with col3:
                            st.subheader("From Attention Model")
                            st.text_area("Attention Enhanced", results['attention'], height=200)
                        
                        # Combined results download
                        combined_text = f"Original Image Text:\n{results['original']}\n\n"
                        combined_text += f"Traditional Enhanced Text:\n{results['traditional']}\n\n"
                        combined_text += f"Attention Enhanced Text:\n{results['attention']}"
                        
                        st.download_button(
                            "Download All Results",
                            data=combined_text,
                            file_name="extracted_text.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error(results)
    
    else:  # Video Analysis
        # Video upload
        uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])
        
        if uploaded_file is not None:
            # Save video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            tfile.close()
            
            # Video parameters
            col1, col2 = st.columns(2)
            with col1:
                key_frame_interval = st.slider("Key frame interval", 1, 30, 5)
            with col2:
                max_frames = st.slider("Maximum frames to process", 5, 50, 10)
            
            if st.button("Process Video"):
                with st.spinner("Processing video with spatiotemporal attention..."):
                    # Process video
                    results = extractor.process_video_frames(
                        video_path, 
                        key_frame_interval=key_frame_interval,
                        max_frames=max_frames
                    )
                    
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        # Display video summary
                        st.subheader("Video Summary")
                        st.write(results["video_summary"])
                        
                        # Display video stats
                        st.subheader("Video Statistics")
                        st.write(f"Duration: {results['duration']:.2f} seconds")
                        st.write(f"Total frames: {results['total_frames']}")
                        st.write(f"Frames processed: {results['frames_processed']}")
                        st.write(f"FPS: {results['fps']:.2f}")
                        
                        # Display processed frames
                        st.subheader("Processed Key Frames")
                        # Create a grid of frames
                        cols = 3  # Number of columns in the grid
                        rows = (len(results["processed_frames"]) + cols - 1) // cols  # Calculate rows needed
                        
                        for i in range(rows):
                            row_cols = st.columns(cols)
                            for j in range(cols):
                                idx = i * cols + j
                                if idx < len(results["processed_frames"]):
                                    with row_cols[j]:
                                        st.image(
                                            results["processed_frames"][idx],
                                            caption=f"Frame {results['frame_results'][idx]['frame_index']} ({results['frame_results'][idx]['timestamp']:.2f}s)",
                                            use_column_width=True
                                        )
                        
                        # Display frame analysis
                        st.subheader("Frame-by-Frame Analysis")
                        for i, result in enumerate(results["frame_results"]):
                            with st.expander(f"Frame {result['frame_index']} ({result['timestamp']:.2f}s)"):
                                st.write("**Enhanced Analysis:**")
                                st.write(result["enhanced_analysis"])
                                st.write("**Original Analysis:**")
                                st.write(result["original_analysis"])
                        
                        # Prepare downloadable report
                        report = f"# Video Analysis Report\n\n"
                        report += f"## Video Summary\n{results['video_summary']}\n\n"
                        report += f"## Video Statistics\n"
                        report += f"- Duration: {results['duration']:.2f} seconds\n"
                        report += f"- Total frames: {results['total_frames']}\n"
                        report += f"- Frames processed: {results['frames_processed']}\n"
                        report += f"- FPS: {results['fps']:.2f}\n\n"
                        report += f"## Frame-by-Frame Analysis\n\n"
                        
                        for result in results["frame_results"]:
                            report += f"### Frame {result['frame_index']} ({result['timestamp']:.2f}s)\n"
                            report += f"**Enhanced Analysis:**\n{result['enhanced_analysis']}\n\n"
                            report += f"**Original Analysis:**\n{result['original_analysis']}\n\n"
                        
                        st.download_button(
                            "Download Complete Analysis Report",
                            data=report,
                            file_name="video_analysis_report.md",
                            mime="text/markdown"
                        )
            
            # Clean up temp file
            try:
                os.unlink(video_path)
            except:
                pass

if __name__ == "__main__":
    main()
