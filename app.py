"""
Streamlit UI for Bone Tumor X-Ray Analysis
===========================================

Simple medical imaging interface that integrates with existing CNN models,
VLM encoder, and LLM chat engine.

ARCHITECTURE:
- Classification: EfficientNet-B0 (frozen)
- Segmentation: MobileNetV2-UNet (frozen)
- VLM: CLIP encoder (frozen)
- LLM: TinyLlama-1.1B-Chat (instruction-tuned)

UI FLOW:
1. User uploads X-ray image
2. User clicks "Analyze Image"
3. System runs CNNs (classification + segmentation)
4. Results cached in session state
5. User asks questions → LLM responds using cached results only

CONSTRAINTS:
- LLM NEVER re-runs inference
- All responses grounded in CNN predictions
- Medical disclaimers enforced
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json

# Add project modules to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'common'))
sys.path.append(str(project_root / 'classification'))
sys.path.append(str(project_root / 'segmentation'))
sys.path.append(str(project_root / 'multimodal'))

# Import existing pipeline (MANDATORY - DO NOT RECREATE)
from multimodal.pipeline import MultimodalPipeline


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'structured_facts' not in st.session_state:
        st.session_state.structured_facts = None
    if 'segmentation_mask' not in st.session_state:
        st.session_state.segmentation_mask = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


# ============================================================================
# PIPELINE LOADING (GPU ONLY)
# ============================================================================

@st.cache_resource
def load_pipeline():
    """
    Load the complete multimodal pipeline.
    
    This function is cached to avoid reloading models on every interaction.
    Models are loaded ONCE and reused throughout the session.
    
    Returns:
        pipeline: Initialized MultimodalPipeline with all models loaded
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Path configurations
    cls_checkpoint = project_root / 'classification' / 'outputs' / 'checkpoint_best.pth'
    seg_checkpoint = project_root / 'segmentation' / 'outputs' / 'checkpoint_best.pth'
    label_encoding = project_root / 'label_encoding.json'
    
    # Initialize pipeline with REAL LLM (TinyLlama-1.1B-Chat)
    # Set llm_model_name=None for mock LLM (faster but template-based responses)
    pipeline = MultimodalPipeline(
        classification_checkpoint=str(cls_checkpoint),
        segmentation_checkpoint=str(seg_checkpoint),
        label_encoding_path=str(label_encoding),
        vlm_model_name='RN50',  # CLIP encoder
        llm_model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',  # Real LLM
        device=device
    )
    
    return pipeline


# ============================================================================
# IMAGE ANALYSIS (CNN INFERENCE)
# ============================================================================

def run_analysis(uploaded_file):
    """
    Run complete CNN inference pipeline on uploaded image.
    
    CRITICAL FLOW:
    1. Save uploaded image to temp file
    2. Call pipeline.run_cnn_inference() - EXISTING FUNCTION
    3. Extract structured facts (tumor class, confidence, mask, etc.)
    4. Store results in session state
    5. Results are cached and reused for all subsequent chat queries
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        None (results stored in session state)
    """
    # Save uploaded image temporarily
    temp_path = project_root / 'temp_upload.jpg'
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Get pipeline (cached)
    pipeline = st.session_state.pipeline
    
    # Run CNN inference (classification + segmentation)
    # This calls the EXISTING run_cnn_inference() method
    with st.spinner('🔬 Running CNN inference...'):
        structured_facts = pipeline.run_cnn_inference(str(temp_path))
    
    # Store results in session state (CRITICAL FOR CHAT)
    st.session_state.structured_facts = structured_facts
    st.session_state.segmentation_mask = structured_facts['segmentation_mask']
    st.session_state.uploaded_image = Image.open(temp_path).convert('RGB')
    st.session_state.analysis_done = True
    
    # Update pipeline's internal state
    pipeline.current_facts = structured_facts
    
    # Clean up temp file
    temp_path.unlink()


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_overlay_image(original_image, mask):
    """
    Create overlay visualization of segmentation mask on original image.
    
    Args:
        original_image: PIL Image
        mask: numpy array (H, W) with binary mask
        
    Returns:
        overlay: PIL Image with red overlay on tumor regions
    """
    # Resize image to match mask dimensions
    target_size = (mask.shape[1], mask.shape[0])  # (width, height)
    img_array = np.array(original_image.resize(target_size))
    
    # Ensure mask is 2D
    if len(mask.shape) == 3:
        mask = mask.squeeze()
    
    # Create red overlay for tumor regions
    overlay = img_array.copy()
    red_mask = np.zeros_like(overlay)
    
    # Apply red color to tumor regions (mask must be 2D boolean)
    tumor_mask = mask > 0.5
    red_mask[tumor_mask] = [255, 0, 0]  # Red color
    
    # Blend overlay (40% transparency)
    overlay = (0.6 * overlay + 0.4 * red_mask).astype(np.uint8)
    
    return Image.fromarray(overlay)


# ============================================================================
# CHAT INTERFACE
# ============================================================================

def handle_chat_message(user_message):
    """
    Handle user chat message and generate LLM response.
    
    CRITICAL CONSTRAINT:
    - LLM uses ONLY the cached structured_facts
    - NO re-inference happens
    - All responses grounded in CNN predictions
    
    Args:
        user_message: User's question string
        
    Returns:
        response: LLM-generated answer (grounded and safety-filtered)
    """
    pipeline = st.session_state.pipeline
    structured_facts = st.session_state.structured_facts
    
    # Call EXISTING chat method (uses ChatEngine internally)
    response = pipeline.chat(user_message)
    
    # Store in chat history for display
    st.session_state.chat_history.append({
        'role': 'user',
        'message': user_message
    })
    st.session_state.chat_history.append({
        'role': 'assistant',
        'message': response
    })
    
    return response


# ============================================================================
# MAIN UI
# ============================================================================

def main():
    """
    Main Streamlit application.
    """
    # Page configuration
    st.set_page_config(
        page_title="Bone Tumor X-Ray Analysis",
        page_icon="🦴",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Title
    st.title("🦴 Bone Tumor X-Ray Analysis System")
    st.markdown("**AI-powered medical imaging with CNN classification, segmentation, and LLM chat**")
    
    # Load pipeline (cached)
    if st.session_state.pipeline is None:
        with st.spinner('🔧 Loading models (this may take a minute)...'):
            st.session_state.pipeline = load_pipeline()
        
        # Show which LLM mode is active
        if st.session_state.pipeline.use_mock_llm:
            st.warning('⚠️ Using MOCK LLM (template responses). Real LLM failed to load.')
        else:
            st.success('✅ Models loaded! Using TinyLlama-1.1B for AI-generated responses.')
    
    # Sidebar for image upload
    st.sidebar.header("📤 Upload X-Ray Image")
    
    # Show LLM status in sidebar
    if st.session_state.pipeline.use_mock_llm:
        st.sidebar.error("🤖 Mock LLM Active")
        st.sidebar.caption("Responses are template-based, not AI-generated")
    else:
        st.sidebar.success("🤖 Real LLM Active")
        st.sidebar.caption("Using TinyLlama-1.1B-Chat")
    
    st.sidebar.markdown("---")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose an X-ray image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a bone tumor X-ray image for analysis"
    )
    
    # Analyze button
    if uploaded_file is not None:
        analyze_button = st.sidebar.button("🔬 Analyze Image", type="primary", width='stretch')
        
        if analyze_button:
            # Run CNN inference
            run_analysis(uploaded_file)
            st.sidebar.success("✅ Analysis complete!")
    
    # Main content area
    if st.session_state.analysis_done:
        # Get cached results
        facts = st.session_state.structured_facts
        original_image = st.session_state.uploaded_image
        mask = st.session_state.segmentation_mask
        
        # =====================================================================
        # RESULTS SECTION
        # =====================================================================
        st.header("📊 Analysis Results")
        
        # Display images side-by-side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(original_image, width='stretch')
        
        with col2:
            st.subheader("Segmentation Overlay")
            # Check if mask has any tumor regions
            if mask.sum() > 0:
                overlay_image = create_overlay_image(original_image, mask)
                st.image(overlay_image, width='stretch')
            else:
                # No tumor detected in segmentation
                st.image(original_image, width='stretch')
                st.warning("⚠️ No tumor region detected by segmentation model")
        
        # Display findings
        st.subheader("🔍 Primary Findings")
        
        # Add context about segmentation performance
        if facts['tumor_area_pct'] == 0:
            st.info(
                "ℹ️ **Note:** The segmentation model did not detect visible tumor boundaries in this image. "
                "This can occur with subtle lesions or poor contrast. The classification model detected **"
                f"{facts['tumor_class']}** based on overall image patterns."
            )
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric(
                label="Tumor Type",
                value=facts['tumor_class']
            )
        
        with col_b:
            confidence_pct = facts['confidence']*100
            # Color code confidence
            if confidence_pct >= 80:
                conf_emoji = "🟢"
            elif confidence_pct >= 50:
                conf_emoji = "🟡"
            else:
                conf_emoji = "🔴"
            
            st.metric(
                label="Confidence",
                value=f"{conf_emoji} {confidence_pct:.1f}%"
            )
        
        with col_c:
            # Show tumor coverage with warning if none detected
            coverage_pct = facts['tumor_area_pct']
            st.metric(
                label="Tumor Coverage",
                value=f"{coverage_pct:.1f}%",
                delta="Not visible" if coverage_pct == 0 else None,
                delta_color="off" if coverage_pct == 0 else "normal"
            )
        
        # Additional details with context
        location = facts['tumor_location']
        if location == 'not detected':
            st.warning(f"📍 **Location:** {location} (segmentation model did not identify tumor boundaries)")
        else:
            st.info(f"📍 **Location:** {location}")
        
        # Top-3 predictions
        st.subheader("🔢 Alternative Diagnoses")
        for idx, (tumor_class, prob) in enumerate(facts['top3_predictions'], 1):
            st.write(f"{idx}. **{tumor_class}** - {prob*100:.1f}%")
        
        # Medical disclaimer
        st.warning(
            "⚠️ **Medical Disclaimer:** This is an AI-generated analysis. "
            "All findings must be validated by a qualified medical professional "
            "before any clinical decisions."
        )
        
        # =====================================================================
        # CHAT SECTION
        # =====================================================================
        col_header, col_reset = st.columns([4, 1])
        with col_header:
            st.header("💬 Ask Questions About This Analysis")
        with col_reset:
            if st.button("🔄 Reset Chat", help="Clear conversation history"):
                st.session_state.chat_history = []
                if hasattr(st.session_state.pipeline, 'chat_engine') and st.session_state.pipeline.chat_engine:
                    st.session_state.pipeline.chat_engine.reset_conversation()
                st.rerun()
        
        st.markdown("Ask about the detected tumor, confidence, location, treatment info, or request medical explanations.")
        
        # Display chat history with better formatting
        if st.session_state.chat_history:
            st.subheader("Conversation")
            for idx, msg in enumerate(st.session_state.chat_history):
                if msg['role'] == 'user':
                    with st.chat_message("user"):
                        st.markdown(msg['message'])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(msg['message'])
        
        # Chat input
        with st.form(key='chat_form', clear_on_submit=True):
            user_message = st.text_area(
                "Your question:",
                placeholder="Examples:\\n• What is osteosarcoma?\\n• Why is the confidence low?\\n• Can you explain the alternative diagnoses?\\n• What does the location mean?",
                height=100,
                key='chat_input'
            )
            col1, col2 = st.columns([1, 5])
            with col1:
                submit_button = st.form_submit_button("Send 🚀")
            with col2:
                st.caption("💡 Tip: Ask follow-up questions for more details")
            
            if submit_button and user_message.strip():
                # Generate response with progress indicator
                with st.spinner('🤔 Generating response using LLM...'):
                    response = handle_chat_message(user_message.strip())
                
                # Show success message briefly
                st.success("✅ Response generated!")
                
                # Force rerun to display updated chat history
                st.rerun()
    
    else:
        # No analysis yet - show instructions
        st.info(
            "👆 **Get Started:**\n\n"
            "1. Upload an X-ray image using the sidebar\n"
            "2. Click 'Analyze Image' to run CNN inference\n"
            "3. View results and ask questions about the findings"
        )
        
        # Show example
        st.subheader("ℹ️ About This System")
        st.markdown("""
        This system uses state-of-the-art deep learning models to analyze bone tumor X-rays:
        
        - **Classification:** EfficientNet-B0 (9 tumor classes)
        - **Segmentation:** MobileNetV2-UNet (tumor region detection)
        - **Chat:** TinyLlama-1.1B-Chat (medical explanation generation)
        
        The system provides:
        ✅ Tumor type classification with confidence scores  
        ✅ Tumor segmentation and location detection  
        ✅ Interactive Q&A about the findings  
        ✅ Medical disclaimers and safety filters  
        """)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
