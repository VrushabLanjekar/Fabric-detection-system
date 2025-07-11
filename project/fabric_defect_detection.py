import numpy as np
import cv2
import streamlit as st
from skimage import filters, measure, feature, morphology
from skimage.filters import gabor, median
from skimage.morphology import disk, square
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import probabilistic_hough_line
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray, rgb2hsv
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Advanced Fabric QA System",
    page_icon="🧵",
    layout="wide"
)

# Custom CSS (maintained exactly as provided)
st.markdown("""
    <style>
        .main {background-color: #0E1117}
        h1 {color: #FFD700; text-align: center; font-family: 'Helvetica Neue'}
        .stSlider > div > div > div {background: #FFD700 !important}
        .st-bq {font-family: 'Helvetica Neue'}
        .defect-box {border: 2px solid #FF4444; border-radius: 8px; padding: 15px; margin: 10px 0}
        .feature-box {border: 1px solid #4CAF50; border-radius: 8px; padding: 15px; margin: 10px 0}
    </style>
""", unsafe_allow_html=True)

# Main content (preserved exactly)
st.title("🧵 Advanced Industrial Fabric Defect Detection System")
st.markdown("---")

# Sidebar controls (maintained structure, added only necessary parameters)
with st.sidebar:
    st.header("System Parameters")
    sensitivity = st.slider("Detection Sensitivity", 1, 10, 5)
    min_defect_size = st.slider("Minimum Defect Size (pixels)",  1000)
    st.markdown("---")
    st.caption("Advanced Settings:")
    check_texture = st.checkbox("Texture Analysis (GLCM+LBP)", True)
    check_structural = st.checkbox("Structural Analysis (Gabor+FFT)", True)
    check_color = st.checkbox("Color Consistency Check (HSV)", True)
    check_weaving = st.checkbox("Weaving Pattern Analysis", True)

# Added original preprocessing logic
def original_preprocessing(image):
    """Your original processing pipeline integrated"""
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (10,10))
    dst = cv2.fastNlMeansDenoising(blur, None, 10, 7, 21)
    _, binary = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    return gray, blur, dst, binary, erosion, dilation

# Modified detect_defects to integrate your logic
def detect_defects(image):
    """Integrated defect detection combining both approaches"""
    # Your original processing
    gray, blur, denoised, binary, erosion, dilation = original_preprocessing(image)
    
    # Existing advanced processing
    enhanced = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(enhanced)
    
    defects = {
        'holes': [], 'stains': [], 'structural': [],
        'color': [], 'weaving': [], 'basic_contour': []
    }

    # Your original contour detection logic
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_defect_size and area < 261121.0:  # Your original threshold
            x, y, w, h = cv2.boundingRect(cnt)
            defects['basic_contour'].append({
                'type': 'Structural Defect',
                'area': area,
                'location': (x, y),
                'contour': cnt
            })

    # Existing advanced analysis (preserved exactly)
    if check_texture:
        glcm = graycomatrix(enhanced, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        if contrast > 500 + (100 * sensitivity):
            defects['stains'].append({'type': 'Texture Anomaly', 'severity': contrast})

    if check_structural:
        fft = np.fft.fft2(enhanced)
        fft_shift = np.fft.fftshift(fft)
        magnitude = 20 * np.log(np.abs(fft_shift))
        if np.std(magnitude) > 15 * (11 - sensitivity):
            defects['structural'].append({'type': 'Pattern Defect', 'severity': np.std(magnitude)})

    if check_color:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if np.std(hsv[:,:,0]) > 0.5 + (0.1 * sensitivity):
            defects['color'].append({'type': 'Color Defect', 'severity': np.std(hsv[:,:,0])})

    return defects, gray, blur, denoised, binary, erosion, dilation, enhanced

# Modified visualization to show original processing pipeline
def visualize_results(image, defects, processed_imgs):
    """Integrated visualization showing both pipelines"""
    display_img = image.copy()
    
    # Draw contours from original logic
    for defect in defects['basic_contour']:
        cv2.drawContours(display_img, [defect['contour']], -1, (0,0,255), 2)
        x, y = defect['location']
        cv2.putText(display_img, defect['type'], (x,y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    
    return display_img, processed_imgs

# Main processing flow (preserved structure with additions)
def main():
    uploaded_file = st.file_uploader("Upload Fabric Image", type=["jpg", "jpeg",'webp', "png", "bmp"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        
        with st.spinner('Analyzing fabric with combined techniques...'):
            defects, gray, blur, denoised, binary, erosion, dilation, enhanced = detect_defects(image)
            display_img, processed_imgs = visualize_results(image, defects, 
                (gray, blur, denoised, binary, erosion, dilation, enhanced))
            
            with col1:
                st.image(image, caption="Original Fabric", use_column_width=True, channels="BGR")
                
                # Your original processing pipeline display
                st.markdown("### Basic Processing Pipeline")
                tabs = st.tabs(["Gray", "Blur", "Denoised", "Binary", "Erosion", "Dilation"])
                with tabs[0]: st.image(gray, caption="Grayscale")
                with tabs[1]: st.image(blur, caption="Blurred")
                with tabs[2]: st.image(denoised, caption="Denoised")
                with tabs[3]: st.image(binary, caption="Binary Threshold")
                with tabs[4]: st.image(erosion, caption="Erosion")
                with tabs[5]: st.image(dilation, caption="Dilation")
            
            with col2:
                st.image(display_img, caption="Defect Visualization", use_column_width=True, channels="BGR")
                
                # Existing defect reporting
                total_defects = sum(len(v) for v in defects.values())
                if total_defects > 1:
                    st.error(f"❌ REJECTED - {total_defects} Defects Found")
                    tabs = st.tabs([k.upper() for k in defects.keys() if defects[k]])
                    for tab, (defect_type, defect_list) in zip(tabs, defects.items()):
                        if defect_list:
                            with tab:
                                for defect in defect_list:
                                    details = f"""
                                    <div class="defect-box">
                                        <b>Type:</b> {defect['type']}<br>
                                        {f"<b>Size:</b> {defect['area']} px²" if 'area' in defect else ""}
                                        {f"<b>Severity:</b> {defect['severity']:.2f}" if 'severity' in defect else ""}
                                    </div>
                                    """
                                    st.markdown(details, unsafe_allow_html=True)
                else:
                    st.success("✅ APPROVED - No Defects Detected")

    else:
        st.markdown("""
        <div style='border: 2px dashed #FFD700; border-radius: 10px; padding: 40px 20px; text-align: center; margin: 20px 0;'>
            <h3 style='color: #FFD700;'>Upload Fabric Image to Begin Inspection</h3>
            <p>Supported formats: JPEG, PNG, BMP</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()