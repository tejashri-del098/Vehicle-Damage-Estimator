import streamlit as st
import os
from cost_estimator import DamageEstimator

def main():
    st.set_page_config(page_title="Vehicle Damage Estimator", page_icon="🚗")
    
    st.title("🚗 Auto AI: Vehicle Damage Estimator")
    st.write("Upload an image of a damaged vehicle to receive an instant repair cost estimation based on our instance segmentation heuristics.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        # Display image
        st.image(uploaded_file, caption='Uploaded Image', width='stretch')
        
        st.write("---")
        st.subheader("Model Inference")
        
        with st.spinner("Analyzing damage via segmentation masks..."):
            from ultralytics import YOLO
            from PIL import Image
            
            try:
                # Load pretrained model since custom isn't fully trained yet
                model = YOLO("yolov8n-seg.pt")
                image = Image.open(uploaded_file)
                results = model(image)
                
                result = results[0]
                damaged_pixels = 0
                detected_part = "None"
                
                if result.masks is not None and len(result.masks) > 0:
                    # Take the first detected object's mask
                    mask = result.masks.data[0]
                    damaged_pixels = int(mask.sum().item())
                    cls = int(result.boxes.cls[0].item())
                    detected_part = model.names[cls]
            except Exception as e:
                st.error(f"Error during YOLO inference: {e}")
                return
            
            estimator = DamageEstimator()
            
            if damaged_pixels > 0:
                report = estimator.get_estimate(detected_part, damaged_pixels)
                
                # Display Report
                if report:
                    st.success("Analysis Complete!")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Detected Part", report["Detected Part"])
                    with col2:
                        st.metric("Severity", report["Damage Severity"])
                    with col3:
                        st.metric("Estimated Cost", report["Estimated Repair Cost"])
                        
                    st.info(f"Technical Info: Detected '{detected_part}' Area refers to {damaged_pixels} pixels via Instance Segmentation MASK.")
            else:
                st.warning("No significant objects or damage masks detected in the image.")

if __name__ == "__main__":
    main()
