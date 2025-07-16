import streamlit as st
import requests
import json
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import pandas as pd
from datetime import datetime
import numpy as np

# Configure page
st.set_page_config(
    page_title="Welding Image Analysis",
    page_icon="🔧",
    layout="wide"
)

# Roboflow API Configuration
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_API_KEY = "lSiEtOHDatuVU8fZN0iS"
MODEL_ID = "weld-4qosh/1"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def draw_detections_on_image(image, predictions, confidence_threshold=0.0):
    """
    Draw bounding boxes and labels on the image
    """
    # Create a copy of the image to draw on
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Try to use a better font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        except:
            font = None
            small_font = None
    
    # Color palette for different classes
    colors = [
        "#FF0000",  # Red
        "#00FF00",  # Green
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
        "#FF00FF",  # Magenta
        "#00FFFF",  # Cyan
        "#FFA500",  # Orange
        "#800080",  # Purple
        "#FFC0CB",  # Pink
        "#A52A2A",  # Brown
    ]
    
    # Keep track of class colors
    class_colors = {}
    color_index = 0
    
    for pred in predictions:
        confidence = pred.get("confidence", 0)
        
        # Only draw boxes above confidence threshold
        if confidence >= confidence_threshold:
            x_center = pred.get("x", 0)
            y_center = pred.get("y", 0)
            width = pred.get("width", 0)
            height = pred.get("height", 0)
            class_name = pred.get("class", "unknown")
            
            # Calculate bounding box coordinates
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Assign color to class
            if class_name not in class_colors:
                class_colors[class_name] = colors[color_index % len(colors)]
                color_index += 1
            
            box_color = class_colors[class_name]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
            
            # Prepare label text
            label = f"{class_name}: {confidence:.2%}"
            
            # Get text size for background rectangle
            if font:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(label) * 6  # Approximate width
                text_height = 14  # Approximate height
            
            # Draw label background
            label_bg = [x1, y1 - text_height - 5, x1 + text_width + 10, y1]
            draw.rectangle(label_bg, fill=box_color)
            
            # Draw label text
            draw.text((x1 + 5, y1 - text_height - 2), label, fill="white", font=font)
    
    return img_with_boxes, class_colors

def call_roboflow_api(image):
    """
    Call the Roboflow welding detection API using requests
    """
    try:
        # Convert PIL Image to base64 string
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare the API request
        url = f"{ROBOFLOW_API_URL}/{MODEL_ID}"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        # Send request to Roboflow API
        response = requests.post(
            url,
            data=image_base64,
            headers=headers,
            params={"api_key": ROBOFLOW_API_KEY}
        )
        
        if response.status_code == 200:
            result = response.json()
            return True, result
        else:
            return False, f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return False, f"Roboflow API Error: {str(e)}"

def process_roboflow_results(raw_result):
    """
    Process Roboflow API results into a more readable format
    """
    processed = {
        "model_id": raw_result.get("model_id", MODEL_ID),
        "time": raw_result.get("time", 0),
        "image_info": {
            "width": raw_result.get("image", {}).get("width", 0),
            "height": raw_result.get("image", {}).get("height", 0)
        },
        "predictions": raw_result.get("predictions", []),
        "detection_count": len(raw_result.get("predictions", [])),
        "confidence_threshold": 0.5
    }
    
    # Process predictions
    if processed["predictions"]:
        classes_detected = list(set([pred.get("class", "unknown") for pred in processed["predictions"]]))
        processed["classes_detected"] = classes_detected
        processed["class_counts"] = {cls: sum(1 for pred in processed["predictions"] if pred.get("class") == cls) for cls in classes_detected}
        
        # Calculate average confidence
        confidences = [pred.get("confidence", 0) for pred in processed["predictions"]]
        processed["average_confidence"] = sum(confidences) / len(confidences) if confidences else 0
        
        # Get highest confidence detection
        if confidences:
            max_conf_idx = confidences.index(max(confidences))
            processed["highest_confidence_detection"] = processed["predictions"][max_conf_idx]
    
    return processed

def display_welding_results(results):
    """
    Display Roboflow welding detection results
    """
    if not results:
        st.error("No results to display")
        return
    
    # Create tabs for different sections
    tabs = st.tabs(["📊 Summary", "�️ Annotated Image", "�🔍 Detections", "📋 Raw Data"])
    
    with tabs[0]:  # Summary
        st.subheader("🔍 Detection Summary")
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Detections", results.get("detection_count", 0))
        with col2:
            st.metric("Average Confidence", f"{results.get('average_confidence', 0):.2%}")
        with col3:
            st.metric("Processing Time", f"{results.get('time', 0):.3f}s")
        
        # Show detected classes
        if "classes_detected" in results:
            st.write("### Detected Classes")
            for cls in results["classes_detected"]:
                count = results.get("class_counts", {}).get(cls, 0)
                st.write(f"- **{cls}**: {count} detection(s)")
        
        # Show image info
        if "image_info" in results:
            img_info = results["image_info"]
            st.write(f"**Image Size**: {img_info.get('width', 0)} x {img_info.get('height', 0)} pixels")
        
        # Show model info
        st.write(f"**Model**: {results.get('model_id', 'N/A')}")
    
    with tabs[1]:  # Annotated Image
        st.subheader("🖼️ Detection Visualization")
        
        # Add confidence threshold slider
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Only show detections above this confidence level"
        )
        
        # Get the original image from session state
        if "analyzed_image" in st.session_state:
            original_image = st.session_state.analyzed_image
            predictions = results.get("predictions", [])
            
            if predictions:
                # Draw bounding boxes on the image
                annotated_image, class_colors = draw_detections_on_image(
                    original_image, predictions, confidence_threshold
                )
                
                # # Display images side by side
                # col1, col2 = st.columns(2)
                
                # with col1:
                #     st.write("**Original Image**")
                #     st.image(original_image, caption="Original", use_container_width=True)
                
                
                st.write("**Annotated Image**")
                st.image(annotated_image, caption="With Detections", use_container_width=True)
                
                # Show color legend
                if class_colors:
                    st.write("### Detection Legend")
                    legend_cols = st.columns(len(class_colors))
                    for i, (class_name, color) in enumerate(class_colors.items()):
                        with legend_cols[i]:
                            st.markdown(f"""
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <div style="width: 20px; height: 20px; background-color: {color}; margin-right: 10px; border: 1px solid #000;"></div>
                                <span>{class_name}</span>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Download button for annotated image
                buffered = io.BytesIO()
                annotated_image.save(buffered, format="PNG")
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=buffered.getvalue(),
                    file_name=f"welding_detection_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
                
            else:
                st.info("No detections found in the image")
                st.image(original_image, caption="Original Image", use_container_width=True)
        else:
            st.warning("Original image not found in session state")
    
    with tabs[2]:  # Detections
        st.subheader("🔍 Detection Details")
        
        predictions = results.get("predictions", [])
        if predictions:
            # Create DataFrame for detections
            detection_data = []
            for i, pred in enumerate(predictions):
                detection_data.append({
                    "Detection": i + 1,
                    "Class": pred.get("class", "unknown"),
                    "Confidence": f"{pred.get('confidence', 0):.2%}",
                    "X": pred.get("x", 0),
                    "Y": pred.get("y", 0),
                    "Width": pred.get("width", 0),
                    "Height": pred.get("height", 0)
                })
            
            df = pd.DataFrame(detection_data)
            st.dataframe(df, use_container_width=True)
            
            # Show individual detection details
            for i, pred in enumerate(predictions):
                with st.expander(f"Detection {i+1}: {pred.get('class', 'unknown')} ({pred.get('confidence', 0):.2%})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Class**: {pred.get('class', 'unknown')}")
                        st.write(f"**Confidence**: {pred.get('confidence', 0):.2%}")
                        st.write(f"**Class ID**: {pred.get('class_id', 'N/A')}")
                    with col2:
                        st.write(f"**X**: {pred.get('x', 0)}")
                        st.write(f"**Y**: {pred.get('y', 0)}")
                        st.write(f"**Width**: {pred.get('width', 0)}")
                        st.write(f"**Height**: {pred.get('height', 0)}")
                        
                    # Show detection box coordinates
                    if all(key in pred for key in ['x', 'y', 'width', 'height']):
                        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                        st.write(f"**Bounding Box**: ({x-w/2:.0f}, {y-h/2:.0f}) to ({x+w/2:.0f}, {y+h/2:.0f})")
        else:
            st.info("No detections found in the image")
    
    with tabs[3]:  # Raw Data
        st.subheader("📋 Raw Detection Data")
        
        # Show formatted JSON
        st.json(results)
        
        # Show raw text for debugging
        with st.expander("Raw Response Text"):
            st.code(json.dumps(results, indent=2), language="json")
        
        # Download button for results
        result_json = json.dumps(results, indent=2)
        st.download_button(
            label="📥 Download Results as JSON",
            data=result_json,
            file_name=f"welding_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    # Main header
    st.markdown('<h1 class="main-header">🔧 Welding Image Analysis</h1>', unsafe_allow_html=True)
    
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Welding Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a welding image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload a clear image of the welding area for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.write(f"**File name:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size:,} bytes")
            st.write(f"**Image dimensions:** {image.size[0]} x {image.size[1]} pixels")
            
            # Analysis button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image with Roboflow AI... This may take a few moments."):
                    try:
                        # Call Roboflow API directly with PIL Image
                        success, result = call_roboflow_api(image)
                        
                        if success:
                            # Process and display results
                            processed_result = process_roboflow_results(result)
                            st.success("✅ Analysis completed successfully!")
                            
                            # Store results in session state
                            st.session_state.analysis_results = processed_result
                            st.session_state.analyzed_image = image
                        else:
                            st.error(f"❌ Analysis failed: {result}")
                            
                            # Add helpful debugging info
                            with st.expander("🔧 Debugging Information"):
                                st.write("**API**: Roboflow Inference API")
                                st.write("**Model**: weld-4qosh/1")
                                st.write("**Image Size:**", f"{image.size[0]}x{image.size[1]}")
                                st.write("**File Size:**", f"{uploaded_file.size:,} bytes")
                                st.write("**Suggested Solutions:**")
                                st.write("1. Check your internet connection")
                                st.write("2. Verify the Roboflow API is accessible")
                                st.write("3. Ensure the image is clear and contains welding content")
                                st.write("4. Try with a different image format")
                                
                    except Exception as e:
                        st.error(f"❌ Error occurred: {str(e)}")
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if "analysis_results" in st.session_state:
            display_welding_results(st.session_state.analysis_results)
        else:
            st.info("👆 Upload an image and click 'Analyze' to see results here")
        
    

if __name__ == "__main__":
    main()
