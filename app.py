import streamlit as st
import requests
import json
from PIL import Image
import io
import base64
import pandas as pd
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Welding Image Analysis",
    page_icon="🔧",
    layout="wide"
)

# API Configuration - Fixed endpoint
API_ENDPOINT = "http://127.0.0.1:8001/predict"

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

def call_welding_api_multipart(image, api_url, api_key=None):
    """
    Call the welding analysis API using multipart/form-data
    """
    try:
        # Prepare headers
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Convert image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        
        # Prepare files for multipart upload
        files = {
            'image': ('image.jpg', image_bytes, 'image/jpeg'),
        }
        
        # Make API call
        response = requests.post(
            api_url,
            headers=headers,
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Multipart API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.RequestException as e:
        return False, f"Multipart Request Error: {str(e)}"
    except Exception as e:
        return False, f"Multipart Unexpected Error: {str(e)}"

def call_welding_api(image_data, api_url, api_key=None):
    """
    Call the welding analysis API with multiple format support
    """
    try:
        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Try different payload formats based on common API patterns
        payloads_to_try = [
            # Format 1: Direct image field
            {
                "image": image_data
            },
            # Format 2: Base64 with data URL prefix
            {
                "image": f"data:image/jpeg;base64,{image_data}"
            },
            # Format 3: Nested structure
            {
                "data": {
                    "image": image_data
                }
            },
            # Format 4: File-like structure
            {
                "image": image_data,
                "timestamp": datetime.now().isoformat(),
                "format": "base64"
            }
        ]
        
        # Try each payload format
        for i, payload in enumerate(payloads_to_try):
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return True, response.json()
                elif response.status_code == 422 and i < len(payloads_to_try) - 1:
                    # Try next format for 422 errors
                    continue
                else:
                    # For other errors or last attempt, return the error
                    return False, f"API Error: {response.status_code} - {response.text}"
                    
            except requests.exceptions.RequestException as e:
                if i == len(payloads_to_try) - 1:  # Last attempt
                    return False, f"Request Error: {str(e)}"
                continue
        
        return False, "All payload formats failed"
            
    except Exception as e:
        return False, f"Unexpected Error: {str(e)}"

def display_welding_results(results):
    """
    Display welding analysis results in a structured format
    """
    if not results:
        st.error("No results to display")
        return
    
    # Create tabs for different sections
    # tabs = st.tabs(["📊 Summary", "�️ Images", "�🔍 Detailed Analysis", "📈 Metrics", "📋 Raw Data"])
    tabs = st.tabs(["�️ Images", "📋 Raw Data"])

    
    # with tabs[0]:  # Summary
    #     st.subheader("🔍 Analysis Summary")
        
    #     # Check for common result fields
    #     if "quality_score" in results:
    #         col1, col2, col3 = st.columns(3)
    #         with col1:
    #             st.metric("Quality Score", f"{results['quality_score']:.2f}")
    #         with col2:
    #             if "defect_count" in results:
    #                 st.metric("Defects Found", results["defect_count"])
    #         with col3:
    #             if "confidence" in results:
    #                 st.metric("Confidence", f"{results['confidence']:.1%}")
        
    #     # Overall status
    #     if "status" in results:
    #         status = results["status"].lower()
    #         if status == "pass" or status == "good":
    #             st.success(f"✅ Status: {results['status']}")
    #         elif status == "fail" or status == "poor":
    #             st.error(f"❌ Status: {results['status']}")
    #         else:
    #             st.warning(f"⚠️ Status: {results['status']}")
        
    #     # Additional summary fields
    #     summary_fields = ["analysis_time", "model_version", "timestamp", "processing_time"]
    #     for field in summary_fields:
    #         if field in results:
    #             st.write(f"**{field.replace('_', ' ').title()}**: {results[field]}")
    
    with tabs[0]:  # Images
        st.subheader("🖼️ Analysis Images")
        
        # Display original image if available
        # if "analyzed_image" in st.session_state:
        #     st.write("### Original Image")
        #     st.image(st.session_state.analyzed_image, caption="Original Uploaded Image", use_column_width=True)
        
        # Display base64 images from API response
        image_fields = ["result_image", "processed_image", "annotated_image", "output_image", "image", "base64_image"]
        
        for field in image_fields:
            if field in results:
                try:
                    image_data = results[field]
                    # Handle different base64 formats
                    if isinstance(image_data, str):
                        # Remove data URL prefix if present
                        if image_data.startswith('data:image'):
                            image_data = image_data.split(',')[1]
                        
                        # Decode base64 image
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(io.BytesIO(image_bytes))
                        
                        st.write(f"### {field.replace('_', ' ').title()}")
                        st.image(image, caption=f"API Response - {field}", use_column_width=True)
                        
                        # Show image info
                        st.write(f"**Dimensions**: {image.size[0]} x {image.size[1]} pixels")
                        
                except Exception as e:
                    st.warning(f"Could not display {field}: {str(e)}")
        
        # Display image arrays or other image data
        if "images" in results and isinstance(results["images"], list):
            st.write("### Additional Images")
            for i, img_data in enumerate(results["images"]):
                try:
                    if isinstance(img_data, str):
                        if img_data.startswith('data:image'):
                            img_data = img_data.split(',')[1]
                        image_bytes = base64.b64decode(img_data)
                        image = Image.open(io.BytesIO(image_bytes))
                        st.image(image, caption=f"Image {i+1}", use_column_width=True)
                except Exception as e:
                    st.warning(f"Could not display image {i+1}: {str(e)}")
    
    # with tabs[2]:  # Detailed Analysis
    #     st.subheader("🔍 Detailed Analysis")
        
    #     # Defects section
    #     if "defects" in results and results["defects"]:
    #         st.write("### Detected Defects")
    #         defects_df = pd.DataFrame(results["defects"])
    #         st.dataframe(defects_df, use_container_width=True)
            
    #         # Show defect details
    #         for i, defect in enumerate(results["defects"]):
    #             with st.expander(f"Defect {i+1}: {defect.get('type', 'Unknown')}"):
    #                 for key, value in defect.items():
    #                     st.write(f"**{key.replace('_', ' ').title()}**: {value}")
        
    #     # Measurements section
    #     if "measurements" in results:
    #         st.write("### Measurements")
    #         measurements = results["measurements"]
    #         if isinstance(measurements, dict):
    #             col1, col2 = st.columns(2)
    #             items = list(measurements.items())
    #             mid = len(items) // 2
                
    #             with col1:
    #                 for key, value in items[:mid]:
    #                     st.metric(key.replace('_', ' ').title(), value)
                
    #             with col2:
    #                 for key, value in items[mid:]:
    #                     st.metric(key.replace('_', ' ').title(), value)
    #         else:
    #             st.write(measurements)
        
    #     # Recommendations
    #     if "recommendations" in results:
    #         st.write("### Recommendations")
    #         recommendations = results["recommendations"]
    #         if isinstance(recommendations, list):
    #             for i, rec in enumerate(recommendations, 1):
    #                 st.write(f"**{i}.** {rec}")
    #         else:
    #             st.write(recommendations)
        
    #     # Additional analysis fields
    #     analysis_fields = ["weld_quality", "structural_integrity", "material_properties", "heat_treatment"]
    #     for field in analysis_fields:
    #         if field in results:
    #             st.write(f"### {field.replace('_', ' ').title()}")
    #             if isinstance(results[field], dict):
    #                 for key, value in results[field].items():
    #                     st.write(f"**{key.replace('_', ' ').title()}**: {value}")
    #             else:
    #                 st.write(results[field])
    
    # with tabs[3]:  # Metrics
    #     st.subheader("📈 Performance Metrics")
        
    #     # Create metrics visualization
    #     metrics_data = []
    #     for key, value in results.items():
    #         if isinstance(value, (int, float)) and not isinstance(value, bool):
    #             metrics_data.append({"Metric": key.replace('_', ' ').title(), "Value": value})
        
    #     if metrics_data:
    #         metrics_df = pd.DataFrame(metrics_data)
    #         st.dataframe(metrics_df, use_container_width=True)
            
    #         # Bar chart for numeric values
    #         numeric_metrics = [m for m in metrics_data if isinstance(m["Value"], (int, float)) and m["Value"] <= 100]
    #         if numeric_metrics:
    #             chart_df = pd.DataFrame(numeric_metrics)
    #             st.bar_chart(chart_df.set_index("Metric"))
        
    #     # Score breakdown if available
    #     if "scores" in results:
    #         st.write("### Score Breakdown")
    #         scores = results["scores"]
    #         if isinstance(scores, dict):
    #             for category, score in scores.items():
    #                 st.progress(score / 100 if score > 1 else score, text=f"{category.replace('_', ' ').title()}: {score}")
    
    with tabs[1]:  # Raw Data
        st.subheader("📋 Raw API Response")
        
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
            file_name=f"welding_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    # Main header
    st.markdown('<h1 class="main-header">🔧 Welding Image Analysis</h1>', unsafe_allow_html=True)
    
    # Show API endpoint
    st.info(f"🔗 API Endpoint: {API_ENDPOINT}")
    
    
    
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
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.write(f"**File name:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size:,} bytes")
            st.write(f"**Image dimensions:** {image.size[0]} x {image.size[1]} pixels")
            
            # Analysis button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image... This may take a few moments."):
                    # Try JSON format first
                    image_base64 = encode_image_to_base64(image)
                    success, result = call_welding_api(image_base64, API_ENDPOINT, None)
                    
                    # If JSON fails, try multipart/form-data
                    if not success and "422" in str(result):
                        st.info("Trying alternative upload format...")
                        success, result = call_welding_api_multipart(image, API_ENDPOINT, None)
                    
                    if success:
                        st.success("✅ Analysis completed successfully!")
                        
                        # Store results in session state
                        st.session_state.analysis_results = result
                        st.session_state.analyzed_image = image
                    else:
                        st.error(f"❌ Analysis failed: {result}")
                        
                        # Add helpful debugging info
                        with st.expander("🔧 Debugging Information"):
                            st.write("**API URL:**", API_ENDPOINT)
                            st.write("**Image Size:**", f"{image.size[0]}x{image.size[1]}")
                            st.write("**File Size:**", f"{uploaded_file.size:,} bytes")
                            st.write("**Suggested Solutions:**")
                            st.write("1. Check if your API endpoint is correct")
                            st.write("2. Verify the API is running and accessible")
                            st.write("3. Check if API requires authentication")
                            st.write("4. Ensure the API accepts image data in the expected format")
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if "analysis_results" in st.session_state:
            display_welding_results(st.session_state.analysis_results)
        else:
            st.info("👆 Upload an image and click 'Analyze' to see results here")
    

if __name__ == "__main__":
    main()
