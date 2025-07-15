# Welding Image Analysis App

A Streamlit application for analyzing welding images using AI/ML APIs. Upload welding images, get detailed analysis results, and view comprehensive reports.

## Features

- 📤 **Image Upload**: Support for multiple image formats (JPG, PNG, BMP, TIFF)
- 🔍 **AI Analysis**: Calls external APIs for welding defect detection
- 📊 **Detailed Results**: Shows analysis summary, defects, measurements, and recommendations
- 📈 **Metrics Visualization**: Interactive charts and metrics display
- 🛠️ **Configurable**: Easy API endpoint and key configuration
- 🎨 **Modern UI**: Clean and responsive interface

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Settings

Edit `config.py` or use the sidebar in the app to set:
- API endpoint URL
- API key (if required)
- Other configuration options

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. **Upload Image**: Click "Choose a welding image..." and select your image file
2. **Configure API**: Enter your API endpoint URL in the sidebar
3. **Add API Key**: If your API requires authentication, enter the key
4. **Analyze**: Click "Analyze Image" to process the image
5. **View Results**: Check the results in the organized tabs:
   - **Summary**: Quick overview with key metrics
   - **Detailed Analysis**: Defects, measurements, and recommendations
   - **Metrics**: Performance metrics and visualizations
   - **Raw Data**: Complete API response

## API Integration

The app expects your API to:
- Accept POST requests with JSON payload
- Include base64-encoded image data
- Return JSON response with analysis results

### Example API Request Format:
```json
{
  "image": "base64_encoded_image_data",
  "timestamp": "2025-07-15T10:30:00"
}
```

### Expected Response Fields:
- `quality_score`: Overall quality score (0-100)
- `defect_count`: Number of defects found
- `confidence`: Analysis confidence level
- `status`: Overall status (pass/fail/warning)
- `defects`: Array of detected defects
- `measurements`: Object with measurement data
- `recommendations`: Array of improvement suggestions

## Customization

- **Styling**: Modify the CSS in `app.py` for custom appearance
- **Result Display**: Customize `display_welding_results()` function
- **API Integration**: Modify `call_welding_api()` for different API formats
- **Configuration**: Update `config.py` for default settings

## Troubleshooting

- **API Connection Issues**: Check endpoint URL and network connectivity
- **Authentication Errors**: Verify API key and authorization method
- **Image Upload Problems**: Ensure image format is supported and file size is reasonable
- **Display Issues**: Check that your API returns expected JSON structure

## Requirements

- Python 3.7+
- Streamlit
- PIL (Pillow)
- Requests
- Pandas
- NumPy

## License

MIT License - feel free to modify and use as needed.
