# Handwritten Equation Solver

A modern, AI-powered application that reads handwritten mathematical equations from images and solves them automatically. Built with deep learning, computer vision, and symbolic computation.

## Features

- **Deep Learning OCR**: Uses Microsoft's TrOCR model for accurate handwritten text recognition
- **Modern Web Interface**: Beautiful Streamlit-based web application
- **Database Storage**: SQLite database to store equation history and solutions
- **Advanced Image Processing**: Preprocessing pipeline for optimal OCR performance
- **Symbolic Computation**: Powered by SymPy for accurate mathematical solving
- **Equation History**: Track and review previously solved equations
- **Fallback Support**: Automatic fallback to Tesseract OCR if deep learning fails

## Quick Start

### Prerequisites

- Python 3.8+
- Tesseract OCR installed on your system

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kryptologyst/handwritten-equation-solver.git
   cd handwritten-equation-solver
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**
   
   **On macOS:**
   ```bash
   brew install tesseract
   ```
   
   **On Ubuntu/Debian:**
   ```bash
   sudo apt-get install tesseract-ocr
   ```
   
   **On Windows:**
   Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

4. **Run the application**
   ```bash
   streamlit run modern_solver.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## Usage

### Web Interface

1. **Upload an Image**: Click "Choose an image file" and select an image containing a handwritten equation
2. **Process**: Click "🔍 Solve Equation" to analyze and solve the equation
3. **View Results**: See the extracted text, parsed equation, and solution
4. **History**: Review previously solved equations in the sidebar

### Supported Equation Types

- **Linear Equations**: `2x + 3 = 7`
- **Quadratic Equations**: `x² - 4 = 0`
- **Polynomial Equations**: `x³ - 8 = 0`
- **Multi-variable Equations**: `2x + 3y = 12`
- **Arithmetic Expressions**: `2 + 3 * 4`

### Example Equations

Try uploading images with these handwritten equations:

```
2x + 3 = 7
x² - 4 = 0
3y + 2 = 11
2x + 3y = 12
x³ - 8 = 0
```

## Architecture

### Components

1. **Image Preprocessing**: OpenCV-based image enhancement
2. **Text Recognition**: TrOCR deep learning model with Tesseract fallback
3. **Equation Parsing**: SymPy-based mathematical expression parsing
4. **Equation Solving**: Symbolic computation engine
5. **Database Storage**: SQLite for persistence
6. **Web Interface**: Streamlit-based user interface

### Technology Stack

- **Deep Learning**: PyTorch, Transformers (TrOCR)
- **Computer Vision**: OpenCV, PIL
- **Mathematical Computation**: SymPy
- **Web Framework**: Streamlit
- **Database**: SQLite
- **OCR**: Tesseract (fallback)

## Testing

Run the test suite:

```bash
pytest test_solver.py -v
```

### Test Coverage

- Unit tests for all core functions
- Integration tests for the complete pipeline
- Database operations testing
- Image processing validation
- SymPy integration testing

## 📁 Project Structure

```
handwritten-equation-solver/
├── modern_solver.py          # Main application with web interface
├── 0117.py                   # Original implementation
├── test_solver.py            # Comprehensive test suite
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── equations.db             # SQLite database (created on first run)
└── sample_images/           # Example equation images
```

## 🔧 Configuration

### Model Configuration

The application automatically downloads the TrOCR model on first run. You can modify the model in the `load_model()` method:

```python
model_name = "microsoft/trocr-base-handwritten"  # Change this
```

### Database Configuration

The SQLite database is automatically created. To reset:

```bash
rm equations.db
```

### OCR Configuration

Tesseract configuration can be modified in the `extract_text_tesseract()` method:

```python
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/=()xXyYzZ'
```

## Deployment

### Local Development

```bash
streamlit run modern_solver.py
```

### Production Deployment

For production deployment, consider:

1. **Docker**: Create a Dockerfile for containerized deployment
2. **Cloud Platforms**: Deploy on AWS, GCP, or Azure
3. **Streamlit Cloud**: Use Streamlit's cloud hosting service

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y tesseract-ocr

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "modern_solver.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
pytest

# Format code
black .

# Lint code
flake8 .

# Type checking
mypy modern_solver.py
```

## Performance

### Benchmarks

- **Image Processing**: ~100ms for typical equation images
- **OCR Recognition**: ~500ms with TrOCR, ~200ms with Tesseract
- **Equation Solving**: ~10ms for simple equations
- **Total Pipeline**: ~1-2 seconds end-to-end

### Optimization Tips

1. **Image Quality**: Use high-contrast, well-lit images
2. **Handwriting**: Clear, separated characters work best
3. **Image Size**: 400x200px minimum recommended
4. **Format**: PNG or JPG formats supported

## Troubleshooting

### Common Issues

1. **Tesseract not found**
   ```bash
   # Install Tesseract
   brew install tesseract  # macOS
   sudo apt-get install tesseract-ocr  # Ubuntu
   ```

2. **Model download fails**
   ```bash
   # Check internet connection
   # Try running with --offline flag
   ```

3. **Poor OCR accuracy**
   - Ensure good image quality
   - Try different preprocessing settings
   - Check handwriting clarity

4. **Equation parsing errors**
   - Verify equation format
   - Check for special characters
   - Ensure proper mathematical notation

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Microsoft TrOCR](https://github.com/microsoft/TrOCR) for handwritten text recognition
- [SymPy](https://www.sympy.org/) for symbolic mathematics
- [Streamlit](https://streamlit.io/) for the web interface
- [OpenCV](https://opencv.org/) for computer vision
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for fallback OCR


# Handwritten-Equation-Solver
