# Modern Handwritten Equation Solver
# Enhanced with deep learning, web interface, and database support

import os
import json
import sqlite3
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import logging
from typing import List, Dict, Optional, Tuple
import base64
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HandwrittenEquationSolver:
    """Modern handwritten equation solver using deep learning and symbolic computation."""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.db_path = "equations.db"
        self.init_database()
        self.load_model()
    
    def init_database(self):
        """Initialize SQLite database for storing equations and solutions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS equations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                extracted_text TEXT,
                equation TEXT,
                solution TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def load_model(self):
        """Load TrOCR model for handwritten text recognition."""
        try:
            # Use TrOCR model for handwritten text recognition
            model_name = "microsoft/trocr-base-handwritten"
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            logger.info("TrOCR model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load TrOCR model: {e}")
            logger.info("Falling back to Tesseract OCR")
            self.model = None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR performance."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text_deep_learning(self, image: np.ndarray) -> str:
        """Extract text using TrOCR deep learning model."""
        if self.model is None:
            return self.extract_text_tesseract(image)
        
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Process image
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
            
            # Generate text
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Deep learning OCR failed: {e}")
            return self.extract_text_tesseract(image)
    
    def extract_text_tesseract(self, image: np.ndarray) -> str:
        """Fallback OCR using Tesseract."""
        try:
            import pytesseract
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/=()xXyYzZ'
            text = pytesseract.image_to_string(image, config=custom_config)
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""
    
    def clean_equation(self, text: str) -> str:
        """Clean and normalize extracted equation text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', '', text)
        
        # Common OCR corrections
        corrections = {
            'O': '0',  # Letter O to zero
            'l': '1',  # Letter l to one
            'I': '1',  # Letter I to one
            'S': '5',  # Letter S to five
            'B': '8',  # Letter B to eight
        }
        
        for old, new in corrections.items():
            text = text.replace(old, new)
        
        return text
    
    def parse_equation(self, equation_text: str) -> Optional[sp.Eq]:
        """Parse equation text into SymPy expression."""
        try:
            # Clean the equation
            cleaned = self.clean_equation(equation_text)
            
            # Handle different equation formats
            if '=' in cleaned:
                lhs, rhs = cleaned.split('=', 1)
                lhs_expr = parse_expr(lhs.strip())
                rhs_expr = parse_expr(rhs.strip())
                return sp.Eq(lhs_expr, rhs_expr)
            else:
                # Assume it's an expression to solve for zero
                expr = parse_expr(cleaned)
                return sp.Eq(expr, 0)
        
        except Exception as e:
            logger.error(f"Failed to parse equation '{equation_text}': {e}")
            return None
    
    def solve_equation(self, equation: sp.Eq) -> Dict:
        """Solve the equation and return results."""
        try:
            # Get variables in the equation
            variables = list(equation.free_symbols)
            
            if not variables:
                # No variables, just evaluate
                result = equation.lhs - equation.rhs
                return {
                    'type': 'evaluation',
                    'result': float(result.evalf()),
                    'variables': []
                }
            
            # Solve for each variable
            solutions = {}
            for var in variables:
                try:
                    sol = sp.solve(equation, var)
                    if sol:
                        solutions[str(var)] = [float(s.evalf()) if s.is_number else str(s) for s in sol]
                except Exception as e:
                    logger.warning(f"Could not solve for {var}: {e}")
            
            return {
                'type': 'solution',
                'solutions': solutions,
                'variables': [str(v) for v in variables]
            }
        
        except Exception as e:
            logger.error(f"Failed to solve equation: {e}")
            return {'type': 'error', 'error': str(e)}
    
    def save_to_database(self, image_path: str, extracted_text: str, 
                        equation: str, solution: Dict, confidence: float = 0.0):
        """Save equation and solution to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO equations (image_path, extracted_text, equation, solution, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (image_path, extracted_text, equation, json.dumps(solution), confidence))
        
        conn.commit()
        conn.close()
    
    def get_equation_history(self, limit: int = 10) -> List[Dict]:
        """Retrieve equation history from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, image_path, extracted_text, equation, solution, confidence, created_at
            FROM equations
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'image_path': row[1],
                'extracted_text': row[2],
                'equation': row[3],
                'solution': json.loads(row[4]),
                'confidence': row[5],
                'created_at': row[6]
            })
        
        conn.close()
        return results
    
    def process_equation(self, image: np.ndarray) -> Dict:
        """Complete pipeline: preprocess, extract text, parse, and solve."""
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Extract text
        extracted_text = self.extract_text_deep_learning(processed_image)
        
        if not extracted_text:
            return {
                'success': False,
                'error': 'No text could be extracted from the image'
            }
        
        # Parse equation
        equation_obj = self.parse_equation(extracted_text)
        
        if equation_obj is None:
            return {
                'success': False,
                'error': 'Could not parse the extracted text as a valid equation',
                'extracted_text': extracted_text
            }
        
        # Solve equation
        solution = self.solve_equation(equation_obj)
        
        return {
            'success': True,
            'extracted_text': extracted_text,
            'equation': str(equation_obj),
            'solution': solution
        }

# Streamlit Web Interface
def create_web_interface():
    """Create modern web interface using Streamlit."""
    st.set_page_config(
        page_title="Handwritten Equation Solver",
        page_icon="🧮",
        layout="wide"
    )
    
    st.title("🧮 Handwritten Equation Solver")
    st.markdown("Upload an image of a handwritten equation and get the solution!")
    
    # Initialize solver
    if 'solver' not in st.session_state:
        with st.spinner("Loading AI model..."):
            st.session_state.solver = HandwrittenEquationSolver()
    
    solver = st.session_state.solver
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        show_history = st.checkbox("Show equation history", value=True)
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image containing a handwritten mathematical equation"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("🔍 Solve Equation", type="primary"):
                with st.spinner("Processing image..."):
                    # Convert PIL to numpy array
                    image_array = np.array(image)
                    
                    # Process the image
                    result = solver.process_equation(image_array)
                    
                    # Display results
                    if result['success']:
                        st.success("✅ Equation solved successfully!")
                        
                        st.subheader("📝 Extracted Text")
                        st.code(result['extracted_text'])
                        
                        st.subheader("🧮 Parsed Equation")
                        st.code(result['equation'])
                        
                        st.subheader("💡 Solution")
                        solution = result['solution']
                        
                        if solution['type'] == 'solution':
                            for var, values in solution['solutions'].items():
                                if len(values) == 1:
                                    st.write(f"**{var} = {values[0]}**")
                                else:
                                    st.write(f"**{var}**: {', '.join(map(str, values))}")
                        
                        elif solution['type'] == 'evaluation':
                            st.write(f"**Result: {solution['result']}**")
                        
                        elif solution['type'] == 'error':
                            st.error(f"Error: {solution['error']}")
                        
                        # Save to database
                        solver.save_to_database(
                            uploaded_file.name,
                            result['extracted_text'],
                            result['equation'],
                            solution
                        )
                    
                    else:
                        st.error(f"❌ {result['error']}")
                        if 'extracted_text' in result:
                            st.write("**Extracted text:**", result['extracted_text'])
    
    with col2:
        st.header("📊 Results & History")
        
        if show_history:
            st.subheader("📚 Recent Equations")
            history = solver.get_equation_history(5)
            
            if history:
                for eq in history:
                    with st.expander(f"Equation {eq['id']} - {eq['created_at'][:19]}"):
                        st.write("**Extracted:**", eq['extracted_text'])
                        st.write("**Equation:**", eq['equation'])
                        st.write("**Solution:**", json.dumps(eq['solution'], indent=2))
            else:
                st.info("No equations processed yet.")
        
        # Example equations
        st.subheader("💡 Example Equations")
        examples = [
            "2x + 3 = 7",
            "x² - 4 = 0",
            "3y + 2 = 11",
            "2x + 3y = 12",
            "x³ - 8 = 0"
        ]
        
        for example in examples:
            st.code(example)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, TrOCR, and SymPy | "
        "[GitHub Repository](https://github.com/yourusername/handwritten-equation-solver)"
    )

if __name__ == "__main__":
    create_web_interface()
