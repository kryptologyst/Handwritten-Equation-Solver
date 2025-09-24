"""
Test suite for Handwritten Equation Solver
"""

import pytest
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
from modern_solver import HandwrittenEquationSolver
import sympy as sp

class TestHandwrittenEquationSolver:
    """Test cases for the HandwrittenEquationSolver class."""
    
    @pytest.fixture
    def solver(self):
        """Create a solver instance for testing."""
        return HandwrittenEquationSolver()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample handwritten equation image."""
        # Create a simple white image with black text
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        
        # Add some text using OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '2x + 3 = 7', (50, 100), font, 1, (0, 0, 0), 2)
        
        return img
    
    def test_database_initialization(self, solver):
        """Test that database is initialized correctly."""
        assert os.path.exists(solver.db_path)
        
        # Test database connection
        import sqlite3
        conn = sqlite3.connect(solver.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='equations'")
        assert cursor.fetchone() is not None
        conn.close()
    
    def test_preprocess_image(self, solver, sample_image):
        """Test image preprocessing."""
        processed = solver.preprocess_image(sample_image)
        
        assert processed.shape[:2] == sample_image.shape[:2]  # Same dimensions
        assert len(processed.shape) == 2  # Grayscale
        assert processed.dtype == np.uint8
    
    def test_clean_equation(self, solver):
        """Test equation text cleaning."""
        test_cases = [
            ("2x + 3 = 7", "2x+3=7"),
            ("x² - 4 = 0", "x²-4=0"),
            ("  x + y = 10  ", "x+y=10"),
            ("2O + 3 = 7", "20+3=7"),  # O -> 0
            ("l + 1 = 2", "1+1=2"),    # l -> 1
        ]
        
        for input_text, expected in test_cases:
            result = solver.clean_equation(input_text)
            assert result == expected
    
    def test_parse_equation(self, solver):
        """Test equation parsing."""
        test_cases = [
            ("2x + 3 = 7", True),
            ("x² - 4 = 0", True),
            ("invalid equation", False),
            ("", False),
        ]
        
        for equation_text, should_succeed in test_cases:
            result = solver.parse_equation(equation_text)
            if should_succeed:
                assert result is not None
                assert isinstance(result, sp.Eq)
            else:
                assert result is None
    
    def test_solve_equation(self, solver):
        """Test equation solving."""
        # Test simple linear equation
        equation = sp.Eq(sp.Symbol('x') * 2 + 3, 7)
        result = solver.solve_equation(equation)
        
        assert result['type'] == 'solution'
        assert 'x' in result['solutions']
        assert result['solutions']['x'] == [2.0]
    
    def test_save_and_retrieve_from_database(self, solver):
        """Test database save and retrieve operations."""
        # Save test data
        solver.save_to_database(
            "test.png",
            "2x + 3 = 7",
            "2*x + 3 = 7",
            {"type": "solution", "solutions": {"x": [2.0]}},
            0.9
        )
        
        # Retrieve and verify
        history = solver.get_equation_history(1)
        assert len(history) >= 1
        
        latest = history[0]
        assert latest['extracted_text'] == "2x + 3 = 7"
        assert latest['equation'] == "2*x + 3 = 7"
        assert latest['confidence'] == 0.9
    
    def test_process_equation_pipeline(self, solver, sample_image):
        """Test the complete processing pipeline."""
        result = solver.process_equation(sample_image)
        
        # Should return a dictionary with success status
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'extracted_text' in result
            assert 'equation' in result
            assert 'solution' in result

class TestImageProcessing:
    """Test cases for image processing utilities."""
    
    def test_create_sample_equation_image(self):
        """Test creating sample equation images."""
        # Test different equation formats
        equations = ["2x + 3 = 7", "x² - 4 = 0", "3y + 2 = 11"]
        
        for eq in equations:
            img = np.ones((200, 400, 3), dtype=np.uint8) * 255
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, eq, (50, 100), font, 1, (0, 0, 0), 2)
            
            assert img.shape == (200, 400, 3)
            assert img.dtype == np.uint8

class TestSymPyIntegration:
    """Test cases for SymPy integration."""
    
    def test_symbol_parsing(self):
        """Test parsing mathematical symbols."""
        from sympy.parsing.sympy_parser import parse_expr
        
        test_cases = [
            ("2*x + 3", "2*x + 3"),
            ("x**2 - 4", "x**2 - 4"),
            ("3*y + 2", "3*y + 2"),
        ]
        
        for input_expr, expected in test_cases:
            result = parse_expr(input_expr)
            assert str(result) == expected
    
    def test_equation_solving(self):
        """Test SymPy equation solving."""
        x = sp.Symbol('x')
        
        # Linear equation
        eq1 = sp.Eq(2*x + 3, 7)
        sol1 = sp.solve(eq1, x)
        assert sol1 == [2]
        
        # Quadratic equation
        eq2 = sp.Eq(x**2 - 4, 0)
        sol2 = sp.solve(eq2, x)
        assert set(sol2) == {-2, 2}

# Integration tests
class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing."""
        solver = HandwrittenEquationSolver()
        
        # Create a test image
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "2x + 3 = 7", (50, 100), font, 1, (0, 0, 0), 2)
        
        # Process the image
        result = solver.process_equation(img)
        
        # Verify the result structure
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'extracted_text' in result
            assert 'equation' in result
            assert 'solution' in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
