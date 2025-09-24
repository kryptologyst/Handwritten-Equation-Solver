#!/usr/bin/env python3
"""
Sample Equation Image Generator
Creates test images with handwritten-style equations for testing the solver
"""

import cv2
import numpy as np
import random
import os
from pathlib import Path

def create_handwritten_style_text(image, text, position, font_scale=1.0, thickness=2):
    """Create handwritten-style text with slight variations."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Add slight random variations to make it look more handwritten
    x_offset = random.randint(-5, 5)
    y_offset = random.randint(-5, 5)
    
    # Add slight rotation
    angle = random.uniform(-2, 2)
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Create rotation matrix
    center = (position[0] + text_width // 2, position[1] + text_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Create a temporary image for the text
    temp_img = np.zeros_like(image)
    cv2.putText(temp_img, text, (position[0] + x_offset, position[1] + y_offset), 
                font, font_scale, (0, 0, 0), thickness)
    
    # Apply rotation
    rotated = cv2.warpAffine(temp_img, rotation_matrix, (image.shape[1], image.shape[0]))
    
    # Add to main image
    image = cv2.add(image, rotated)
    
    return image

def generate_sample_equations():
    """Generate sample equation images."""
    
    # Create output directory
    output_dir = Path("sample_images")
    output_dir.mkdir(exist_ok=True)
    
    # Sample equations
    equations = [
        "2x + 3 = 7",
        "x² - 4 = 0", 
        "3y + 2 = 11",
        "2x + 3y = 12",
        "x³ - 8 = 0",
        "5x - 2 = 13",
        "x² + 5x + 6 = 0",
        "2x - 7 = 3",
        "x + y = 10",
        "3x² - 12 = 0"
    ]
    
    for i, equation in enumerate(equations):
        # Create white background
        img = np.ones((300, 600, 3), dtype=np.uint8) * 255
        
        # Add some noise to make it look more realistic
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # Add equation text
        img = create_handwritten_style_text(img, equation, (50, 150), font_scale=1.5, thickness=3)
        
        # Add some random lines/doodles to make it look more handwritten
        for _ in range(random.randint(2, 5)):
            pt1 = (random.randint(0, 600), random.randint(0, 300))
            pt2 = (random.randint(0, 600), random.randint(0, 300))
            cv2.line(img, pt1, pt2, (200, 200, 200), 1)
        
        # Save image
        filename = f"equation_{i+1:02d}.png"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), img)
        print(f"Generated: {filepath}")
    
    print(f"\nGenerated {len(equations)} sample equation images in '{output_dir}' directory")
    print("You can now test the solver with these images!")

def generate_custom_equation(equation_text, filename="custom_equation.png"):
    """Generate a custom equation image."""
    
    # Create output directory
    output_dir = Path("sample_images")
    output_dir.mkdir(exist_ok=True)
    
    # Create white background
    img = np.ones((300, 600, 3), dtype=np.uint8) * 255
    
    # Add some noise
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Add equation text
    img = create_handwritten_style_text(img, equation_text, (50, 150), font_scale=1.5, thickness=3)
    
    # Save image
    filepath = output_dir / filename
    cv2.imwrite(str(filepath), img)
    print(f"Generated custom equation: {filepath}")
    
    return str(filepath)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample equation images")
    parser.add_argument("--equation", help="Generate a custom equation image")
    parser.add_argument("--filename", default="custom_equation.png", help="Output filename for custom equation")
    
    args = parser.parse_args()
    
    if args.equation:
        generate_custom_equation(args.equation, args.filename)
    else:
        generate_sample_equations()
