#!/usr/bin/env python3
"""
Command Line Interface for Handwritten Equation Solver
"""

import argparse
import sys
import os
from pathlib import Path
import json
from modern_solver import HandwrittenEquationSolver
import cv2
import numpy as np
from PIL import Image

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Solve handwritten equations from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_solver.py equation.png
  python cli_solver.py equation.png --output results.json
  python cli_solver.py equation.png --verbose
  python cli_solver.py equation.png --save-image processed.png
        """
    )
    
    parser.add_argument(
        "image_path",
        help="Path to the image containing handwritten equation"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path for results (JSON format)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--save-image",
        help="Save processed image to specified path"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for OCR (0.0-1.0)"
    )
    
    parser.add_argument(
        "--no-database",
        action="store_true",
        help="Don't save results to database"
    )
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found")
        sys.exit(1)
    
    # Initialize solver
    print("Initializing Handwritten Equation Solver...")
    solver = HandwrittenEquationSolver()
    
    # Load image
    try:
        image = cv2.imread(args.image_path)
        if image is None:
            print(f"Error: Could not load image '{args.image_path}'")
            sys.exit(1)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if args.verbose:
            print(f"Loaded image: {image.shape}")
        
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    # Process equation
    print("Processing equation...")
    try:
        result = solver.process_equation(image)
        
        if result['success']:
            print("✅ Equation solved successfully!")
            print(f"📝 Extracted text: {result['extracted_text']}")
            print(f"🧮 Parsed equation: {result['equation']}")
            
            solution = result['solution']
            if solution['type'] == 'solution':
                print("💡 Solution:")
                for var, values in solution['solutions'].items():
                    if len(values) == 1:
                        print(f"  {var} = {values[0]}")
                    else:
                        print(f"  {var}: {', '.join(map(str, values))}")
            
            elif solution['type'] == 'evaluation':
                print(f"💡 Result: {solution['result']}")
            
            elif solution['type'] == 'error':
                print(f"❌ Error: {solution['error']}")
            
            # Save to database if requested
            if not args.no_database:
                solver.save_to_database(
                    args.image_path,
                    result['extracted_text'],
                    result['equation'],
                    solution
                )
                if args.verbose:
                    print("💾 Results saved to database")
            
            # Save processed image if requested
            if args.save_image:
                processed_image = solver.preprocess_image(image)
                cv2.imwrite(args.save_image, processed_image)
                if args.verbose:
                    print(f"🖼️ Processed image saved to: {args.save_image}")
            
            # Save results to file if requested
            if args.output:
                output_data = {
                    'image_path': args.image_path,
                    'extracted_text': result['extracted_text'],
                    'equation': result['equation'],
                    'solution': solution,
                    'timestamp': str(Path(args.image_path).stat().st_mtime)
                }
                
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                
                if args.verbose:
                    print(f"📄 Results saved to: {args.output}")
        
        else:
            print(f"❌ Failed to solve equation: {result['error']}")
            if 'extracted_text' in result:
                print(f"📝 Extracted text: {result['extracted_text']}")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error processing equation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
