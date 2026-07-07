"""
Pipeline Step 01: Image Import, Landmark Reading, CLAHE, and Blur

This script performs the initial preprocessing steps:
1. Load image and landmarks from TPS file
2. Draw landmarks on image
3. Crop to landmark region
4. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
5. Apply blur filter
6. Save intermediate results
"""

from pathlib import Path
import sys

# Add parent directory to path to import pipeline_helpers
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_helpers import (
    load_config,
    validate_config,
    read_landmarks_from_tps, draw_landmarks_on_image, get_landmark_bounds,
    prepare_images_with_landmarks,
    preprocess_pipeline,
    save_images
)


def main():
    """Main pipeline execution."""
    
    # Load configuration
    config_path = Path(
        "C:/Users/korbi/Desktop/A_Master_Thesis/Pipeline/Scripts/Outline_Parameters.yaml"
    )
    config = load_config(config_path)
    
    print(f"Processing image: {config.image_name}")
    print(f"Image path: {config.image_file}")
    print(f"TPS file: {config.tps_file}")
    print(f"Output directory: {config.output_dir}")
    print("-" * 60)
    
    # Read landmarks from TPS file
    landmarks = read_landmarks_from_tps(config.tps_file, config.image_name)
    
    if landmarks is None:
        print(f"Warning: No landmarks found for {config.image_name}")
        landmarks = []
    else:
        print(f"Found {len(landmarks)} landmarks")
    
    # Load and prepare images with landmarks
    images = prepare_images_with_landmarks(
        config.image_file,
        landmarks=landmarks,
        crop=True,
        landmark_radius=8,
        landmark_color=(255, 0, 0)
    )
    
    print(f"Images prepared:")
    print(f"  - RGB: {images['rgb'].shape}")
    print(f"  - Grayscale: {images['gray'].shape}")
    print(f"  - With landmarks: {images['with_landmarks'].shape}")
    print("-" * 60)
    
    # Apply preprocessing pipeline
    preprocessed = preprocess_pipeline(
        images['gray'],
        clahe_params=config.clahe_params,
        blur_params=config.blur_params
    )
    
    print(f"Preprocessing applied:")
    print(f"  - CLAHE clip limit: {config.clahe_params['clipLimit']}")
    print(f"  - CLAHE tile size: {config.clahe_params['tileGridSize']}")
    print(f"  - Blur type: {config.blur_params['type']}")
    print(f"  - Blur kernel size: {config.blur_params['ksize']}")
    print("-" * 60)
    
    # Prepare output images
    output_images = {
        "01_raw.png": images['rgb'],
        "02_landmarks.png": images['with_landmarks'],
        "03_clahe.png": preprocessed['clahe'],
        "04_blur.png": preprocessed['blurred']
    }
    
    # Save all images
    save_images(config.output_dir, output_images, overwrite=True)
    
    print("-" * 60)
    print("Pipeline step 01 completed successfully!")
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()