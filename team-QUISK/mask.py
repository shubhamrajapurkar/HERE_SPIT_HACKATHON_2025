import os
import io
import base64
import json
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import uuid
import google.generativeai as genai
from google.generativeai import types
from itertools import cycle
import requests
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# ======== CONFIGURATION - UPDATE THESE VALUES AS NEEDED ========

# Dataset paths - updated to match your local paths
BASE_DIR = "D:\HERE\Actual\datasets"  # Your dataset directory

# Output directory
OUTPUT_DIR = "D:\HERE\Actual\datasets\masked"  # Where to save masked images

# Processing parameters
BATCH_SIZE = 2  # Number of images to process in parallel
MAX_IMAGES = 0  # Maximum number of images to process (0 for all)

# Gemini API key(s) - ADD YOUR API KEY(S) HERE
API_KEYS = [
    "AIzaSyAGu5CIMhQf10BxjCQ7AWZSpm_9TMdypfI",  
    "AIzaSyDacxJlZj1hCd1hDO2TWNm9PI0Mk3jd3u4",
    "AIzaSyAqfzrYDqh7lRvaX7YIZrCAosyMerJmHnY", 
    "AIzaSyDiiIydsJAzuFNgwdUPRc9FhfA3cYIaEXM",
    "AIzaSyDGGGXjrwPDlkDYSbWr-yx1jOUF1ViYS1U",
]

# Gemini model configuration
MODEL_ID = "gemini-2.0-flash"  # The model used for image processing
TEMP_DIR = tempfile.mkdtemp()  # Temporary directory for processing

# ======== END OF CONFIGURATION ========

# Set up safety settings for Gemini - updated to use SafetySettingDict
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH",
    }
]

# The prompt for roundabout segmentation
roundabout_segmentation_prompt = """
Your task is to create a precise binary mask for the roundabout in this Map image.

Instructions:
1. Analyze the image and identify the roundabout structure.
2. Generate a binary mask where:
   - White pixels (255) represent the roundabout area
   - Black pixels (0) represent everything else
3. Include the entire roundabout structure including:
   - The central island
   - The circulatory roadway
   - The approach/exit lanes that are part of the roundabout design

Return a JSON with exactly these fields:
{
  "mask_coordinates": [
    [x1, y1], [x2, y2], ... [xn, yn]
  ],
  "confidence": 0.0-1.0,
  "description": "Brief description of the roundabout's appearance and location in the image"
}

The mask_coordinates should form the polygon outline of the roundabout area. Be as precise as possible. I also want the Masked image remember.
"""

def get_roundabout_images(base_dir):
    """Get paths to all roundabout images in the train and validation sets."""
    image_paths = []
    
    # Process both train and validation sets
    for dataset_type in ['train', 'val']:
        roundabout_dir = os.path.join(base_dir, dataset_type, 'roundabout')
        if os.path.exists(roundabout_dir):
            for filename in os.listdir(roundabout_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append({
                        'path': os.path.join(roundabout_dir, filename),
                        'dataset_type': dataset_type,
                        'filename': filename
                    })
    
    return image_paths

def clean_json_response(text):
    """Extract JSON from text response."""
    try:
        # Handle None response
        if text is None:
            print("Received None response from API")
            return {"mask_coordinates": [], "confidence": 0, "description": "Empty API response"}
            
        # Try to find JSON content
        if "```json" in text:
            json_text = text.split("```json")[1].split("```")[0].strip()
        elif "{" in text and "}" in text:
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1
            json_text = text[start_idx:end_idx]
        else:
            json_text = text
            
        # Parse and validate JSON
        return json.loads(json_text)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(f"Original text: {text}")
        return {"mask_coordinates": [], "confidence": 0, "description": f"Failed to parse response: {str(e)}"}

def create_mask_from_coordinates(coordinates, image_size):
    """Create a binary mask from polygon coordinates."""
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    if coordinates and len(coordinates) > 2:
        # Convert to tuple format for PIL
        polygon = [(int(x), int(y)) for x, y in coordinates]
        draw.polygon(polygon, fill=255)
    
    return mask

def process_image_with_gemini(image_info, api_cycle):
    """Process a single image with Gemini to generate a mask."""
    api_key = next(api_cycle)
    genai.configure(api_key=api_key)
    
    try:
        # Load and resize image
        image_path = image_info['path']
        image = Image.open(image_path).convert('RGB')
        
        # Keep original size for the mask
        original_size = image.size
        
        # Resize if too large (Gemini has input limits)
        img_copy = image.copy()
        if max(original_size) > 1024:
            img_copy.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        # Call Gemini API to analyze the image - updated API usage
        model = genai.GenerativeModel(MODEL_ID)
        response = model.generate_content(
            [roundabout_segmentation_prompt, img_copy],
            generation_config=genai.GenerationConfig(
                temperature=0.2,
            ),
            safety_settings=safety_settings,
        )
        
        # Check if response has text content
        if not hasattr(response, 'text') or not response.text:
            print(f"Warning: Empty response for {image_path}")
            result = {"mask_coordinates": [], "confidence": 0, "description": "Empty API response"}
        else:
            # Parse JSON response
            result = clean_json_response(response.text)
        
        # Create mask from coordinates
        mask_coordinates = result.get("mask_coordinates", [])
        
        # If coordinates were successfully extracted
        if mask_coordinates:
            mask = create_mask_from_coordinates(mask_coordinates, original_size)
            
            # Create output directory structure
            output_base = os.path.join(OUTPUT_DIR, image_info['dataset_type'], 'roundabout')
            os.makedirs(output_base, exist_ok=True)
            
            # Save the mask
            mask_filename = f"{os.path.splitext(image_info['filename'])[0]}_mask.png"
            mask_path = os.path.join(output_base, mask_filename)
            mask.save(mask_path)
            
            # Also save the original image to the output directory for convenience
            output_image_path = os.path.join(output_base, image_info['filename'])
            image.save(output_image_path)
            
            # Optional: Save a visualization of the mask overlaid on the image
            overlay = image.copy()
            overlay_mask = mask.convert('RGBA')
            overlay_mask.putalpha(128)  # 50% transparency
            overlay.paste(overlay_mask, (0, 0), overlay_mask)
            
            overlay_filename = f"{os.path.splitext(image_info['filename'])[0]}_overlay.png"
            overlay_path = os.path.join(output_base, overlay_filename)
            overlay.save(overlay_path)
            
            return {
                "original_image": image_info['path'],
                "mask_path": mask_path,
                "overlay_path": overlay_path,
                "confidence": result.get("confidence", 0),
                "description": result.get("description", ""),
                "success": True
            }
        else:
            print(f"No valid mask coordinates returned for {image_path}")
            return {
                "original_image": image_info['path'],
                "success": False,
                "error": "No valid mask coordinates"
            }
        
    except Exception as e:
        print(f"Error processing image {image_info['path']}: {e}")
        return {
            "original_image": image_info['path'],
            "success": False,
            "error": str(e)
        }

def visualize_results(results, num_samples=3):
    """Visualize a sample of the masking results."""
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        print("No successful masking results to visualize")
        return
    
    # Select a few samples to display
    samples = successful_results[:min(num_samples, len(successful_results))]
    
    fig, axs = plt.subplots(len(samples), 3, figsize=(15, 5*len(samples)))
    if len(samples) == 1:
        axs = [axs]  # Handle single row case
        
    for i, result in enumerate(samples):
        # Original image
        original = Image.open(result['original_image'])
        axs[i][0].imshow(original)
        axs[i][0].set_title('Original Image')
        axs[i][0].axis('off')
        
        # Mask
        mask = Image.open(result['mask_path'])
        axs[i][1].imshow(mask, cmap='gray')
        axs[i][1].set_title(f'Mask (Confidence: {result["confidence"]:.2f})')
        axs[i][1].axis('off')
        
        # Overlay
        overlay = Image.open(result['overlay_path'])
        axs[i][2].imshow(overlay)
        axs[i][2].set_title('Overlay')
        axs[i][2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_results.png'))
    plt.show()

def main():
    """Main function to run the masking pipeline."""
    print("======== Roundabout Masking Script ========")
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Validate API keys - modified check to avoid hardcoded key comparison
    if not API_KEYS or any(not key or len(key) < 20 for key in API_KEYS):
        print("\nERROR: No valid API keys provided!")
        print("Please edit this script and add your Gemini API key(s) in the CONFIGURATION section.")
        return
        
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create cycling iterator for API keys
    api_cycle = cycle(API_KEYS)
    
    # Get all roundabout images
    image_info_list = get_roundabout_images(BASE_DIR)
    print(f"Found {len(image_info_list)} roundabout images")
    
    # Limit the number of images if specified
    if MAX_IMAGES > 0:
        image_info_list = image_info_list[:MAX_IMAGES]
        print(f"Processing {len(image_info_list)} images (limited by MAX_IMAGES)")
    
    # Process images in batches to avoid overwhelming the API
    results = []
    for i in tqdm(range(0, len(image_info_list), BATCH_SIZE), desc="Processing batches"):
        batch = image_info_list[i:i+BATCH_SIZE]
        
        for image_info in batch:
            result = process_image_with_gemini(image_info, api_cycle)
            results.append(result)
            # Small delay to avoid rate limits
            time.sleep(1)
    
    # Count successes and failures
    successes = sum(1 for r in results if r.get('success', False))
    failures = len(results) - successes
    
    print(f"\nProcessing complete!")
    print(f"Total images processed: {len(results)}")
    print(f"Successful masks: {successes}")
    print(f"Failed masks: {failures}")
    
    # Visualize some examples
    if successes > 0:
        visualize_results(results)
        
    # Save results log
    with open(os.path.join(OUTPUT_DIR, 'processing_results.json'), 'w') as f:
        json.dump({
            'total': len(results),
            'successes': successes,
            'failures': failures,
            'results': results
        }, f, indent=2)

if __name__ == "__main__":
    main()
