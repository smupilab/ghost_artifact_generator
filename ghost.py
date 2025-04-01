import cv2
import numpy as np
import os
import argparse
import glob

def ghost_artifacts(image, offset_pixels, ghost_edge_ratio=0.1, ghost_alpha=0.3):
    """
    Detect edges in the input image and select a continuous region that corresponds to
    ghost_edge_ratio of the total edge pixels. The selected region is shifted by offset_pixels
    in a random direction, generating a blurred ghost effect and creating a mask
    (where the labeled region is 255 and the rest is 0).
    
    Blurred ghost effect:
      The input image is shifted, then a Gaussian blur is applied,
      and the result is blended with the original image.
    """
    # Convert image to grayscale for edge detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Perform Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Calculate total edge pixels and the target number of pixels
    total_edge_pixels = np.count_nonzero(edges)
    target_pixels = total_edge_pixels * ghost_edge_ratio
    
    # Find contours (continuous edge regions)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    
    # Create a mask by accumulating contours until reaching ghost_edge_ratio
    selected_mask = np.zeros_like(edges)
    current_pixels = 0
    np.random.shuffle(contours)
    for cnt in contours:
        temp_mask = np.zeros_like(edges)
        cv2.drawContours(temp_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        cnt_pixels = np.count_nonzero(temp_mask)
        if current_pixels == 0 or current_pixels + cnt_pixels <= target_pixels:
            selected_mask = cv2.bitwise_or(selected_mask, temp_mask)
            current_pixels += cnt_pixels
        if current_pixels >= target_pixels:
            break

    # Shift the mask: choose a random angle between 0 and 2π and compute offset
    angle = np.random.uniform(0, 2 * np.pi)
    dx = int(round(offset_pixels * np.cos(angle)))
    dy = int(round(offset_pixels * np.sin(angle)))
    
    height, width = selected_mask.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_mask = cv2.warpAffine(selected_mask, M, (width, height))
    
    # Apply dilation to the shifted mask to thicken the labeled lines
    kernel = np.ones((3, 3), np.uint8)
    thick_mask = cv2.dilate(shifted_mask, kernel, iterations=1)
    
    # Generate the blurred ghost effect: shift the image, apply Gaussian blur, and blend
    if len(image.shape) == 2:
        composite_blur = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        composite_blur = image.copy()
    ghost_layer = cv2.warpAffine(image, M, (width, height))
    ghost_layer = cv2.GaussianBlur(ghost_layer, (3, 3), 0)
    mask_bool = thick_mask.astype(bool)
    composite_blur[mask_bool] = cv2.addWeighted(image[mask_bool], 1 - ghost_alpha,
                                                 ghost_layer[mask_bool], ghost_alpha, 0)
    
    # Return the blurred ghost result and the mask image (labeled region is 255, rest is 0)
    return composite_blur, thick_mask

def get_image_files_from_folder(folder):
    # Supported image file extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return files

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Ghost Artifact Generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', nargs='+', help="Input image file paths (multiple files allowed)")
    group.add_argument('--folder', help="Path to the folder containing input images")
    
    parser.add_argument('--offset', type=int, default=10, help="Offset pixels (default: 10)")
    parser.add_argument('--ratio', type=float, default=0.1, help="Edge ratio (0~1, default: 0.1)")
    parser.add_argument('--alpha', type=float, default=0.3, help="Ghost blur alpha (0~1, default: 0.3)")
    args = parser.parse_args()
    
    # Create the list of input files using either --input or --folder
    if args.folder:
        if not os.path.isdir(args.folder):
            print(f"Not a valid folder: {args.folder}")
            return
        input_files = get_image_files_from_folder(args.folder)
        if not input_files:
            print("No supported image files found in the folder.")
            return
    else:
        input_files = args.input

    # Create output folders if they do not exist
    output_folders = {
        "original": "original",
        "ghost": "ghost artifact",
        "mask": "mask"  # Mask files are saved in the "mask" folder
    }
    for folder in output_folders.values():
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Process each input image
    for file_path in input_files:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Cannot read file: {file_path}. Skipping.")
            continue

        # Save the original image (preserving original dimensions) in the "original" folder
        original_save_path = os.path.join(output_folders["original"], os.path.basename(file_path))
        cv2.imwrite(original_save_path, image)

        # Generate blurred ghost effect and mask using ghost_artifacts function
        result_ghost, result_mask = ghost_artifacts(image, args.offset, args.ratio, args.alpha)
        
        # Append option values to the filename for later analysis
        base_name, ext = os.path.splitext(os.path.basename(file_path))
        params = f"_offset{args.offset}_ratio{args.ratio}_alpha{args.alpha}"
        ghost_save_path = os.path.join(output_folders["ghost"], base_name + params + "_ghost_blur" + ext)
        mask_save_path = os.path.join(output_folders["mask"], base_name + params + "_mask" + ext)
        
        cv2.imwrite(ghost_save_path, result_ghost)
        cv2.imwrite(mask_save_path, result_mask)
        
        print(f"Processed: {file_path}\n  Original: {original_save_path}\n  Blurred Ghost: {ghost_save_path}\n  Mask: {mask_save_path}")

if __name__ == '__main__':
    main()
