# Task 1: Gradient and Laplacian of Gaussian (LoG)

## Overview
Task 1 computes gradient images (magnitude and angle) and Laplacian of Gaussian (LoG) filtered versions of input images. These are fundamental edge detection and image analysis techniques.

## Features
- **Gradient Magnitude**: Shows the strength of edges at each pixel
- **Gradient Angle**: Shows the direction of edges (visualized as HSV color)
- **Laplacian of Gaussian (LoG)**: Edge detection using second-order derivatives

## Step-by-Step Instructions

### Step 1: Load Images
1. Open `index.html` in your web browser
2. Wait for "OpenCV ready" message in the status bar
3. Click **"Load Images"** button
4. Select one or more images from your dataset folder
   - Recommended: Use images from `dataset/base/` folder
   - Supported formats: JPG, PNG

### Step 2: Select Gradient Magnitude Mode
1. In the mode dropdown, select **"Gradient – Magnitude"**
2. The image will update automatically
3. **What you'll see:**
   - Grayscale image showing edge strength
   - Brighter pixels = stronger edges
   - Darker pixels = weaker/no edges

### Step 3: Select Gradient Angle Mode
1. In the mode dropdown, select **"Gradient – Angle (HSV)"**
2. The image will update automatically
3. **What you'll see:**
   - Color-coded image showing edge directions
   - Colors represent different angles:
     - Red/Magenta: Horizontal edges (0°/180°)
     - Yellow/Green: Diagonal edges (45°/135°)
     - Cyan/Blue: Vertical edges (90°)
   - Only pixels with significant gradient magnitude are colored

### Step 4: Select LoG Mode
1. In the mode dropdown, select **"Laplacian of Gaussian (LoG)"**
2. The image will update automatically
3. **What you'll see:**
   - Grayscale image showing zero-crossings (edge locations)
   - Bright areas = positive Laplacian response
   - Dark areas = negative Laplacian response
   - Zero-crossings occur at edges

### Step 5: Adjust Parameters (Optional)
- **Edge Boost**: 
  - `0` = No boost
  - `1` = Slight dilation for visibility
  - `2` = Dilation + histogram equalization
- **Half-Res**: Check to process at half resolution (faster processing)

### Step 6: Save Results
1. Click **"Save PNG"** button to save the current view
2. File will be named: `{original_name}__{mode}.png`

### Step 7: Export All Images (Batch Processing)
1. Load multiple images
2. Click **"Export All (Task 1)"** button
3. Browser will download:
   - `{name}__grad_mag.png` - Gradient magnitude
   - `{name}__grad_angle.png` - Gradient angle
   - `{name}__log.png` - Laplacian of Gaussian

## Expected Output Files

For each input image, you'll get:
- `IMG_01__grad_mag.png` - Gradient magnitude visualization
- `IMG_01__grad_angle.png` - Gradient angle (HSV color-coded)
- `IMG_01__log.png` - Laplacian of Gaussian result

## Understanding the Results

### Gradient Magnitude
- **High values**: Strong edges (object boundaries, sharp transitions)
- **Low values**: Smooth regions, gradual transitions
- **Use case**: Edge detection, object boundary detection

### Gradient Angle
- **Color hue**: Represents edge direction
- **Brightness**: Represents edge strength (from magnitude)
- **Use case**: Edge orientation analysis, texture analysis

### Laplacian of Gaussian
- **Zero-crossings**: Actual edge locations
- **Positive values**: Regions with positive curvature
- **Negative values**: Regions with negative curvature
- **Use case**: Precise edge detection, blob detection

## Tips

1. **For better edge visibility**: Use Edge Boost = 2
2. **For faster processing**: Enable Half-Res checkbox
3. **Compare results**: Switch between modes to see different edge characteristics
4. **Large images**: May take a few seconds - be patient

## Troubleshooting

**Problem**: Image doesn't update
- **Solution**: Wait for OpenCV to finish loading (check status bar)

**Problem**: Results look too dark/bright
- **Solution**: Try Edge Boost settings or check image exposure

**Problem**: Processing is slow
- **Solution**: Enable Half-Res checkbox or use smaller images

## Technical Details

- **Gradient**: Computed using Sobel operator (3x3 kernel)
- **Gaussian Blur**: Applied before gradient (3x3, sigma=0.8)
- **LoG**: Gaussian blur (5x5, sigma=1.0) + Laplacian (3x3 kernel)
- **Normalization**: All outputs normalized to 0-255 range

