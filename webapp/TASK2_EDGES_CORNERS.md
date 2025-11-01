# Task 2: Edge and Corner Detection

## Overview
Task 2 implements custom algorithms for detecting **edges** and **corners** in images. These are fundamental feature detection techniques used in computer vision.

## Features
- **Edge Detection**: Custom Canny-like algorithm with Non-Maximum Suppression (NMS) and Hysteresis thresholding
- **Corner Detection**: Harris corner detector with heatmap visualization

## Part A: Edge Detection

### Step-by-Step Instructions

#### Step 1: Load Images
1. Open `index.html` in your web browser
2. Wait for "OpenCV ready" message
3. Click **"Load Images"** button
4. Select images from your dataset

#### Step 2: Select Edge Detection Mode
1. In the mode dropdown, select **"Edge – NMS + Hysteresis"**
2. The image will update automatically

#### Step 3: Adjust Edge Detection Parameters
- **Low Threshold**: Lower bound for edge pixels (default: 15)
  - Lower values = more edges detected
  - Higher values = only strong edges
- **High Threshold**: Upper bound for strong edges (default: 40)
  - Lower values = more edges marked as strong
  - Higher values = only very strong edges
- **Auto Threshold**: Check to automatically calculate thresholds
  - Uses percentile-based method (85th percentile for high, 40% of high for low)
- **Binary View**: Check to see pure edge map (black/white)
  - Unchecked: Shows edges overlaid on original image (red)

#### Step 4: View Results
**What you'll see:**
- **Binary mode OFF**: Red edges overlaid on original image (65% original + 35% red edges)
- **Binary mode ON**: Pure black/white edge map
- **Status bar**: Shows number of edge pixels detected

#### Step 5: Optimize Detection
1. If too many edges: Increase Low/High thresholds
2. If missing edges: Decrease Low/High thresholds
3. Use Auto Threshold for best automatic results

### Understanding Edge Detection

**Algorithm Steps:**
1. **Gradient Computation**: Calculate gradient magnitude and angle using Sobel operator
2. **Non-Maximum Suppression (NMS)**: Keep only local maxima in gradient direction
3. **Hysteresis Thresholding**: 
   - Pixels above high threshold = strong edges
   - Pixels between thresholds = weak edges
   - Weak edges connected to strong edges = kept
   - Isolated weak edges = removed
4. **Dilation**: Thicken edges slightly for visibility

## Part B: Corner Detection

### Step-by-Step Instructions

#### Step 1: Select Corner Detection Mode
1. In the mode dropdown, select **"Corner – Harris"**
2. The image will update automatically

#### Step 2: Adjust Corner Detection Parameters
- **Win (Window Size)**: Size of the correlation window (3, 5, 7, or 9)
  - Larger = more stable but less precise
  - Smaller = more precise but sensitive to noise
- **k×100**: Harris corner response parameter (default: 4 = 0.04)
  - Lower values = detect more corners
  - Higher values = detect fewer, stronger corners
- **Th (Threshold)**: Minimum corner response value (0-255)
  - Lower = more corners detected
  - Higher = only strong corners
- **Heatmap**: Check to visualize corner response as color overlay
  - Shows where corners are detected (not just corner points)

#### Step 3: View Results
**What you'll see:**
- **Green circles**: Detected corner locations
- **With Heatmap**: Color overlay showing corner response strength
  - Bright colors = strong corner response
  - Dark colors = weak/no response
- **Status bar**: Shows number of corners detected

#### Step 4: Optimize Detection
1. **Too many corners**: Increase Threshold or k value
2. **Too few corners**: Decrease Threshold or k value
3. **Corners at wrong locations**: Adjust Window Size

### Understanding Corner Detection

**Harris Corner Detector:**
1. **Compute gradients**: Ix and Iy using Sobel operator
2. **Structure tensor**: Compute Ixx, Iyy, Ixy (products of gradients)
3. **Gaussian smoothing**: Average over correlation window
4. **Corner response**: R = det(M) - k × trace(M)²
   - M = structure tensor matrix
   - k = empirical constant (typically 0.04)
5. **Thresholding**: Keep pixels with R > threshold
6. **Non-maximum suppression**: Keep only local maxima

## Export Results

### Single Image Export
1. Select desired mode (Edge or Corner)
2. Adjust parameters
3. Click **"Save PNG"** button

### Batch Export
1. Load multiple images
2. Click **"Export All (Task 2)"** button
3. Browser will download:
   - `{name}__edge.png` - Edge detection results
   - `{name}__corners.png` - Corner detection results

## Expected Output Files

For each input image:
- `IMG_01__edge.png` - Edge detection visualization
- `IMG_01__corners.png` - Corner detection visualization

## Tips

### Edge Detection Tips
1. **Start with Auto Threshold**: Usually gives good results
2. **Fine-tune manually**: If auto doesn't work well
3. **Use Binary view**: To see clean edge map
4. **Use overlay view**: To see edges in context

### Corner Detection Tips
1. **Enable Heatmap**: Helps understand corner response
2. **Adjust Window Size**: Larger for noisy images
3. **Lower Threshold**: If missing obvious corners
4. **Higher Threshold**: If detecting noise as corners

## Troubleshooting

**Problem**: No edges detected
- **Solution**: Lower the thresholds or enable Auto Threshold

**Problem**: Too many edges (noise)
- **Solution**: Increase thresholds or use Edge Boost = 0

**Problem**: No corners detected
- **Solution**: Lower Threshold value or decrease k value

**Problem**: Too many corners
- **Solution**: Increase Threshold value or adjust Window Size

**Problem**: Corner detection seems slow
- **Solution**: Enable Half-Res checkbox or use smaller images

## Technical Details

### Edge Detection
- **Gradient**: Sobel operator (3x3)
- **Gaussian Blur**: 3x3 kernel, sigma=0.8
- **NMS**: 4-directional (0°, 45°, 90°, 135°)
- **Hysteresis**: Connected component analysis
- **Dilation**: 3x3 kernel for visibility

### Corner Detection
- **Gradient**: Sobel operator (3x3)
- **Correlation Window**: Gaussian blur (default 5x5)
- **Harris k**: Default 0.04
- **Non-maximum suppression**: 3x3 dilation + comparison
- **Fallback**: Uses Shi-Tomasi if Harris fails

## Use Cases

**Edge Detection:**
- Object boundary detection
- Feature extraction
- Image segmentation preprocessing
- Contour detection

**Corner Detection:**
- Feature point detection
- Image matching and tracking
- Camera calibration
- Structure from motion

