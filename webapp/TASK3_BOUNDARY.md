# Task 3: Object Boundary Detection

## Overview
Task 3 implements an algorithm to find the exact boundaries of objects in images using contour detection. This uses OpenCV's built-in functions but applies intelligent selection to find the best object boundary.

## Features
- **Contour Detection**: Finds object boundaries using edge detection + morphological operations
- **Smart Selection**: Automatically selects the best contour based on area and position
- **Polygon Approximation**: Simplifies boundaries while preserving shape
- **Visualization**: Shows boundary overlay with optional binary mask view

## Step-by-Step Instructions

### Step 1: Load Images
1. Open `index.html` in your web browser
2. Wait for "OpenCV ready" message
3. Click **"Load Images"** button
4. Select images from your dataset
   - **Best results**: Images with clear object boundaries
   - Images with single main object work best

### Step 2: Select Boundary Detection Mode
1. In the mode dropdown, select **"Boundary – Contour (Task 3)"**
2. The image will update automatically

### Step 3: Adjust Boundary Detection Parameters

#### Basic Parameters (Shared with Edge Detection)
- **Low T / High T**: Canny edge detection thresholds
  - Lower values = more edges = more contours
  - Higher values = fewer edges = fewer contours
- **Auto**: Automatically calculates thresholds using Otsu's method

#### Boundary-Specific Parameters
- **CloseK (Closing Kernel)**: Size of morphological closing kernel (1, 3, 5, 7, 9, 11, 13, 15)
  - **Purpose**: Closes gaps in edges to form closed contours
  - **Smaller values**: Preserves detail but may miss gaps
  - **Larger values**: Closes larger gaps but may merge objects
  - **Default**: 5 pixels

- **MinArea%**: Minimum contour area as percentage of image area (0-30%)
  - **Purpose**: Filters out small noise contours
  - **Lower values**: Keeps more contours (including noise)
  - **Higher values**: Only keeps large objects
  - **Default**: 2%

- **Polyε% (Polygon Epsilon)**: Simplification tolerance as % of perimeter (0-5%)
  - **Purpose**: Simplifies boundary to polygon with fewer vertices
  - **Lower values**: More vertices = more detailed shape
  - **Higher values**: Fewer vertices = simpler shape
  - **Default**: 1.5%

- **CenterR% (Center Radius)**: Preferred distance from image center (0-80%)
  - **Purpose**: Prioritizes contours near image center
  - **Lower values**: Stricter center preference
  - **Higher values**: More flexible position preference
  - **Default**: 40%

#### Display Options
- **Binary**: Check to see pure mask (black/white)
  - Unchecked: Shows green overlay on original image

### Step 4: View Results
**What you'll see:**
- **Normal mode**: 
  - Green overlay on detected object (35% green, 65% original)
  - Green boundary outline
  - Status shows: `area=X px, verts=Y` (area and vertex count)
- **Binary mode**: 
  - White mask on black background
  - Shows exact segmentation mask

### Step 5: Optimize Detection

#### If object boundary is not detected:
1. **Lower CloseK**: Try 3 instead of 5
2. **Lower MinArea%**: Try 1% instead of 2%
3. **Lower thresholds**: Use Auto or manually lower Low/High T
4. **Increase CenterR%**: Make position preference more flexible

#### If detecting wrong object or noise:
1. **Increase MinArea%**: Filter out small contours
2. **Increase thresholds**: Reduce edge detection sensitivity
3. **Increase CloseK**: May help merge fragmented boundaries
4. **Decrease CenterR%**: Stricter center preference

#### If boundary is too detailed/jagged:
1. **Increase Polyε%**: Simplify polygon (try 2-3%)
2. **Increase CloseK**: Smooth edges with larger kernel

#### If boundary misses parts:
1. **Decrease CloseK**: Preserve more detail
2. **Lower thresholds**: Detect more edges
3. **Decrease Polyε%**: Keep more vertices

### Step 6: Save Results
1. Click **"Save PNG"** button to save current view
2. File will be named: `{original_name}__boundary_contour.png`

### Step 7: Export All Images (Batch Processing)
1. Load multiple images
2. Click **"Export All (Task 3)"** button
3. Browser will download:
   - `{name}__mask.png` - Binary mask
   - `{name}__boundary.png` - Overlay visualization

## Understanding the Algorithm

### Processing Pipeline

1. **Preprocessing**
   - Convert to grayscale
   - Apply Gaussian blur (5x5, sigma=0.8)

2. **Edge Detection**
   - Compute gradient magnitude
   - Use Otsu's method or manual thresholds
   - Apply Canny edge detector

3. **Morphological Closing**
   - Close gaps in edges using elliptical kernel
   - Creates closed contours from fragmented edges

4. **Contour Finding**
   - Find all external contours
   - Filter by minimum area

5. **Contour Selection**
   - Score each contour based on:
     - Area (larger = better)
     - Distance from center (closer = better)
     - Weighted combination
   - Select best scoring contour

6. **Polygon Approximation**
   - Simplify contour using Douglas-Peucker algorithm
   - Creates smooth polygon boundary

7. **Mask Generation**
   - Fill polygon to create binary mask
   - Optional dilation for refinement

## Expected Output Files

For each input image:
- `IMG_01__mask.png` - Binary segmentation mask (white=object, black=background)
- `IMG_01__boundary.png` - Visual overlay showing boundary

## Tips

1. **Start with defaults**: Usually works well for most images
2. **Use Auto thresholds**: Often better than manual tuning
3. **Adjust MinArea%**: Most important parameter for filtering
4. **Try different CloseK**: Affects how gaps are closed
5. **Check Binary view**: See exact mask without overlay
6. **For complex objects**: May need multiple parameter adjustments

## Troubleshooting

**Problem**: No boundary detected
- **Cause**: No contours found or all filtered out
- **Solution**: 
  - Lower MinArea%
  - Lower edge detection thresholds
  - Decrease CloseK

**Problem**: Wrong object selected
- **Cause**: Multiple objects, algorithm picked wrong one
- **Solution**:
  - Increase MinArea% to focus on larger objects
  - Decrease CenterR% for stricter center preference
  - Adjust thresholds to better isolate target object

**Problem**: Boundary has gaps
- **Cause**: Edges not connected properly
- **Solution**:
  - Increase CloseK (try 7 or 9)
  - Lower edge thresholds to detect more edges
  - Check if object has weak edges

**Problem**: Boundary too jagged/rough
- **Cause**: Too many vertices in polygon
- **Solution**:
  - Increase Polyε% (try 2-3%)
  - Increase CloseK for smoother edges

**Problem**: Boundary misses parts of object
- **Cause**: Edges not detected or gaps too large
- **Solution**:
  - Lower edge thresholds
  - Decrease CloseK
  - Lower Polyε% to preserve detail

## Technical Details

- **Edge Detection**: Canny with automatic threshold selection
- **Morphology**: Elliptical structuring element
- **Contour Finding**: RETR_EXTERNAL mode (only outer contours)
- **Approximation**: Douglas-Peucker algorithm
- **Scoring**: `area × (center_factor) × (1 - 0.6 × distance_ratio)`
- **Memory**: Proper cleanup of all OpenCV Mat objects

## Use Cases

- Object segmentation
- Foreground/background separation
- Region of interest (ROI) extraction
- Preprocessing for other computer vision tasks
- Measurement and analysis

## Best Practices

1. **Image Quality**: Use well-lit images with clear boundaries
2. **Single Object**: Works best with one main object
3. **Background**: Contrasting background helps detection
4. **Parameter Tuning**: Start with defaults, adjust one parameter at a time
5. **Export Masks**: Use binary masks for further processing

