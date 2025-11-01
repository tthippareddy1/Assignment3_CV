# Task 4: ArUco Marker-Based Segmentation

## Overview
Task 4 implements object segmentation using ArUco markers placed on object boundaries. The algorithm detects markers, extracts their positions, and creates a segmentation mask by connecting the markers.

## Features
- **ArUco Marker Detection**: Detects markers using OpenCV's ArUco module
- **Multiple Dictionary Support**: Supports various ArUco dictionary types
- **Flexible Point Collection**: Use marker corners or centers
- **Boundary Creation**: Forms polygon boundary from marker positions
- **Mask Generation**: Creates segmentation mask from polygon

## Prerequisites

### Required Setup
1. **ArUco Markers**: You need printed ArUco markers
   - Download markers from `dataset/aruco/aruco.pdf`
   - Print markers (use marker42.png as example)
   - Place markers on object boundary

2. **OpenCV.js with ArUco**: Ensure your OpenCV.js build includes ArUco module
   - Standard OpenCV.js builds should include it
   - If not available, use OpenCV.js build with contrib modules

### Creating Test Images
1. **Prepare Object**: Choose a non-rectangular object
2. **Place Markers**: Stick ArUco markers on object boundary
   - Use at least 3-4 markers (more = better)
   - Place markers evenly around boundary
   - Ensure markers are clearly visible
3. **Capture Images**: Take photos from various angles and distances
   - Capture at least 10 images for evaluation
   - Try different lighting conditions
   - Vary camera distance
   - Capture from different angles

## Step-by-Step Instructions

### Step 1: Load Images with ArUco Markers
1. Open `index.html` in your web browser
2. Wait for "OpenCV ready" message
3. Click **"Load Images"** button
4. Select images from `dataset/aruco/` folder
   - Images should contain visible ArUco markers
   - Supported formats: JPG, PNG

### Step 2: Select ArUco Segmentation Mode
1. In the mode dropdown, select **"Segmentation – ArUco (Task 4)"**
2. The image will update automatically

### Step 3: Configure ArUco Parameters

#### Dictionary Selection
- **Dict**: Select ArUco dictionary matching your markers
  - `4x4_50` (default): Smallest markers, 50 unique IDs
  - `5x5_100`: Medium markers, 100 unique IDs
  - `6x6_250`: Larger markers, 250 unique IDs
  - `7x7_1000`: Largest markers, 1000 unique IDs
  - `AprilTag 36h11`: AprilTag markers (alternative)

**Important**: Dictionary must match your printed markers!

#### Point Collection Mode
- **Use Corners**: Checked (default)
  - Uses all 4 corners of each marker (more points = smoother boundary)
  - Recommended for best results
- **Unchecked**: Uses only marker centers
  - Fewer points = simpler polygon
  - Use if corners are unreliable

#### Visualization Options
- **Show IDs**: Checked (default)
  - Displays marker ID numbers on overlay
  - Helps verify marker detection
- **Dilate(px)**: Pixels to dilate mask (0-12, default: 2)
  - Expands mask outward to better cover object
  - Higher values = larger mask
  - Use if mask is too tight

#### Display Mode
- **Binary**: Check to see pure mask (black/white)
  - Unchecked: Shows green overlay on original image

### Step 4: View Results
**What you'll see:**
- **Normal mode**:
  - Green overlay on segmented object (35% green, 65% original)
  - Green boundary outline connecting markers
  - Marker IDs displayed (if enabled)
  - Status shows: `aruco pts=X` (number of points used)
- **Binary mode**:
  - White mask on black background
  - Shows exact segmentation mask

### Step 5: Troubleshoot Detection

#### If no markers detected:
1. **Check Dictionary**: Ensure it matches your printed markers
2. **Check Lighting**: Markers need good contrast
3. **Check Focus**: Markers must be in focus
4. **Check Size**: Markers must be large enough (at least 20-30 pixels)
5. **Check Angle**: Very oblique angles may not detect

#### If wrong markers detected:
1. **Change Dictionary**: Try different dictionary
2. **Check Marker Quality**: Ensure markers aren't damaged/folded
3. **Improve Lighting**: Better contrast helps

#### If boundary is wrong:
1. **Adjust Dilate**: Increase if mask too small, decrease if too large
2. **Check Use Corners**: Toggle to see difference
3. **Verify Marker Placement**: Markers should be on boundary
4. **Check Marker Order**: Algorithm orders by angle; verify visually

### Step 6: Save Results
1. Click **"Save PNG"** button to save current view
2. File will be named: `{original_name}__seg_aruco.png`

### Step 7: Export All Images (Batch Processing)
1. Load multiple images (at least 10 for evaluation)
2. Click **"Export All (Task 4)"** button
3. Browser will download:
   - `{name}__aruco_mask.png` - Binary mask
   - `{name}__aruco_boundary.png` - Overlay visualization

## Understanding the Algorithm

### Processing Pipeline

1. **Marker Detection**
   - Convert image to grayscale
   - Detect ArUco markers using selected dictionary
   - Extract marker corners and IDs

2. **Point Collection**
   - Extract points from detected markers
   - Option A: All 4 corners of each marker (if "Use Corners" checked)
   - Option B: Center point of each marker (if unchecked)

3. **Point Ordering**
   - Calculate centroid of all points
   - Sort points by angle around centroid
   - Creates non-self-intersecting polygon

4. **Mask Generation**
   - Create polygon from ordered points
   - Fill polygon to create binary mask
   - Optional dilation for refinement

5. **Visualization**
   - Overlay mask on original image
   - Draw boundary outline
   - Display marker IDs (optional)

## Expected Output Files

For each input image:
- `IMG_01__aruco_mask.png` - Binary segmentation mask
- `IMG_01__aruco_boundary.png` - Visual overlay showing boundary

## Tips for Best Results

### Marker Placement
1. **Use Enough Markers**: At least 3-4, more is better
2. **Even Distribution**: Place markers evenly around boundary
3. **Boundary Alignment**: Place markers exactly on object edge
4. **Good Visibility**: Ensure markers are clearly visible
5. **Avoid Occlusion**: Don't place markers where they might be hidden

### Image Capture
1. **Multiple Angles**: Capture from various viewpoints
2. **Different Distances**: Try close-up and far shots
3. **Consistent Lighting**: Avoid harsh shadows on markers
4. **Good Focus**: Ensure markers are sharp
5. **Adequate Size**: Markers should be at least 20-30 pixels

### Parameter Tuning
1. **Dictionary First**: Match dictionary to your markers
2. **Use Corners**: Usually gives better results
3. **Dilate Carefully**: Start with 2, adjust as needed
4. **Show IDs**: Helps verify detection

## Troubleshooting

**Problem**: "ARUCO MODULE MISSING"
- **Cause**: OpenCV.js build doesn't include ArUco module
- **Solution**: Use OpenCV.js build with contrib modules

**Problem**: "ARUCO – NONE"
- **Cause**: No markers detected
- **Solution**:
  - Verify dictionary matches markers
  - Check marker visibility and size
  - Improve lighting
  - Ensure markers are in focus

**Problem**: "ARUCO – TOO FEW PTS"
- **Cause**: Less than 3 points collected
- **Solution**:
  - Enable "Use Corners" to get more points
  - Ensure at least 1 marker is detected
  - Check marker detection

**Problem**: Boundary doesn't match object
- **Cause**: Marker placement or ordering issue
- **Solution**:
  - Verify markers are on boundary
  - Adjust Dilate parameter
  - Check point ordering visually

**Problem**: Mask too small/large
- **Cause**: Dilate parameter incorrect
- **Solution**: Adjust Dilate slider (increase for larger, decrease for smaller)

## Technical Details

- **Detection**: OpenCV ArUco detector
- **Point Ordering**: Angle-based sorting around centroid
- **Polygon**: Non-self-intersecting polygon via angle sorting
- **Mask**: Filled polygon using OpenCV fillPoly
- **Dilation**: Elliptical structuring element
- **Dictionary Support**: Multiple ArUco dictionary types

## Use Cases

- Object segmentation with marker assistance
- Precise boundary definition
- Measurement applications
- Augmented reality
- Object tracking initialization

## Evaluation Requirements

For assignment evaluation, ensure:
- ✅ At least 10 images with ArUco markers
- ✅ Images from different angles
- ✅ Images from different distances
- ✅ Consistent marker placement
- ✅ All images successfully segmented
- ✅ Exported masks and overlays

## Best Practices

1. **Consistent Setup**: Use same dictionary and marker placement across images
2. **Quality Control**: Verify each image detects markers correctly
3. **Documentation**: Note any issues or parameter adjustments
4. **Export Everything**: Export both masks and overlays for comparison
5. **Test Parameters**: Try different settings to find optimal configuration

