# Task 4: ArUco Marker-Based Object Segmentation

## What is This Task?

Imagine you want to automatically identify and outline an object in a photo, but the object has an irregular shape (not a simple rectangle). Task 4 solves this by using **ArUco markers** - special black-and-white square patterns that act like "landmarks" placed around the object's boundary. The computer detects these markers and connects them to create a segmentation mask that outlines the object.

**Think of it like:** Placing numbered flags around a property boundary, then drawing a fence connecting all the flags.

## How Does It Work? (Simple Explanation)

1. **Place markers** on the object's boundary (like placing flags)
2. **Take a photo** of the object with markers
3. **Computer detects** all the markers in the image
4. **Extracts points** from marker positions (corners or centers)
5. **Connects the points** to form a polygon boundary
6. **Creates a mask** showing what's inside vs. outside the boundary

## Step-by-Step Guide

### Step 1: Prepare Your Images

**What you need:**
- Print ArUco markers (available in `dataset/aruco/` folder)
- Place at least 3-4 markers around your object's boundary
- Take photos from different angles and distances
- **Minimum 10 images** for evaluation

**Tips for best results:**
- Place markers evenly spaced around the boundary
- Make sure markers are clearly visible (good lighting)
- Keep markers flat and not wrinkled
- Use the same dictionary for all markers

### Step 2: Open the Web Application

1. Navigate to the `webapp` folder
2. Start a local web server:
   ```bash
   cd webapp
   python3 -m http.server 8000
   ```
3. Open your browser and go to: `http://localhost:8000`
4. Wait for "OpenCV ready. ✅ ArUco available." message

### Step 3: Load Your Images

1. Click the **"Load Images"** button
2. Select your images with ArUco markers
3. The first image will appear automatically

### Step 4: Select Task 4 Mode

1. Find the **mode dropdown** (usually near the top)
2. Select **"Segmentation – ArUco (Task 4)"**
3. The image will process automatically

### Step 5: Adjust Settings (If Needed)

**Dictionary Selection:**
- Choose the dictionary that matches your printed markers
- Default is `DICT_4X4_50` (most common)
- If markers aren't detected, try other dictionaries

**Use Corners:**
- ✅ **Checked** (default): Uses all 4 corners of each marker = smoother boundary
- ❌ **Unchecked**: Uses center point of each marker = simpler polygon

**Show IDs:**
- ✅ **Checked**: Shows marker ID numbers on the image
- ❌ **Unchecked**: No ID labels

**Dilate:**
- Adjusts how much the mask expands outward
- Range: 0-12 pixels
- Default: 2 pixels
- Increase if mask is too tight around object

**Binary Mode:**
- ✅ **Checked**: Shows pure black/white mask only
- ❌ **Unchecked** (default): Shows green overlay on original image

### Step 6: View Results

**What you should see:**
- **Green overlay** on the object area (if Binary mode is off)
- **Green boundary line** connecting the markers
- **Marker IDs** displayed (if "Show IDs" is checked)
- **Status message** showing number of points detected

**Example status:** `aruco pts=24` means 24 points were collected (6 markers × 4 corners each)

### Step 7: Save Individual Results

1. Click **"Save PNG"** button
2. File downloads as: `{image_name}__aruco_segmentation.png`

### Step 8: Export All Images (Batch Processing)

1. Make sure all your images are loaded
2. Click **"Export All (Task 4)"** button
3. Browser will download two files for each image:
   - `{name}__aruco_mask.png` - Pure black/white mask
   - `{name}__aruco_boundary.png` - Visual overlay with boundary

**Note:** Your browser may ask permission to allow multiple downloads.

## Understanding the Output

### Visual Overlay Mode (Default)
- **Green tinted area**: The segmented object region
- **Green boundary line**: The polygon connecting markers
- **Original image**: Still visible underneath (65% opacity)
- **Marker IDs**: Numbers showing which markers were detected

### Binary Mask Mode
- **White pixels**: Object area (inside boundary)
- **Black pixels**: Background area (outside boundary)
- Useful for further processing or analysis

## Common Issues and Solutions

### Problem: "ARUCO MODULE MISSING"
**What it means:** Your OpenCV.js build doesn't include ArUco support.

**Solution:**
- Make sure you're using the local OpenCV.js build from `build_wasm/bin/opencv.js`
- Check that `index.html` loads the local file, not a CDN version
- Verify the file exists: `webapp/build_wasm/bin/opencv.js`

### Problem: "ARUCO – NONE"
**What it means:** No markers detected in the image.

**Solutions:**
- Check that markers are clearly visible
- Try a different dictionary (maybe markers are from different dictionary)
- Improve lighting - markers need good contrast
- Make sure markers aren't too small or too large
- Check that markers aren't rotated too much (should be mostly flat)

### Problem: "ARUCO – TOO FEW PTS"
**What it means:** Less than 3 points collected (need at least 3 to make a polygon).

**Solutions:**
- Add more markers (need at least 3 markers)
- Check marker visibility
- Try adjusting "Use Corners" setting
- Improve image quality

### Problem: Boundary Looks Wrong
**What it means:** The polygon doesn't match the object shape.

**Solutions:**
- Add more markers for smoother boundary
- Try "Use Corners" mode for more points
- Adjust "Dilate" setting to expand/contract mask
- Place markers closer to actual object boundary

### Problem: Markers Detected But No Overlay
**What it means:** Processing completed but visualization isn't showing.

**Solutions:**
- Check browser console for errors (F12 → Console tab)
- Try refreshing the page
- Check that "Binary Mode" isn't accidentally enabled
- Verify image loaded correctly

## Technical Details (For Understanding)

### How Marker Detection Works

1. **Image Conversion**: Color image → Grayscale (easier to detect patterns)
2. **Pattern Recognition**: OpenCV scans for ArUco marker patterns
3. **Corner Extraction**: Finds the 4 corners of each detected marker
4. **ID Assignment**: Each marker gets a unique ID number

### How Point Collection Works

**Option A: Use All Corners (Default)**
- Each marker has 4 corners
- Collects all 4 corner points from each marker
- Example: 6 markers = 24 points
- **Result**: Smoother, more detailed boundary

**Option B: Use Centers Only**
- Calculates center point of each marker
- Collects 1 point per marker
- Example: 6 markers = 6 points
- **Result**: Simpler polygon, faster processing

### How Point Ordering Works

**The Challenge:** Points are detected in random order, but we need them in order around the boundary.

**The Solution:**
1. Calculate the **centroid** (average position) of all points
2. For each point, calculate its **angle** relative to the centroid
3. **Sort points by angle** (like arranging numbers on a clock)
4. This creates a **non-self-intersecting polygon**

**Why this matters:** Without proper ordering, the polygon might cross itself or form weird shapes.

### How Mask Creation Works

1. **Create empty mask**: All pixels start as black (background)
2. **Fill polygon**: All pixels inside the ordered polygon become white (object)
3. **Optional dilation**: Expand the mask outward by a few pixels
4. **Result**: Binary mask (white = object, black = background)

## What Makes This Implementation Good?

✅ **Robust Detection**: Handles multiple ArUco dictionary types  
✅ **Flexible Point Collection**: Choose corners or centers  
✅ **Smart Ordering**: Automatically creates proper polygon  
✅ **Visual Feedback**: See markers and boundary in real-time  
✅ **Batch Processing**: Export all images at once  
✅ **Error Handling**: Clear messages when something goes wrong  
✅ **Memory Management**: Properly cleans up resources  

## Evaluation Checklist

Before submitting, make sure:

- ✅ At least 10 images processed successfully
- ✅ Images from different angles and distances
- ✅ Markers clearly visible in all images
- ✅ Consistent marker placement
- ✅ All images exported (masks and overlays)
- ✅ Boundary looks reasonable for each image
- ✅ No error messages in console
- ✅ Can explain how the algorithm works

## Tips for Best Results

1. **More markers = Better boundary**: Use 6-10 markers for smooth boundaries
2. **Even spacing**: Place markers evenly around the object
3. **Good lighting**: Markers need clear contrast to be detected
4. **Flat markers**: Avoid wrinkled or curved markers
5. **Consistent dictionary**: Use the same dictionary for all markers
6. **Test first**: Try with one image before processing all 10+

## Understanding the Code Structure

**Main Function:** `segAruco(src)`
- Takes input image
- Detects markers
- Collects points
- Creates mask
- Returns result

**Helper Functions:**
- `detectAruco()`: Finds markers in image
- `cornersToPointList()`: Extracts points from markers
- `pointsToContourMat()`: Orders points and creates polygon
- `getArucoMask()`: Creates binary mask from points

## Real-World Applications

This technique is useful for:
- **Robotics**: Identifying objects for manipulation
- **Augmented Reality**: Tracking object boundaries
- **Quality Control**: Inspecting irregular-shaped products
- **Medical Imaging**: Outlining organs or regions
- **Agriculture**: Identifying crops or fields

## Summary

Task 4 uses ArUco markers as "landmarks" to create segmentation masks for irregular objects. The process is:
1. Detect markers → 2. Collect points → 3. Order points → 4. Create mask → 5. Visualize result

The key advantage is that it works for **any shape** as long as you can place markers around it, making it very flexible for real-world applications.
