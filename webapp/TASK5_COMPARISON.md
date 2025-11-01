# Task 5: ArUco vs SAM2 Segmentation Comparison

## Overview
Task 5 compares segmentation results from Task 4 (ArUco marker-based) with SAM2 (Segment Anything Model 2), a state-of-the-art deep learning segmentation model. This comparison helps evaluate the performance and characteristics of both methods.

## Features
- **Side-by-Side Comparison**: Visual comparison of ArUco and SAM2 results
- **Difference Visualization**: Highlights areas where methods differ
- **Quantitative Metrics**: IoU (Intersection over Union), area measurements, difference analysis
- **Batch Processing**: Compare all images in dataset

## Prerequisites

### Required Setup
1. **Complete Task 4**: Ensure ArUco segmentation works first
2. **Images with ArUco Markers**: Same images used in Task 4
3. **OpenCV.js**: Must include ArUco module
4. **SAM2 Model** (Optional): 
   - Current implementation uses simulated SAM2
   - For full SAM2: Requires ONNX model conversion and hosting

## Step-by-Step Instructions

### Step 1: Verify Task 4 Works
**IMPORTANT**: Task 5 depends on Task 4. Verify ArUco segmentation works first.

1. Load an image with ArUco markers
2. Select **"Segmentation – ArUco (Task 4)"** mode
3. Verify markers are detected and boundary is shown
4. If not working, fix Task 4 first before proceeding

### Step 2: Load Images
1. Open `index.html` in your web browser
2. Wait for "OpenCV ready" message
3. Click **"Load Images"** button
4. Select images from `dataset/aruco/` folder
   - Use same images as Task 4
   - At least 10 images recommended for evaluation

### Step 3: Select Comparison Mode
1. In the mode dropdown, select **"Compare – ArUco vs SAM2 (Task 5)"**
2. Wait a few seconds for processing (comparison takes longer)

### Step 4: View Comparison Results

**What you'll see:**
- **Three panels side-by-side:**

  **Left Panel (Green) - ArUco Result:**
  - Green overlay showing ArUco segmentation
  - Same result as Task 4
  - Label: "ArUco"

  **Middle Panel (Red) - SAM2 Result:**
  - Red overlay showing SAM2 segmentation
  - May have smoother boundaries
  - May extend beyond marker positions
  - Label: "SAM2"

  **Right Panel (Yellow) - Difference:**
  - Yellow highlights show areas where masks differ
  - More yellow = more difference between methods
  - Shows SAM2's edge refinement areas
  - Label: "Diff"

- **Status Bar Metrics:**
  ```
  IoU: 85.3% | ArUco: 12543px | SAM2: 13456px | Diff: 2341px
  ```
  - **IoU**: Intersection over Union (0-100%)
    - Higher = more similar masks
    - 80%+ = very similar
    - 50-80% = moderately similar
    - <50% = quite different
  - **ArUco**: Number of pixels in ArUco mask
  - **SAM2**: Number of pixels in SAM2 mask
  - **Diff**: Number of pixels that differ between masks

### Step 5: Check Browser Console (Recommended)
1. Open Developer Tools (F12 or Cmd+Option+I)
2. Go to Console tab
3. Look for debug logs:
   ```
   [Task 5] Starting comparison...
   [Task 5] Getting ArUco mask...
   [Task 5] ArUco mask: Found (480x640)
   [Task 5] Getting SAM2 mask...
   [Task 5] SAM2 mask: Found (480x640)
   [Task 5] Calculating metrics...
   [Task 5] Metrics: {iou: 0.85, arucoArea: 12345, sam2Area: 13456, diffArea: 2341}
   [Task 5] Comparison complete. IoU: 85.0%
   ```

### Step 6: Adjust ArUco Parameters (If Needed)
You can adjust Task 4 parameters and see how comparison changes:
- **Dictionary**: Ensure correct dictionary is selected
- **Use Corners**: Toggle to see effect on comparison
- **Dilate**: Adjust to see how it affects comparison
- Comparison will update automatically

### Step 7: Save Results
1. Click **"Save PNG"** button to save current comparison
2. File will be named: `{original_name}__compare_sam2.png`
3. This saves the three-panel comparison image

### Step 8: Export All Images (Batch Processing)
1. Load multiple images (at least 10 for evaluation)
2. Click **"Export All (Task 5)"** button
3. Browser will download for each image:
   - `{name}__compare_sam2.png` - Three-panel comparison
   - `{name}__aruco_mask_task5.png` - ArUco mask only
   - `{name}__sam2_mask_task5.png` - SAM2 mask only

## Understanding the Comparison

### Visual Differences

**ArUco Segmentation (Left):**
- Polygon-based boundary
- Follows marker positions exactly
- Sharp corners at marker locations
- Limited by marker placement

**SAM2 Segmentation (Middle):**
- Edge-aware refinement
- Smoother boundaries
- May extend beyond markers
- Better follows object edges

**Difference Visualization (Right):**
- Shows where methods disagree
- Yellow areas = differences
- Helps identify strengths/weaknesses of each method

### Metrics Interpretation

**IoU (Intersection over Union):**
- **Formula**: IoU = (Intersection) / (Union)
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Interpretation**:
  - **90-100%**: Excellent agreement
  - **70-90%**: Good agreement
  - **50-70%**: Moderate agreement
  - **<50%**: Significant differences

**Area Comparison:**
- **ArUco Area**: Size of ArUco mask
- **SAM2 Area**: Size of SAM2 mask
- **Difference**: How much they differ
- **Interpretation**:
  - Similar areas = similar segmentation
  - Large difference = different object coverage

### Expected Results

**Typical Behavior:**
- SAM2 mask may be slightly larger (extends beyond markers)
- SAM2 boundaries should be smoother
- IoU typically 60-90% (depending on marker placement)
- Difference areas show SAM2's edge refinement

**Why Differences Occur:**
1. **Marker Limitations**: ArUco limited to marker positions
2. **Edge Following**: SAM2 better follows object edges
3. **Boundary Smoothness**: SAM2 produces smoother boundaries
4. **Occlusion Handling**: SAM2 may handle occlusions better

## Troubleshooting

**Problem**: "COMPARE – MISSING MASKS"
- **Cause**: No ArUco markers detected
- **Solution**:
  - Verify Task 4 works first
  - Check marker visibility
  - Ensure correct dictionary selected
  - Improve lighting/focus

**Problem**: White/blank comparison image
- **Cause**: Image processing error
- **Solution**:
  - Check browser console for errors
  - Try smaller image or enable "Half-Res"
  - Refresh page and try again
  - Verify both masks are generated

**Problem**: Metrics show 0% IoU
- **Cause**: Masks don't overlap at all
- **Solution**:
  - Verify both masks are being generated
  - Check individual mask exports
  - May indicate issue with SAM2 simulation
  - Review console logs

**Problem**: Very slow processing
- **Cause**: Large images or complex processing
- **Solution**:
  - Enable "Half-Res" checkbox
  - Process smaller images
  - Close other browser tabs
  - Be patient (comparison takes longer)

**Problem**: SAM2 mask looks identical to ArUco
- **Cause**: SAM2 simulation may be too similar
- **Solution**:
  - This is expected with current simulation
  - Real SAM2 would show more differences
  - Check difference panel for subtle changes

## Technical Details

### Comparison Algorithm

1. **Get ArUco Mask**
   - Uses same function as Task 4
   - Extracts binary mask from ArUco segmentation

2. **Get SAM2 Mask**
   - Simulated SAM2 segmentation
   - Uses ArUco points as prompts
   - Applies edge-aware refinement
   - Produces smoother boundaries

3. **Calculate Metrics**
   - **IoU**: Intersection / Union
   - **Areas**: Pixel counts using countNonZero
   - **Difference**: Absolute difference between masks

4. **Create Visualization**
   - Three-panel layout (ArUco | SAM2 | Diff)
   - Color-coded overlays
   - Labels for clarity

### SAM2 Simulation Details

**Current Implementation:**
- Uses ArUco marker points as prompts
- Applies edge detection for refinement
- Uses morphological operations for smoothing
- Simulates SAM2's better boundary following

**Future Enhancement:**
- Can integrate actual SAM2 ONNX model
- Requires model conversion and hosting
- Would show real SAM2 capabilities

## Export Analysis

### What to Export
1. **Comparison Images**: Three-panel visualizations
2. **Individual Masks**: For detailed analysis
3. **Metrics**: Check console for numerical values

### Analysis Tips
1. **Compare Across Images**: Look for consistent patterns
2. **Check IoU Values**: Track agreement across dataset
3. **Examine Differences**: Understand where methods diverge
4. **Document Findings**: Note observations for report

## Expected Output Files

For each input image:
- `IMG_01__compare_sam2.png` - Three-panel comparison
- `IMG_01__aruco_mask_task5.png` - ArUco mask
- `IMG_01__sam2_mask_task5.png` - SAM2 mask

## Tips for Best Results

1. **Complete Task 4 First**: Ensure ArUco works properly
2. **Use Consistent Images**: Same images as Task 4
3. **Check Console**: Monitor processing and metrics
4. **Export Everything**: Save all comparison results
5. **Document Metrics**: Record IoU values for analysis
6. **Compare Visually**: Look at difference panel carefully

## Evaluation Checklist

For assignment evaluation, ensure:
- ✅ Comparison works for all images
- ✅ Three-panel display shows correctly
- ✅ Metrics are calculated and displayed
- ✅ Exports work properly
- ✅ Can explain differences between methods
- ✅ Can interpret IoU and other metrics

## Best Practices

1. **Start with Task 4**: Verify ArUco works first
2. **Use Multiple Images**: Compare across dataset
3. **Analyze Metrics**: Understand what they mean
4. **Visual Inspection**: Examine difference panels
5. **Document Results**: Keep notes on findings
6. **Export Consistently**: Use same export settings

## Understanding SAM2 (Conceptual)

**SAM2 (Segment Anything Model 2):**
- Advanced deep learning segmentation model
- Developed by Meta AI
- Can segment objects from prompts (points, boxes)
- Produces high-quality segmentation masks
- Better at following object boundaries

**Why Compare:**
- ArUco: Marker-based, requires physical markers
- SAM2: AI-based, works from prompts
- Comparison shows strengths/weaknesses
- Helps understand trade-offs

## Notes

- Current implementation uses **simulated SAM2** for demonstration
- Real SAM2 integration requires ONNX model setup
- Comparison framework is ready for real SAM2
- Metrics and visualization work with any segmentation method

