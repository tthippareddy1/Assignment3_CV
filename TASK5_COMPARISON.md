# Task 5: ArUco vs SAM2 Segmentation Comparison

## What is This Task?

Task 5 compares two different methods for segmenting objects:
1. **ArUco markers** (from Task 4) - Uses physical markers placed around objects
2. **SAM2** (Segment Anything Model 2) - An advanced AI model that can segment objects

**Why compare them?** Each method has strengths and weaknesses. This comparison helps you understand:
- Which method works better for your images
- What are the trade-offs between marker-based and AI-based approaches
- How accurate each method is compared to the other

**Think of it like:** Comparing two different ways to draw a boundary around a property - one using physical markers, one using satellite imagery analysis.

## Prerequisites

**⚠️ IMPORTANT:** Task 5 requires Task 4 to work first!

Before starting Task 5:
1. ✅ Make sure Task 4 works correctly
2. ✅ Verify ArUco markers are detected in your images
3. ✅ Use the same images for Task 5 as you used for Task 4

## Step-by-Step Guide

### Step 1: Verify Task 4 Works

**Don't skip this step!** Task 5 depends on Task 4.

1. Load an image with ArUco markers
2. Select **"Segmentation – ArUco (Task 4)"** from the mode dropdown
3. Verify you see:
   - Green overlay on the object
   - Green boundary line
   - Status showing number of points detected
4. If Task 4 doesn't work, **fix it first** before trying Task 5

### Step 2: Load Your Images

1. Open the web application (`http://localhost:8000`)
2. Wait for "OpenCV ready" message
3. Click **"Load Images"** button
4. Select images from `dataset/aruco/` folder
   - Use the **same images** you used for Task 4
   - At least 10 images recommended

### Step 3: Select Comparison Mode

1. Find the **mode dropdown** menu
2. Select **"Compare – ArUco vs SAM2 (Task 5)"**
3. Wait a few seconds - comparison takes longer than single segmentation

**What happens:**
- Computer processes ArUco segmentation (like Task 4)
- Computer processes SAM2 segmentation (simulated)
- Computer compares the two results
- Computer calculates metrics

### Step 4: Understand the Three-Panel Display

You'll see **three images side-by-side**:

#### Left Panel: ArUco Result (Green)
- **What it shows:** The segmentation from Task 4
- **Color:** Green overlay
- **Label:** "ArUco" at the top
- **Characteristics:**
  - Follows marker positions exactly
  - Polygon-based boundary
  - Sharp corners at marker locations
  - Limited by where markers were placed

#### Middle Panel: SAM2 Result (Red)
- **What it shows:** The AI-based segmentation
- **Color:** Red overlay
- **Label:** "SAM2" at the top
- **Characteristics:**
  - Smoother boundaries
  - Better at following object edges
  - May extend beyond marker positions
  - More "intelligent" boundary detection

#### Right Panel: Difference (Yellow)
- **What it shows:** Where the two methods disagree
- **Color:** Yellow highlights
- **Label:** "Diff" at the top
- **What yellow means:**
  - Yellow areas = methods produced different results
  - More yellow = more disagreement
  - Less yellow = methods agree more

### Step 5: Read the Metrics

Look at the **status bar** at the bottom. You'll see something like:

```
IoU: 85.3% | ArUco: 12543px | SAM2: 13456px | Diff: 2341px
```

**What each metric means:**

#### IoU (Intersection over Union)
- **Range:** 0% to 100%
- **What it measures:** How much the two masks overlap
- **Interpretation:**
  - **90-100%**: Excellent agreement - methods produce very similar results
  - **70-90%**: Good agreement - methods mostly agree
  - **50-70%**: Moderate agreement - some differences
  - **<50%**: Significant differences - methods disagree substantially

**Simple explanation:** If IoU is 85%, it means 85% of the segmented area is the same in both methods.

#### ArUco Area
- **What it shows:** Number of pixels in the ArUco mask
- **Example:** 12543px means 12,543 pixels were marked as "object" by ArUco
- **Use:** Compare with SAM2 area to see which method segments more/less

#### SAM2 Area
- **What it shows:** Number of pixels in the SAM2 mask
- **Example:** 13456px means 13,456 pixels were marked as "object" by SAM2
- **Use:** Compare with ArUco area

#### Difference Area
- **What it shows:** Number of pixels where the two masks disagree
- **Example:** 2341px means 2,341 pixels differ between methods
- **Use:** Lower is better (means methods agree more)

### Step 6: Check Browser Console (Optional but Helpful)

1. Open **Developer Tools** (F12 or Cmd+Option+I)
2. Go to **Console** tab
3. Look for messages like:

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

**What to look for:**
- ✅ Both masks found successfully
- ✅ Metrics calculated
- ❌ Any error messages

### Step 7: Adjust Settings (Optional)

You can adjust Task 4 settings and see how it affects the comparison:

**Dictionary:**
- Make sure correct dictionary is selected
- Affects ArUco detection, which affects comparison

**Use Corners:**
- Toggle to see how it changes comparison
- More corners = smoother ArUco boundary = potentially better match with SAM2

**Dilate:**
- Adjust mask expansion
- May help ArUco match SAM2 better

**Note:** Changes update automatically when you adjust settings.

### Step 8: Save Single Comparison

1. Click **"Save PNG"** button
2. File downloads as: `{image_name}__compare_sam2.png`
3. This saves the **entire three-panel comparison** image

### Step 9: Export All Images (Batch Processing)

1. Make sure all your images are loaded
2. Click **"Export All (Task 5)"** button
3. Browser downloads **three files** for each image:
   - `{name}__compare_sam2.png` - Three-panel comparison (main result)
   - `{name}__aruco_mask_task5.png` - ArUco mask only
   - `{name}__sam2_mask_task5.png` - SAM2 mask only

**Note:** Browser may ask permission for multiple downloads.

## Understanding the Results

### What to Look For

**Good Comparison:**
- ✅ Both panels show reasonable segmentation
- ✅ IoU is 60-90% (methods mostly agree)
- ✅ Difference panel shows minimal yellow (mostly in edge areas)
- ✅ Boundaries look smooth and accurate

**Expected Differences:**
- SAM2 boundaries are usually **smoother** than ArUco
- SAM2 may extend **slightly beyond** marker positions
- SAM2 better follows **object edges** (not just marker positions)
- ArUco is more **precise** at marker locations

### Why Differences Occur

**ArUco Limitations:**
- Can only segment where markers are placed
- Boundary is limited to connecting marker points
- Sharp corners at marker locations
- Requires physical markers

**SAM2 Advantages:**
- Can extend beyond prompt points
- Smoother boundary following
- Better edge detection
- More "intelligent" segmentation

**Trade-offs:**
- **ArUco**: Precise but requires markers
- **SAM2**: Flexible but may be less precise at specific points

## Common Issues and Solutions

### Problem: "COMPARE – MISSING MASKS"
**What it means:** One or both segmentation methods failed.

**Solutions:**
1. **Check Task 4 first** - Make sure ArUco works
2. **Verify markers detected** - Check console for detection messages
3. **Check dictionary** - Ensure correct dictionary selected
4. **Improve image quality** - Better lighting, focus, marker visibility

### Problem: White/Blank Comparison Image
**What it means:** Image processing error occurred.

**Solutions:**
1. **Check browser console** - Look for error messages
2. **Try smaller image** - Enable "Half-Res" checkbox
3. **Refresh page** - Sometimes fixes temporary issues
4. **Verify both masks generated** - Check console logs

### Problem: IoU Shows 0%
**What it means:** Masks don't overlap at all (very unusual).

**Solutions:**
1. **Verify both masks exist** - Check individual mask exports
2. **Check SAM2 simulation** - May indicate issue with SAM2 processing
3. **Review console logs** - Look for processing errors
4. **Try different image** - May be image-specific issue

### Problem: Very Slow Processing
**What it means:** Comparison takes longer than single segmentation.

**Solutions:**
1. **This is normal** - Comparison processes two methods
2. **Enable "Half-Res"** - Reduces processing time
3. **Process smaller images** - Fewer pixels = faster
4. **Be patient** - Can take 5-10 seconds per image

### Problem: SAM2 Looks Identical to ArUco
**What it means:** Current implementation uses simulated SAM2.

**Solutions:**
1. **This is expected** - Real SAM2 would show more differences
2. **Check difference panel** - May show subtle differences
3. **Note in report** - Mention that SAM2 is simulated
4. **Real SAM2** - Would require ONNX model integration

## Understanding the Metrics (Detailed)

### IoU Calculation

**Formula:**
```
IoU = (Area where both masks agree) / (Total area covered by either mask)
```

**Visual Example:**
- Imagine two overlapping circles
- **Intersection** = area where both circles overlap
- **Union** = total area covered by both circles combined
- **IoU** = intersection ÷ union

**Why it matters:**
- Higher IoU = methods agree more = more reliable segmentation
- Lower IoU = methods disagree = need to investigate why

### Area Comparison

**ArUco Area vs SAM2 Area:**
- **Similar sizes**: Methods agree on object size
- **ArUco larger**: ArUco segmented more area
- **SAM2 larger**: SAM2 segmented more area (common - SAM2 extends beyond markers)

**Difference Area:**
- Shows how many pixels differ
- Lower = better agreement
- Higher = more disagreement

## Tips for Best Results

1. **Complete Task 4 first** - Don't skip this step!
2. **Use same images** - Use identical images for both tasks
3. **Check console** - Monitor processing and catch errors early
4. **Export everything** - Save all comparison results
5. **Document metrics** - Record IoU values for your report
6. **Compare visually** - Look at difference panel carefully
7. **Try different settings** - See how parameters affect comparison

## What Makes This Implementation Good?

✅ **Visual Comparison**: Easy to see differences side-by-side  
✅ **Quantitative Metrics**: Numbers to measure agreement  
✅ **Difference Highlighting**: Shows exactly where methods differ  
✅ **Batch Processing**: Compare all images at once  
✅ **Error Handling**: Clear messages when something goes wrong  
✅ **Flexible**: Works with any segmentation method  

## Evaluation Checklist

Before submitting, ensure:

- ✅ Task 4 works correctly first
- ✅ Comparison works for all images
- ✅ Three-panel display shows correctly
- ✅ Metrics are calculated and displayed
- ✅ Exports work properly
- ✅ Can explain differences between methods
- ✅ Can interpret IoU and other metrics
- ✅ Understand trade-offs between methods

## Real-World Applications

**When to use ArUco:**
- Need precise control over segmentation points
- Have physical access to place markers
- Want consistent, repeatable results
- Working with known objects

**When to use SAM2:**
- Can't place physical markers
- Need smooth, edge-aware boundaries
- Working with unknown objects
- Want AI-powered segmentation

**When to use both:**
- Compare accuracy
- Validate results
- Understand limitations
- Choose best method for your application

## Technical Notes

### Current SAM2 Implementation

**What it is:**
- **Simulated SAM2** - Not the real SAM2 model
- Uses ArUco points as prompts
- Applies edge detection for refinement
- Simulates SAM2's smoother boundaries

**Why simulated:**
- Real SAM2 requires:
  - ONNX model conversion
  - Model hosting
  - Additional setup
- Simulation demonstrates comparison framework
- Metrics and visualization work the same way

**Future enhancement:**
- Can integrate real SAM2 ONNX model
- Would show actual SAM2 capabilities
- Comparison framework is ready

### How Comparison Works

1. **Get ArUco mask** - Uses same function as Task 4
2. **Get SAM2 mask** - Processes image with SAM2 (simulated)
3. **Calculate metrics** - Computes IoU, areas, differences
4. **Create visualization** - Three-panel side-by-side display
5. **Add labels** - Clear labels for each panel

## Summary

Task 5 compares ArUco marker-based segmentation (Task 4) with SAM2 AI-based segmentation. The comparison shows:
- **Visual differences** between methods
- **Quantitative metrics** (IoU, areas, differences)
- **Where methods agree/disagree**

**Key takeaway:** Each method has strengths. ArUco is precise but requires markers. SAM2 is flexible but may be less precise. The comparison helps you understand which method works better for your specific use case.

**Remember:** Always complete Task 4 successfully before attempting Task 5!
