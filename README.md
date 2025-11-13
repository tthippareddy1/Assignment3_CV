# Computer Vision Assignment 3 - Web Application Documentation

## Overview
This web application implements 5 computer vision tasks for object segmentation and analysis. Each task builds upon previous ones, creating a comprehensive computer vision toolkit.

## Quick Start

1. **Open the Application**
   - Navigate to `webapp/` directory
   - Start a local web server: `python3 -m http.server 8000`
   - Open browser to `http://localhost:8000`
   - Wait for "OpenCV ready" message

2. **Load Images**
   - Click "Load Images" button
   - Select images from your dataset
   - Or use webcam to capture images

3. **Select Task Mode**
   - Use the mode dropdown to select desired task
   - Adjust parameters as needed
   - View results in real-time

4. **Export Results**
   - Click "Save PNG" for single image
   - Click "Export All (Task X)" for batch processing

## Task Documentation

### [Task 1: Gradient and LoG](./TASK1_GRADIENT.md)
**Gradient Magnitude, Gradient Angle, and Laplacian of Gaussian**

- Compute gradient images (magnitude and angle)
- Apply Laplacian of Gaussian filtering
- Visualize edge information

**Key Features:**
- Gradient magnitude visualization
- HSV color-coded gradient angles
- LoG edge detection

---

### [Task 2: Edge and Corner Detection](./TASK2_EDGES_CORNERS.md)
**Custom Edge Detection and Harris Corner Detection**

- Implement edge detection with NMS and hysteresis
- Detect corners using Harris corner detector
- Visualize detected features

**Key Features:**
- Custom Canny-like edge detection
- Harris corner detection with heatmap
- Adjustable thresholds and parameters

---

### [Task 3: Object Boundary Detection](./TASK3_BOUNDARY.md)
**Contour-Based Boundary Detection**

- Find exact object boundaries using contours
- Smart contour selection algorithm
- Polygon approximation and mask generation

**Key Features:**
- Automatic contour selection
- Polygon simplification
- Binary mask generation

---

### [Task 4: ArUco Marker Segmentation](./TASK4_ARUCO.md)
**Marker-Based Object Segmentation**

- Detect ArUco markers on object boundaries
- Create segmentation mask from marker positions
- Support for multiple ArUco dictionaries

**Key Features:**
- Multiple dictionary support
- Corner or center point collection
- Flexible mask refinement

**Prerequisites:**
- ArUco markers printed and placed on objects
- Images with visible markers

---

### [Task 5: ArUco vs SAM2 Comparison](./TASK5_COMPARISON.md)
**Segmentation Method Comparison**

- Compare ArUco (Task 4) with SAM2 segmentation
- Visual side-by-side comparison
- Quantitative metrics (IoU, areas, differences)

**Key Features:**
- Three-panel comparison view
- Intersection over Union (IoU) metrics
- Difference visualization

**Prerequisites:**
- Task 4 must work first
- Same images as Task 4

## File Structure

```
assignment3/
├── README.md                    # Main documentation (this file)
├── TASK1_GRADIENT.md            # Task 1 documentation
├── TASK2_EDGES_CORNERS.md       # Task 2 documentation
├── TASK3_BOUNDARY.md            # Task 3 documentation
├── TASK4_ARUCO.md              # Task 4 documentation
├── TASK5_COMPARISON.md         # Task 5 documentation
│
├── webapp/                      # Web application directory
│   ├── index.html               # Main HTML file
│   ├── app.js                   # Application logic (all 5 tasks)
│   ├── styles.css               # Styling
│   └── build_wasm/              # OpenCV.js build with ArUco support
│       └── bin/
│           ├── opencv.js        # OpenCV.js library (local build)
│           ├── opencv_js.js     # OpenCV.js helper
│           └── loader.js        # OpenCV.js loader
│
├── dataset/                     # Image datasets
│   ├── base/                    # General images for Tasks 1-3
│   │   └── IMG_01.JPG ...      # Base dataset images
│   └── aruco/                   # ArUco marker images for Tasks 4-5
│       ├── singlemarkersoriginal.jpg
│       ├── marker42.png
│       └── ...                  # ArUco marker images
│
├── outputs/                     # Exported results
│   └── task1/                   # Task 1 outputs
│       ├── grad_mag/            # Gradient magnitude results
│       ├── grad_angle/          # Gradient angle results
│       └── log/                 # LoG results
│
└── backup/                      # Backup files (previous versions)
    ├── task1/
    ├── task2/
    ├── task3/
    └── task4/
```

## Common Workflow

### For Task 1-3 (General Images)
1. Load images from `dataset/base/`
2. Select task mode
3. Adjust parameters
4. Export results

### For Task 4-5 (ArUco Images)
1. Prepare images with ArUco markers
2. Load images from `dataset/aruco/`
3. Configure ArUco dictionary
4. Run Task 4 first
5. Then run Task 5 for comparison

## Browser Requirements

- **Chrome/Edge**: Recommended (best performance)
- **Firefox**: Supported
- **Safari**: Supported (may have limitations)
- **Mobile**: Limited support (desktop recommended)

## System Requirements

- Modern web browser with JavaScript enabled
- **Local web server required** (OpenCV.js files must be served, not opened via file://)
- Python 3 (for `python3 -m http.server`) or Node.js (for `npx http-server`)
- Sufficient RAM (for image processing)
- Recommended: 4GB+ RAM for large images

## Running the Application

**Important:** You must run a local web server. Opening `index.html` directly via `file://` protocol will not work due to CORS restrictions and OpenCV.js loading requirements.

### Quick Start:
```bash
cd webapp
python3 -m http.server 8000
```
Then open `http://localhost:8000` in your browser.

### Alternative (Node.js):
```bash
cd webapp
npx http-server -p 8000
```

## Troubleshooting

### OpenCV Not Loading
- **Verify local build exists**: Check `webapp/build_wasm/bin/opencv.js` file exists
- **Check file paths**: Ensure `index.html` loads from correct path
- **Wait longer**: First load takes time (OpenCV.js is large)
- **Check browser console**: Look for 404 errors or loading messages
- **Verify server**: Make sure you're running a local web server (not file://)
- **Hard refresh**: Clear cache with Ctrl+Shift+R (Cmd+Shift+R on Mac)

### Images Not Loading
- Use JPG or PNG format
- Avoid HEIC/HEIF (convert first)
- Check file size (very large may fail)
- Try smaller images

### Slow Performance
- Enable "Half-Res" checkbox
- Use smaller images
- Close other browser tabs
- Wait for processing to complete

### ArUco Not Working
- Verify OpenCV.js includes ArUco module
- Check marker visibility
- Ensure correct dictionary
- Improve lighting/focus

## Quick Reference

### Keyboard Shortcuts
- None currently (mouse/touch interface)

### Status Bar Indicators
- **"OpenCV ready"**: System ready
- **"Image X/Y"**: Current image in dataset
- **"Mode: ..."**: Current processing mode
- **Metrics**: Task-specific statistics

### Export Naming
- Format: `{original_name}__{mode}.png`
- Examples:
  - `IMG_01__grad_mag.png`
  - `IMG_01__edge.png`
  - `IMG_01__boundary.png`
  - `IMG_01__aruco_mask.png`
  - `IMG_01__compare_sam2.png`

## Tips for Best Results

1. **Start Simple**: Begin with Task 1, work through sequentially
2. **Use Good Images**: Well-lit, clear boundaries work best
3. **Adjust Parameters**: Defaults work, but tuning improves results
4. **Check Console**: Browser console shows errors and debug info
5. **Export Regularly**: Save results as you work
6. **Test Incrementally**: Verify each task before moving to next

## Getting Help

1. **Check Documentation**: Read task-specific .md files
2. **Check Console**: Browser console shows errors
3. **Try Defaults**: Reset parameters to defaults
4. **Simplify**: Try with simpler images first
5. **Test Mode**: Use "Load Test Image" for quick verification

## Features Summary

✅ **Task 1**: Gradient computation and LoG filtering  
✅ **Task 2**: Edge and corner detection  
✅ **Task 3**: Object boundary detection  
✅ **Task 4**: ArUco marker-based segmentation  
✅ **Task 5**: ArUco vs SAM2 comparison  

✅ **Real-time Processing**: Works with webcam  
✅ **Batch Export**: Process multiple images  
✅ **Parameter Tuning**: Adjustable for all tasks  
✅ **Visual Feedback**: Real-time visualization  
✅ **Error Handling**: Graceful error messages  

## License & Credits

- **OpenCV.js**: Computer vision library
- **SAM2**: Meta AI's Segment Anything Model 2 (simulated)
- **ArUco**: Augmented Reality marker system

---

**For detailed instructions, see individual task documentation files.**

