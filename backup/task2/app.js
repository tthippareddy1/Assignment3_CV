// ---------- Element refs ----------
const els = {
  fileInput:   document.getElementById('fileInput'),
  loadDemo:    document.getElementById('loadDemo'),
  camStart:    document.getElementById('camStart'),
  camCapture:  document.getElementById('camCapture'),
  camStop:     document.getElementById('camStop'),
  prev:        document.getElementById('prev'),
  next:        document.getElementById('next'),
  mode:        document.getElementById('mode'),
  edgeBoost:   document.getElementById('edgeBoost'),
  halfRes:     document.getElementById('halfRes'),
  // (optional controls—OK if null)
  edgeLow:     document.getElementById('edgeLow'),
  edgeHigh:    document.getElementById('edgeHigh'),
  edgeAuto:    document.getElementById('edgeAuto'),
  edgeBinary:  document.getElementById('edgeBinary'),
  hWin:        document.getElementById('hWin'),
  hK:          document.getElementById('hK'),
  hTh:         document.getElementById('hTh'),
  hHeat:       document.getElementById('hHeat'),

  save:        document.getElementById('save'),
  exportAll1:  document.getElementById('exportAll1'),
  exportAll2:  document.getElementById('exportAll2'),
  video:       document.getElementById('video'),
  canvas:      document.getElementById('canvas'),
  status:      document.getElementById('status')
};

// ---------- App state ----------
let state = { images: [], index: 0, stream: null, source: 'none', mode: 'original' };

// ---------- OpenCV readiness ----------
let cvReady = false;
(function hookOpenCV() {
  function markReady() { cvReady = true; setStatus('OpenCV ready.'); }
  if (window.cv) {
    if (typeof cv.Mat === 'function') { markReady(); }
    else { cv.onRuntimeInitialized = () => { markReady(); renderCurrent(); }; }
  } else {
    const intv = setInterval(() => {
      if (window.cv) {
        clearInterval(intv);
        if (typeof cv.Mat === 'function') { markReady(); renderCurrent(); }
        else { cv.onRuntimeInitialized = () => { markReady(); renderCurrent(); }; }
      }
    }, 50);
  }
})();
async function waitForCV() {
  if (cvReady && window.cv && typeof cv.Mat === 'function') return;
  setStatus('Loading OpenCV…');
  await new Promise((resolve) => {
    const t0 = Date.now();
    const intv = setInterval(() => {
      if (cvReady && window.cv && typeof cv.Mat === 'function') { clearInterval(intv); resolve(); }
      else if (Date.now() - t0 > 15000) { clearInterval(intv); setStatus('Error: OpenCV failed to load.'); resolve(); }
    }, 50);
  });
}

// ---------- UI helpers ----------
function setStatus(msg) { els.status.textContent = msg; }
function enableNav(enable) {
  els.prev.disabled = els.next.disabled = !enable;
  els.save.disabled = !enable;
  els.exportAll1.disabled = !enable || !state.images.length;
  els.exportAll2.disabled = !enable || !state.images.length; // NEW

}
function showVideo(show) { els.video.style.display = show ? 'block' : 'none'; }
function showCanvas(show) { els.canvas.style.display = show ? 'block' : 'none'; }

// ---------- Robust image loading ----------
const MAX_SIDE = 4096; // prevent huge GPU bitmaps
async function fileToBitmap(file) {
  if (/hei[cf]/i.test(file.type) || /\.heic$|\.heif$/i.test(file.name)) {
    throw new Error('HEIC/HEIF is not supported in browsers. Please convert to JPG/PNG.');
  }
  try {
    return await createImageBitmap(file);
  } catch (_) {
    const url = URL.createObjectURL(file);
    try {
      const img = await new Promise((resolve, reject) => {
        const el = new Image();
        el.onload = () => resolve(el);
        el.onerror = () => reject(new Error('Image decode failed'));
        el.src = url;
      });
      const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
      const w = Math.max(1, Math.round(img.naturalWidth * scale));
      const h = Math.max(1, Math.round(img.naturalHeight * scale));
      const c = document.createElement('canvas');
      c.width = w; c.height = h;
      c.getContext('2d').drawImage(img, 0, 0, w, h);
      return await createImageBitmap(c);
    } finally { URL.revokeObjectURL(url); }
  }
}
async function filesToBitmaps(files) {
  const out = [], errors = [];
  for (const f of files) {
    try { out.push({ name: f.name, bitmap: await fileToBitmap(f) }); }
    catch (e) { errors.push(`${f.name}: ${e.message}`); }
  }
  if (errors.length) console.warn('Some files failed to load:\n' + errors.join('\n'));
  if (!out.length) throw new Error('No images loaded. ' + (errors[0] || 'Unknown error.'));
  return out;
}

// ---------- Canvas helpers ----------
function drawBitmapToCanvas(bmp) {
  const ctx = els.canvas.getContext('2d');
  els.canvas.width = bmp.width; els.canvas.height = bmp.height;
  ctx.drawImage(bmp, 0, 0);
  overlayModeLabel('ORIGINAL');
}
function getSrcMatFromBitmap(bmp) {
  const oc = document.createElement('canvas');
  oc.width = bmp.width; oc.height = bmp.height;
  oc.getContext('2d').drawImage(bmp, 0, 0);
  return cv.imread(oc); // RGBA Mat
}
function overlayModeLabel(text) {
  const ctx = els.canvas.getContext('2d');
  ctx.save();
  ctx.font = 'bold 14px system-ui, -apple-system, Segoe UI, Roboto, sans-serif';
  const w = ctx.measureText(text).width + 16;
  ctx.fillStyle = 'rgba(0,0,0,0.4)';
  ctx.fillRect(10, 10, w, 26);
  ctx.fillStyle = '#e2e8f0';
  ctx.fillText(text, 18, 28);
  ctx.restore();
}

// ---------- Params ----------
const params = { sobelK: 3, gaussK: 5, gaussSigma: 1.0, laplaceK: 3 };

// ---------- Helpers ----------
function toGray(src) { const g = new cv.Mat(); cv.cvtColor(src, g, cv.COLOR_RGBA2GRAY); return g; }
function imshowFit(mat) { els.canvas.width = mat.cols; els.canvas.height = mat.rows; cv.imshow(els.canvas, mat); }
function maybeResize(src) {
  if (!els.halfRes || !els.halfRes.checked) return src.clone();
  const dst = new cv.Mat();
  cv.resize(src, dst, new cv.Size(Math.round(src.cols/2), Math.round(src.rows/2)), 0, 0, cv.INTER_AREA);
  return dst;
}
function percentileFrom8U(mat8, p /*0..100*/) {
  const hist = new Array(256).fill(0);
  const d = mat8.data;
  for (let i = 0; i < d.length; i++) hist[d[i]]++;
  const target = (p / 100) * d.length;
  let cum = 0;
  for (let v = 0; v < 256; v++) { cum += hist[v]; if (cum >= target) return v; }
  return 255;
}

// ---------- Task 1 compute ----------
function gradMagnitudeMat(src) {
  const work = maybeResize(src);
  const gray = toGray(work);
  const dx = new cv.Mat(), dy = new cv.Mat();
  const mag = new cv.Mat(), mag8 = new cv.Mat();

  cv.GaussianBlur(gray, gray, new cv.Size(3,3), 0.8, 0.8);
  cv.Sobel(gray, dx, cv.CV_32F, 1, 0, params.sobelK, 1, 0, cv.BORDER_DEFAULT);
  cv.Sobel(gray, dy, cv.CV_32F, 0, 1, params.sobelK, 1, 0, cv.BORDER_DEFAULT);
  cv.magnitude(dx, dy, mag);
  cv.normalize(mag, mag, 0, 255, cv.NORM_MINMAX);
  mag.convertTo(mag8, cv.CV_8U);

  if (els.edgeBoost && +els.edgeBoost.value > 0) {
    const kernel = cv.Mat.ones(3,3,cv.CV_8U);
    cv.dilate(mag8, mag8, kernel);
    kernel.delete();
    if (+els.edgeBoost.value === 2) cv.equalizeHist(mag8, mag8);
  }

  gray.delete(); dx.delete(); dy.delete(); mag.delete(); work.delete();
  return mag8;
}
function gradAngleMat(src) {
  const work = maybeResize(src);
  const gray = toGray(work);
  const dx = new cv.Mat(), dy = new cv.Mat();
  const mag = new cv.Mat(), ang = new cv.Mat();
  const mag8 = new cv.Mat(), angH = new cv.Mat();

  cv.GaussianBlur(gray, gray, new cv.Size(3,3), 0.8, 0.8);
  cv.Sobel(gray, dx, cv.CV_32F, 1, 0, params.sobelK, 1, 0, cv.BORDER_DEFAULT);
  cv.Sobel(gray, dy, cv.CV_32F, 0, 1, params.sobelK, 1, 0, cv.BORDER_DEFAULT);
  cv.cartToPolar(dx, dy, mag, ang, true); // degrees

  cv.normalize(mag, mag, 0, 255, cv.NORM_MINMAX);
  mag.convertTo(mag8, cv.CV_8U);

  const mask = new cv.Mat();
  cv.threshold(mag8, mask, 20, 255, cv.THRESH_BINARY);
  if (els.edgeBoost && +els.edgeBoost.value > 0) {
    const kernel = cv.Mat.ones(3,3,cv.CV_8U);
    cv.dilate(mask, mask, kernel);
    kernel.delete();
  }

  ang.convertTo(angH, cv.CV_8U, 0.5); // 0..360 -> 0..180
  const hueMax = new cv.Mat(angH.rows, angH.cols, cv.CV_8U); hueMax.setTo(new cv.Scalar(179));
  cv.min(angH, hueMax, angH);
  hueMax.delete();

  const sat = new cv.Mat(gray.rows, gray.cols, cv.CV_8U); sat.setTo(new cv.Scalar(255));
  const val = new cv.Mat(); cv.bitwise_and(mag8, mask, val);

  const mv = new cv.MatVector();
  mv.push_back(angH); mv.push_back(sat); mv.push_back(val);
  const hsv = new cv.Mat(); cv.merge(mv, hsv);
  const rgb = new cv.Mat(); cv.cvtColor(hsv, rgb, cv.COLOR_HSV2RGB);

  work.delete(); gray.delete(); dx.delete(); dy.delete();
  mag.delete(); ang.delete(); mag8.delete(); angH.delete();
  sat.delete(); val.delete(); mask.delete(); mv.delete(); hsv.delete();

  return rgb;
}
function logMat(src) {
  const work = maybeResize(src);
  const gray = toGray(work);
  const blur = new cv.Mat();
  const lap = new cv.Mat(), lap8 = new cv.Mat();

  cv.GaussianBlur(gray, blur, new cv.Size(params.gaussK, params.gaussK), params.gaussSigma, params.gaussSigma, cv.BORDER_DEFAULT);
  cv.Laplacian(blur, lap, cv.CV_32F, params.laplaceK, 1, 0, cv.BORDER_DEFAULT);
  cv.convertScaleAbs(lap, lap8);

  if (els.edgeBoost && +els.edgeBoost.value > 0) {
    const kernel = cv.Mat.ones(3,3,cv.CV_8U);
    cv.dilate(lap8, lap8, kernel);
    kernel.delete();
    if (+els.edgeBoost.value === 2) cv.equalizeHist(lap8, lap8);
  }

  work.delete(); gray.delete(); blur.delete(); lap.delete();
  return lap8;
}

// ---------- Task 2: Edge – NMS + Hysteresis (robust, no raw array access) ----------
function edgeSimple(src, returnBinary = true) {
  const work = maybeResize(src);
  const gray = toGray(work);
  cv.GaussianBlur(gray, gray, new cv.Size(3,3), 0.8, 0.8);

  // Gradients
  const dx = new cv.Mat(), dy = new cv.Mat(), mag = new cv.Mat(), ang = new cv.Mat();
  cv.Sobel(gray, dx, cv.CV_32F, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT);
  cv.Sobel(gray, dy, cv.CV_32F, 0, 1, 3, 1, 0, cv.BORDER_DEFAULT);
  // cartToPolar gives BOTH magnitude and angle (avoid cv.phase variations)
  cv.cartToPolar(dx, dy, mag, ang, true); // angle in degrees

  const rows = mag.rows, cols = mag.cols;

  // -------- Non-Maximum Suppression (4 dirs) on float matrices --------
  const nms = cv.Mat.zeros(rows, cols, cv.CV_32F);
  const M = mag.data32F, A = ang.data32F, N = nms.data32F;

  for (let y = 1; y < rows - 1; y++) {
    const r = y * cols;
    for (let x = 1; x < cols - 1; x++) {
      const i = r + x;
      let a = A[i];
      if (a < 0) a += 180;
      if (a >= 180) a -= 180;

      let o1 = 0, o2 = 0;
      if (a < 22.5 || a >= 157.5)      { o1 = -1;           o2 = +1; }
      else if (a < 67.5)               { o1 = -cols - 1;    o2 = +cols + 1; }
      else if (a < 112.5)              { o1 = -cols;        o2 = +cols; }
      else                             { o1 = -cols + 1;    o2 = +cols - 1; }

      const m = M[i];
      if (m >= M[i + o1] && m >= M[i + o2]) N[i] = m;
    }
  }

  // Normalize to 8U for thresholding
  const nms8 = new cv.Mat();
  cv.normalize(nms, nms, 0, 255, cv.NORM_MINMAX);
  nms.convertTo(nms8, cv.CV_8U);

  // -------- Hysteresis via morphology (no manual array walking) --------
  // Auto thresholds (percentiles) if toggle is missing or checked
  let high = 40, low = 15;
  const auto = (!els.edgeAuto || els.edgeAuto.checked);
  if (auto) {
    high = percentileFrom8U(nms8, 85);
    low  = Math.max(5, Math.round(0.4 * high));
    if (els.edgeHigh) els.edgeHigh.value = String(high);
    if (els.edgeLow)  els.edgeLow.value  = String(low);
  } else {
    high = +els.edgeHigh.value; low = +els.edgeLow.value;
  }

  // strong = nms8 >= high;  weakOnly = low<=nms8<high
  const strong = new cv.Mat(), weak = new cv.Mat(), weakOnly = new cv.Mat();
  cv.threshold(nms8, strong, high, 255, cv.THRESH_BINARY);
  cv.threshold(nms8, weak,   low,  255, cv.THRESH_BINARY);
  cv.subtract(weak, strong, weakOnly); // remove strong from weak

  // propagate: iteratively add weak neighbors touching strong
  const connected = strong.clone();
  const kernel = cv.Mat.ones(3,3,cv.CV_8U);
  const dil = new cv.Mat(), add = new cv.Mat();

  for (let it = 0; it < 12; it++) { // few iterations are enough
    cv.dilate(connected, dil, kernel);
    cv.bitwise_and(dil, weakOnly, add);
    const nz = cv.countNonZero(add);
    if (nz === 0) break;
    cv.bitwise_or(connected, add, connected);
    cv.subtract(weakOnly, add, weakOnly);
  }

  // final edges
  const edges = connected; // already 0/255

  // Optional: thicken a bit so it’s visible on high-res photos
  const kernel2 = cv.Mat.ones(3,3,cv.CV_8U);
  cv.dilate(edges, edges, kernel2);
  kernel2.delete();

  // Output image
  let out;
  if (returnBinary || (els.edgeBinary && els.edgeBinary.checked)) {
    out = new cv.Mat();
    cv.cvtColor(edges, out, cv.COLOR_GRAY2RGB);
  } else {
    const base = new cv.Mat();
    cv.cvtColor(work, base, cv.COLOR_RGBA2RGB);
    const red = new cv.Mat(base.rows, base.cols, cv.CV_8UC3, new cv.Scalar(255,0,0,0));
    const edgeRGB = new cv.Mat();
    red.copyTo(edgeRGB, edges);
    out = new cv.Mat();
    cv.addWeighted(base, 0.6, edgeRGB, 1.0, 0, out);
    base.delete(); red.delete(); edgeRGB.delete();
  }

  // cleanup
  gray.delete(); dx.delete(); dy.delete(); mag.delete(); ang.delete();
  nms.delete(); nms8.delete(); strong.delete(); weak.delete(); weakOnly.delete();
  kernel.delete(); dil.delete(); add.delete(); work.delete();

  return { out, count: cv.countNonZero(edges) };
}















// ---------- Corner – Harris (robust + automatic Shi–Tomasi fallback) ----------
function cornersHarris(src) {
  try {
    const work = maybeResize(src);
    const gray = toGray(work);
    cv.GaussianBlur(gray, gray, new cv.Size(3,3), 0.8, 0.8);

    // UI params with safe defaults
    const blockSize = (els.hWin ? +els.hWin.value : 5);    // odd window
    const k = (els.hK ? (+els.hK.value)/100.0 : 0.04);     // Harris k
    let th = (els.hTh ? +els.hTh.value : 70);              // response threshold (0–255)

    // Gradients
    const Ix = new cv.Mat(), Iy = new cv.Mat();
    cv.Sobel(gray, Ix, cv.CV_32F, 1, 0, 3, 1, 0);
    cv.Sobel(gray, Iy, cv.CV_32F, 0, 1, 3, 1, 0);

    // Structure tensor entries
    const Ixx = new cv.Mat(), Iyy = new cv.Mat(), Ixy = new cv.Mat();
    cv.multiply(Ix, Ix, Ixx);
    cv.multiply(Iy, Iy, Iyy);
    cv.multiply(Ix, Iy, Ixy);
    Ix.delete(); Iy.delete();

    const win = new cv.Size(blockSize, blockSize);
    const Sxx = new cv.Mat(), Syy = new cv.Mat(), Sxy = new cv.Mat();
    cv.GaussianBlur(Ixx, Sxx, win, 1.0);
    cv.GaussianBlur(Iyy, Syy, win, 1.0);
    cv.GaussianBlur(Ixy, Sxy, win, 1.0);
    Ixx.delete(); Iyy.delete(); Ixy.delete();

    // Harris response R = det(M) - k * trace(M)^2
    const det = new cv.Mat(), tmp = new cv.Mat();
    cv.multiply(Sxx, Syy, det);  // Sxx*Syy
    cv.multiply(Sxy, Sxy, tmp);  // Sxy^2
    cv.subtract(det, tmp, det);  // det -= Sxy^2
    const trace = new cv.Mat(), trace2 = new cv.Mat();
    cv.add(Sxx, Syy, trace);
    cv.multiply(trace, trace, trace2);

    const R = new cv.Mat();
    cv.addWeighted(det, 1.0, trace2, -k, 0.0, R);
    Sxx.delete(); Syy.delete(); Sxy.delete(); det.delete(); tmp.delete(); trace.delete(); trace2.delete();

    // Normalize 0..255 (8U) for thresholding
    const Rn = new cv.Mat(); cv.normalize(R, Rn, 0, 255, cv.NORM_MINMAX, cv.CV_32F);
    const R8 = new cv.Mat(); Rn.convertTo(R8, cv.CV_8U);
    R.delete(); Rn.delete();

    // Prepare output
    const out = new cv.Mat(); cv.cvtColor(work, out, cv.COLOR_RGBA2RGB);
    const showHeat = (!els.hHeat || els.hHeat.checked);
    if (showHeat) {
      const eq = new cv.Mat(), heat = new cv.Mat();
      cv.equalizeHist(R8, eq);
      cv.cvtColor(eq, heat, cv.COLOR_GRAY2RGB);
      cv.addWeighted(out, 0.45, heat, 0.55, 0, out);
      eq.delete(); heat.delete();
    }

    // Threshold + 3x3 local maxima
    const thMask = new cv.Mat(); cv.threshold(R8, thMask, th, 255, cv.THRESH_BINARY);
    const dil = new cv.Mat(); const ker = cv.Mat.ones(3,3,cv.CV_8U);
    cv.dilate(R8, dil, ker); ker.delete();
    const eq = new cv.Mat(); cv.compare(R8, dil, eq, cv.CMP_EQ);
    const maxima = new cv.Mat(); cv.bitwise_and(eq, thMask, maxima);

    // Corner coordinates
    const pts = new cv.Mat(); cv.findNonZero(maxima, pts);
    let count = pts.rows || 0;

    // If none, auto-lower threshold first; if still none => Shi–Tomasi fallback
    if (count === 0) {
      cv.threshold(R8, thMask, 40, 255, cv.THRESH_BINARY);
      cv.bitwise_and(eq, thMask, maxima);
      cv.findNonZero(maxima, pts);
      count = pts.rows || 0;
      if (els.hTh) els.hTh.value = '40';
    }

    if (count === 0) {
      // clean up then fallback
      R8.delete(); thMask.delete(); dil.delete(); eq.delete(); maxima.delete(); pts.delete();
      work.delete(); gray.delete();
      return cornersShiTomasi(src); // <- fallback
    }

    // Draw circles (use intPtr when data32S is unavailable)
    const color = new cv.Scalar(0, 255, 0, 255);
    const useInt = (typeof pts.data32S === 'undefined');
    for (let i = 0; i < count; i++) {
      let x, y;
      if (useInt) { const p = pts.intPtr(i, 0); x = p[0]; y = p[1]; }
      else { x = pts.data32S[i*2]; y = pts.data32S[i*2 + 1]; }
      cv.circle(out, new cv.Point(x, y), 4, color, 2, cv.LINE_AA);
    }

    // Cleanup
    R8.delete(); thMask.delete(); dil.delete(); eq.delete(); maxima.delete(); pts.delete();
    work.delete(); gray.delete();

    return { out, count };
  } catch (err) {
    // Absolute safety valve: Shi–Tomasi if ANYTHING throws
    console.warn('Harris failed, falling back to Shi–Tomasi:', err);
    return cornersShiTomasi(src);
  }
}


// ---------- Corner – Shi–Tomasi fallback (goodFeaturesToTrack) ----------
function cornersShiTomasi(src) {
  const work = maybeResize(src);
  const gray = toGray(work);

  // Tunables (safe defaults)
  const maxCorners   = 300;
  const qualityLevel = 0.01;   // 1% of strongest
  const minDistance  = 6;      // pixels between corners
  const blockSize    = (els.hWin ? +els.hWin.value : 5);
  const useHarris    = false;  // pure Shi–Tomasi

  const corners = new cv.Mat();
  try {
    cv.goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance,
                           new cv.Mat(), blockSize, useHarris, 0.04);
  } catch (e) {
    // Extremely rare: if gFTT not available, show original with message
    console.error('goodFeaturesToTrack failed:', e);
    const outOrig = new cv.Mat(); cv.cvtColor(work, outOrig, cv.COLOR_RGBA2RGB);
    gray.delete(); work.delete();
    return { out: outOrig, count: 0 };
  }

  const out = new cv.Mat(); cv.cvtColor(work, out, cv.COLOR_RGBA2RGB);
  const count = corners.rows || 0;
  const color = new cv.Scalar(0, 255, 0, 255);

  // corners is Nx1xCV_32FC2 (float x,y)
  for (let i = 0; i < count; i++) {
    const p = corners.floatPtr(i, 0);
    const x = Math.round(p[0]), y = Math.round(p[1]);
    cv.circle(out, new cv.Point(x, y), 4, color, 2, cv.LINE_AA);
  }

  corners.delete(); gray.delete(); work.delete();
  return { out, count };
}






















// ---------- Rendering ----------
async function renderCurrent() {
  if (state.source === 'dataset' && state.images.length) {
    showVideo(false); showCanvas(true);
    await waitForCV();

    const bmp = state.images[state.index].bitmap;
    const mode = state.mode;
    const src = getSrcMatFromBitmap(bmp); // RGBA
    let out = null, stats = '';

    try {
      if (mode === 'original') {
        drawBitmapToCanvas(bmp); overlayModeLabel('ORIGINAL');
      } else if (mode === 'grad_mag') {
        out = gradMagnitudeMat(src); imshowFit(out); overlayModeLabel('GRADIENT MAG');
      } else if (mode === 'grad_angle') {
        out = gradAngleMat(src);    imshowFit(out); overlayModeLabel('GRADIENT ANGLE');
      } else if (mode === 'log') {
        out = logMat(src);          imshowFit(out); overlayModeLabel('LoG');
      } else if (mode === 'edge_simple') {
        const { out: E, count } = edgeSimple(src, true); // show binary edges by default
        out = E; imshowFit(out); overlayModeLabel('EDGE – NMS + HYST'); stats = ` | edges: ${count}`;
      } else if (mode === 'corners_harris') {
        const { out: C, count } = cornersHarris(src);
        out = C; imshowFit(out); overlayModeLabel('CORNERS – HARRIS'); stats = ` | corners: ${count}`;
      } else {
        // Unknown mode -> just show original so UI never blanks
        drawBitmapToCanvas(bmp); overlayModeLabel('ORIGINAL');
      }
    } catch (e) {
      console.error('Render error:', e);
      drawBitmapToCanvas(bmp);
      overlayModeLabel('ERROR (original)');
      setStatus(`Error rendering “${mode}”: ${e?.message || e}`);
    } finally {
      src.delete();
      if (out) out.delete();
    }

    setStatus(`Image ${state.index+1}/${state.images.length} — Mode: ${state.mode} — Source: dataset${stats}`);
    els.save.disabled = (mode === 'original');
  } else if (state.source === 'webcam' && state.stream) {
    showCanvas(false); showVideo(true);
    setStatus(`Live webcam — Mode: ${state.mode}`);
    els.save.disabled = true;
  } else {
    showCanvas(false); showVideo(false);
  }
}

// ---------- UI wiring ----------
els.fileInput.onchange = async (e) => {
  try {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    state.images = await filesToBitmaps(files);
    state.index = 0; state.source = 'dataset';
    enableNav(true);
    els.camCapture.disabled = true; els.camStop.disabled = true;
    await renderCurrent();
  } catch (err) {
    console.error(err);
    setStatus('Failed to load images. Try JPG/PNG (not HEIC), or smaller files.');
  }
};
els.loadDemo.onclick = async () => {
  const demoDataURL =
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABkCAYAAABoQy0/AAAACXBIWXMAAAsSAAALEgHS3X78AAABtUlEQVR4nO3RMQEAIAzAsFv/0m0kF3m4JY2nQbdk2iEJ8Qe3GgAAAAAAAAAAAAAAAAAAgL3M3wqf9y3iXWg+0l7Yf8m1oG6D2wE1qj8aXy9w8g7dD3vQb1m4R6k9sBNao/Gn8vcPJO3Q971G9ZuEepPbATWqPxpfL3DyDt0Pe9BvWbhHqT2wE1qj8aXy9w8g7dD3vQb1m4R6k9sBNao/Gn8vcPJO3Q971G9ZuEepPbATWqPxpfL3DyDt0Pe9BvWbhHqT2wE1qj8aXy9w8g7dD3vQb1m4R6k9sBNao/Gn8vcPJO3Q971G9ZuEepPbATWqPxpfL3DyDs3C1k2iEJ8Qe3GgAAAAAAAAAAAAAAAOA3GUM1c7Jt0AAAAABJRU5ErkJggg==";
  const res = await fetch(demoDataURL);
  const blob = await res.blob();
  const bmp = await createImageBitmap(blob);
  state.images = [{ name: "demo.png", bitmap: bmp }];
  state.index = 0; state.source = 'dataset';
  enableNav(true);
  await renderCurrent();
};
els.prev.onclick = async () => { if (!state.images.length) return; state.index = (state.index - 1 + state.images.length) % state.images.length; await renderCurrent(); };
els.next.onclick = async () => { if (!state.images.length) return; state.index = (state.index + 1) % state.images.length; await renderCurrent(); };
els.camStart.onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } }, audio: false });
    state.stream = stream; els.video.srcObject = stream; state.source = 'webcam';
    els.camCapture.disabled = false; els.camStop.disabled = false; enableNav(false); await renderCurrent();
  } catch (err) { setStatus(`Camera error: ${err.message}`); }
};
els.camCapture.onclick = () => {
  const v = els.video, c = els.canvas, ctx = c.getContext('2d');
  c.width = v.videoWidth; c.height = v.videoHeight; ctx.drawImage(v, 0, 0, c.width, c.height);
  v.pause();
  c.toBlob(async (blob) => {
    const bmp = await createImageBitmap(blob);
    state.images.push({ name: `capture_${Date.now()}.png`, bitmap: bmp });
    state.index = state.images.length - 1; state.source = 'dataset'; enableNav(true); await renderCurrent();
  });
};
els.camStop.onclick = () => { if (state.stream) { state.stream.getTracks().forEach(t => t.stop()); state.stream = null; } showVideo(false); setStatus('Webcam stopped.'); els.camCapture.disabled = true; els.camStop.disabled = true; };

// Re-render when knobs change (if they exist)
function re(){ renderCurrent(); }
if (els.mode)        els.mode.onchange     = () => { state.mode = els.mode.value; re(); };
if (els.edgeBoost)   els.edgeBoost.oninput = re;
if (els.halfRes)     els.halfRes.onchange  = re;
if (els.edgeLow)     els.edgeLow.oninput   = () => { if (els.edgeAuto) els.edgeAuto.checked = false; re(); };
if (els.edgeHigh)    els.edgeHigh.oninput  = () => { if (els.edgeAuto) els.edgeAuto.checked = false; re(); };
if (els.edgeAuto)    els.edgeAuto.onchange = re;
if (els.edgeBinary)  els.edgeBinary.onchange = re;
if (els.hWin)        els.hWin.oninput      = re;
if (els.hK)          els.hK.oninput        = re;
if (els.hTh)         els.hTh.oninput       = re;
if (els.hHeat)       els.hHeat.onchange    = re;

// ---------- Save current canvas ----------
els.save.onclick = () => {
  const srcW = els.canvas.width, srcH = els.canvas.height;
  if (!srcW || !srcH) { setStatus('Nothing to save.'); return; }
  const link = document.createElement('a');
  const base = (state.images[state.index]?.name || 'capture').replace(/\.[^.]+$/, '');
  link.download = `${base}__${state.mode}.png`;
  link.href = els.canvas.toDataURL('image/png');
  try { link.click(); } catch { setStatus('If no download appears, allow multiple downloads for this site.'); }
};

// ---------- Export All (Task 1) ----------
els.exportAll1.onclick = async () => {
  await waitForCV();
  if (!state.images.length) { setStatus('Load images first, then try Export All.'); return; }

  const off = document.createElement('canvas');
  function saveMatAsPNG(mat, filename) {
    off.width = mat.cols; off.height = mat.rows;
    cv.imshow(off, mat);
    const a = document.createElement('a');
    a.download = filename;
    a.href = off.toDataURL('image/png');
    try { a.click(); } catch {}
  }

  setStatus('Exporting… your browser may ask to allow multiple downloads.');
  for (let i = 0; i < state.images.length; i++) {
    const item = state.images[i];
    const srcRGBA = getSrcMatFromBitmap(item.bitmap);
    const stem = item.name.replace(/\.[^.]+$/, '');

    try {
      const gmag = gradMagnitudeMat(srcRGBA);  saveMatAsPNG(gmag, `${stem}__grad_mag.png`);  gmag.delete();
      const gang = gradAngleMat(srcRGBA);      saveMatAsPNG(gang, `${stem}__grad_angle.png`); gang.delete();
      const glog = logMat(srcRGBA);            saveMatAsPNG(glog, `${stem}__log.png`);        glog.delete();
    } catch (e) {
      console.error('Export error on', item.name, e);
      setStatus(`Export error on ${item.name}: ${e?.message || e}`);
    } finally {
      srcRGBA.delete();
    }
    await new Promise(r => setTimeout(r, 120));
  }
  setStatus(`Exported Task 1 outputs for ${state.images.length} image(s). If nothing downloaded, enable multiple downloads.`);
};

// ---------- Export All (Task 2): edges + corners ----------
els.exportAll2.onclick = async () => {
  await waitForCV();
  if (!state.images.length) { setStatus('Load images first, then try Export All (Task 2).'); return; }

  const off = document.createElement('canvas');
  function saveMatAsPNG(mat, filename) {
    off.width = mat.cols; off.height = mat.rows;
    cv.imshow(off, mat);
    const a = document.createElement('a');
    a.download = filename;
    a.href = off.toDataURL('image/png');
    try { a.click(); } catch {}
  }

  setStatus('Exporting Task 2… your browser may ask to allow multiple downloads.');
  for (let i = 0; i < state.images.length; i++) {
    const item = state.images[i];
    const srcRGBA = getSrcMatFromBitmap(item.bitmap);
    const stem = item.name.replace(/\.[^.]+$/, '');

    try {
      // Edge map (binary for clarity)
      const { out: E } = edgeSimple(srcRGBA, /*returnBinary=*/true);
      saveMatAsPNG(E, `${stem}__edge.png`);
      E.delete();

      // Corners (Harris with auto Shi–Tomasi fallback already inside)
      const { out: C } = cornersHarris(srcRGBA);
      saveMatAsPNG(C, `${stem}__corners.png`);
      C.delete();
    } catch (e) {
      console.error('Task 2 export error on', item.name, e);
      setStatus(`Task 2 export error on ${item.name}: ${e?.message || e}`);
    } finally {
      srcRGBA.delete();
    }
    // small delay so browsers don’t block multiple downloads
    await new Promise(r => setTimeout(r, 120));
  }
  setStatus(`Exported Task 2 outputs for ${state.images.length} image(s). If nothing downloaded, enable multiple downloads for this site.`);
};

