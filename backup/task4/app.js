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

  edgeLow:     document.getElementById('edgeLow'),
  edgeHigh:    document.getElementById('edgeHigh'),
  edgeAuto:    document.getElementById('edgeAuto'),
  edgeBinary:  document.getElementById('edgeBinary'),

  hWin:        document.getElementById('hWin'),
  hK:          document.getElementById('hK'),
  hTh:         document.getElementById('hTh'),
  hHeat:       document.getElementById('hHeat'),

  // Task 3
  bClose:      document.getElementById('bClose'),
  bMinArea:    document.getElementById('bMinArea'),
  bEps:        document.getElementById('bEps'),
  bCenterR:    document.getElementById('bCenterR'),

  // Task 4
  arucoDict:        document.getElementById('arucoDict'),
  arucoUseCorners:  document.getElementById('arucoUseCorners'),
  arucoShowIds:     document.getElementById('arucoShowIds'),
  arucoDilate:      document.getElementById('arucoDilate'),

  save:        document.getElementById('save'),
  exportAll1:  document.getElementById('exportAll1'),
  exportAll2:  document.getElementById('exportAll2'),
  exportAll3:  document.getElementById('exportAll3'),
  exportAll4:  document.getElementById('exportAll4'),
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
  els.exportAll2.disabled = !enable || !state.images.length;
  els.exportAll3.disabled = !enable || !state.images.length;
  els.exportAll4.disabled = !enable || !state.images.length;
}
function showVideo(show) { els.video.style.display = show ? 'block' : 'none'; }
function showCanvas(show) { els.canvas.style.display = show ? 'block' : 'none'; }

// ---------- Robust image loading ----------
const MAX_SIDE = 4096;
async function fileToBitmap(file) {
  if (/hei[cf]/i.test(file.type) || /\.heic$|\.heif$/i.test(file.name)) {
    throw new Error('HEIC/HEIF is not supported. Please convert to JPG/PNG.');
  }
  try { return await createImageBitmap(file); }
  catch (_) {
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
  if (errors.length) console.warn('Some files failed:\n' + errors.join('\n'));
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
  ctx.fillStyle = 'rgba(0,0,0,0.45)';
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
function percentileFrom8U(mat8, p) {
  const hist = new Array(256).fill(0);
  const d = mat8.data;
  for (let i = 0; i < d.length; i++) hist[d[i]]++;
  const target = (p / 100) * d.length;
  let cum = 0;
  for (let v = 0; v < 256; v++) { cum += hist[v]; if (cum >= target) return v; }
  return 255;
}

// ---------- Task 1 ----------
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
  cv.cartToPolar(dx, dy, mag, ang, true);
  cv.normalize(mag, mag, 0, 255, cv.NORM_MINMAX);
  mag.convertTo(mag8, cv.CV_8U);
  const mask = new cv.Mat(); cv.threshold(mag8, mask, 20, 255, cv.THRESH_BINARY);
  if (els.edgeBoost && +els.edgeBoost.value > 0) {
    const kernel = cv.Mat.ones(3,3,cv.CV_8U);
    cv.dilate(mask, mask, kernel); kernel.delete();
  }
  ang.convertTo(angH, cv.CV_8U, 0.5);
  const hueMax = new cv.Mat(angH.rows, angH.cols, cv.CV_8U); hueMax.setTo(new cv.Scalar(179));
  cv.min(angH, hueMax, angH); hueMax.delete();
  const sat = new cv.Mat(gray.rows, gray.cols, cv.CV_8U); sat.setTo(new cv.Scalar(255));
  const val = new cv.Mat(); cv.bitwise_and(mag8, mask, val);
  const mv = new cv.MatVector(); mv.push_back(angH); mv.push_back(sat); mv.push_back(val);
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
    cv.dilate(lap8, lap8, kernel); kernel.delete();
    if (+els.edgeBoost.value === 2) cv.equalizeHist(lap8, lap8);
  }
  work.delete(); gray.delete(); blur.delete(); lap.delete();
  return lap8;
}

// ---------- Task 2 ----------
function edgeSimple(src, returnBinary=true) {
  const work = maybeResize(src);
  const gray = toGray(work);
  cv.GaussianBlur(gray, gray, new cv.Size(3,3), 0.8, 0.8);
  const dx = new cv.Mat(), dy = new cv.Mat();
  const mag = new cv.Mat(), ang = new cv.Mat();
  cv.Sobel(gray, dx, cv.CV_32F, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT);
  cv.Sobel(gray, dy, cv.CV_32F, 0, 1, 3, 1, 0, cv.BORDER_DEFAULT);
  cv.magnitude(dx, dy, mag); cv.phase(dx, dy, ang, true);
  const rows = mag.rows, cols = mag.cols;
  const nms = cv.Mat.zeros(rows, cols, cv.CV_32F);
  const M = mag.data32F, A = ang.data32F, N = nms.data32F;
  for (let y = 1; y < rows - 1; y++) {
    const r = y * cols;
    for (let x = 1; x < cols - 1; x++) {
      const i = r + x; let a = A[i];
      if (a < 0) a += 180; if (a >= 180) a -= 180;
      let o1 = 0, o2 = 0;
      if (a < 22.5 || a >= 157.5) { o1 = -1; o2 = +1; }
      else if (a < 67.5) { o1 = -cols - 1; o2 = +cols + 1; }
      else if (a < 112.5) { o1 = -cols; o2 = +cols; }
      else { o1 = -cols + 1; o2 = +cols - 1; }
      const m = M[i];
      if (m >= M[i + o1] && m >= M[i + o2]) N[i] = m;
    }
  }
  const nms8 = new cv.Mat();
  cv.normalize(nms, nms, 0, 255, cv.NORM_MINMAX);
  nms.convertTo(nms8, cv.CV_8U);
  let high = 40, low = 15;
  if (!els.edgeAuto || els.edgeAuto.checked) {
    high = percentileFrom8U(nms8, 85);
    low  = Math.max(5, Math.round(0.4 * high));
    if (els.edgeHigh) els.edgeHigh.value = String(high);
    if (els.edgeLow)  els.edgeLow.value  = String(low);
  } else {
    high = +els.edgeHigh.value; low = +els.edgeLow.value;
  }
  const edges = cv.Mat.zeros(rows, cols, cv.CV_8U);
  const D = nms8.data, E = edges.data; const STR=255, WEAK=128;
  for (let i = 0; i < D.length; i++) { const v = D[i]; if (v >= high) E[i] = STR; else if (v >= low) E[i] = WEAK; }
  const stack = [];
  for (let i = 0; i < E.length; i++) if (E[i] === STR) stack.push(i);
  while (stack.length) {
    const p = stack.pop(); const y = (p/cols)|0, x = p - y*cols;
    for (let yy = y-1; yy <= y+1; yy++) for (let xx = x-1; xx <= x+1; xx++) {
      if (yy<=0||yy>=rows-1||xx<=0||xx>=cols-1) continue;
      const q = yy*cols + xx;
      if (E[q] === WEAK) { E[q] = STR; stack.push(q); }
    }
  }
  for (let i = 0; i < E.length; i++) E[i] = (E[i] === STR) ? 255 : 0;
  const kernel = cv.Mat.ones(3,3,cv.CV_8U); cv.dilate(edges, edges, kernel); kernel.delete();
  if (returnBinary || (els.edgeBinary && els.edgeBinary.checked)) {
    const out = new cv.Mat(); cv.cvtColor(edges, out, cv.COLOR_GRAY2RGB);
    gray.delete(); dx.delete(); dy.delete(); mag.delete(); ang.delete(); nms.delete(); nms8.delete(); work.delete();
    return { out, count: cv.countNonZero(edges) };
  } else {
    const base = new cv.Mat();
    cv.cvtColor(work, base, cv.COLOR_RGBA2RGB);
    const red = new cv.Mat(base.rows, base.cols, cv.CV_8UC3, new cv.Scalar(255,0,0,0));
    const edgeRGB = new cv.Mat(); red.copyTo(edgeRGB, edges);
    const out = new cv.Mat(); cv.addWeighted(base, 0.6, edgeRGB, 1.0, 0, out);
    const count = cv.countNonZero(edges);
    gray.delete(); dx.delete(); dy.delete(); mag.delete(); ang.delete(); nms.delete(); nms8.delete(); edges.delete(); base.delete(); red.delete(); edgeRGB.delete(); work.delete();
    return { out, count };
  }
}

// ---------- Task 2: Corners (Harris + fallback) ----------
function cornersHarris(src) {
  try {
    const work = maybeResize(src);
    const gray = toGray(work);
    cv.GaussianBlur(gray, gray, new cv.Size(3,3), 0.8, 0.8);
    const blockSize = (els.hWin ? +els.hWin.value : 5);
    const k = (els.hK ? (+els.hK.value)/100.0 : 0.04);
    let th = (els.hTh ? +els.hTh.value : 70);
    const Ix = new cv.Mat(), Iy = new cv.Mat();
    cv.Sobel(gray, Ix, cv.CV_32F, 1, 0, 3, 1, 0);
    cv.Sobel(gray, Iy, cv.CV_32F, 0, 1, 3, 1, 0);
    const Ixx = new cv.Mat(), Iyy = new cv.Mat(), Ixy = new cv.Mat();
    cv.multiply(Ix, Ix, Ixx); cv.multiply(Iy, Iy, Iyy); cv.multiply(Ix, Iy, Ixy);
    Ix.delete(); Iy.delete();
    const win = new cv.Size(blockSize, blockSize);
    const Sxx = new cv.Mat(), Syy = new cv.Mat(), Sxy = new cv.Mat();
    cv.GaussianBlur(Ixx, Sxx, win, 1.0); cv.GaussianBlur(Iyy, Syy, win, 1.0); cv.GaussianBlur(Ixy, Sxy, win, 1.0);
    Ixx.delete(); Iyy.delete(); Ixy.delete();
    const det = new cv.Mat(), tmp = new cv.Mat();
    cv.multiply(Sxx, Syy, det); cv.multiply(Sxy, Sxy, tmp); cv.subtract(det, tmp, det);
    const trace = new cv.Mat(), trace2 = new cv.Mat(); cv.add(Sxx, Syy, trace); cv.multiply(trace, trace, trace2);
    const R = new cv.Mat(); cv.addWeighted(det, 1.0, trace2, -k, 0.0, R);
    Sxx.delete(); Syy.delete(); Sxy.delete(); det.delete(); tmp.delete(); trace.delete(); trace2.delete();
    const Rn = new cv.Mat(); cv.normalize(R, Rn, 0, 255, cv.NORM_MINMAX, cv.CV_32F);
    const R8 = new cv.Mat(); Rn.convertTo(R8, cv.CV_8U); R.delete(); Rn.delete();
    const out = new cv.Mat(); cv.cvtColor(work, out, cv.COLOR_RGBA2RGB);
    const showHeat = (!els.hHeat || els.hHeat.checked);
    if (showHeat) {
      const eq = new cv.Mat(), heat = new cv.Mat();
      cv.equalizeHist(R8, eq); cv.cvtColor(eq, heat, cv.COLOR_GRAY2RGB);
      cv.addWeighted(out, 0.45, heat, 0.55, 0, out); eq.delete(); heat.delete();
    }
    const thMask = new cv.Mat(); cv.threshold(R8, thMask, th, 255, cv.THRESH_BINARY);
    const dil = new cv.Mat(); const ker = cv.Mat.ones(3,3,cv.CV_8U); cv.dilate(R8, dil, ker); ker.delete();
    const eq = new cv.Mat(); cv.compare(R8, dil, eq, cv.CMP_EQ);
    const maxima = new cv.Mat(); cv.bitwise_and(eq, thMask, maxima);
    const pts = new cv.Mat(); cv.findNonZero(maxima, pts);
    let count = pts.rows || 0;
    if (count === 0) {
      cv.threshold(R8, thMask, 40, 255, cv.THRESH_BINARY);
      cv.bitwise_and(eq, thMask, maxima);
      cv.findNonZero(maxima, pts);
      count = pts.rows || 0;
      if (els.hTh) els.hTh.value = '40';
    }
    const color = new cv.Scalar(0,255,0,255);
    for (let i = 0; i < count; i++) {
      const p = pts.intPtr(i, 0);
      cv.circle(out, new cv.Point(p[0], p[1]), 4, color, 2, cv.LINE_AA);
    }
    R8.delete(); thMask.delete(); dil.delete(); eq.delete(); maxima.delete(); pts.delete();
    work.delete(); gray.delete();
    return { out, count };
  } catch (e) {
    return cornersShiTomasi(src);
  }
}
function cornersShiTomasi(src) {
  const work = maybeResize(src);
  const gray = toGray(work);
  const maxCorners=300, quality=0.01, minDist=6, block=(els.hWin?+els.hWin.value:5), useHarris=false;
  const corners = new cv.Mat();
  cv.goodFeaturesToTrack(gray, corners, maxCorners, quality, minDist, new cv.Mat(), block, useHarris, 0.04);
  const out = new cv.Mat(); cv.cvtColor(work, out, cv.COLOR_RGBA2RGB);
  const color = new cv.Scalar(0,255,0,255);
  const count = corners.rows || 0;
  for (let i=0;i<count;i++){ const p=corners.floatPtr(i,0); cv.circle(out,new cv.Point(Math.round(p[0]),Math.round(p[1])),4,color,2,cv.LINE_AA); }
  corners.delete(); gray.delete(); work.delete();
  return { out, count };
}

// ---------- Task 3: Boundary – Contour ----------
function boundaryContour(src) {
  const work = maybeResize(src);
  const gray = toGray(work);
  cv.GaussianBlur(gray, gray, new cv.Size(5,5), 0.8, 0.8);
  let low = 20, high = 60;
  if (!els.edgeAuto || els.edgeAuto.checked) {
    const mag8 = gradMagnitudeMat(work);
    const tmp = new cv.Mat();
    const otsu = cv.threshold(mag8, tmp, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
    tmp.delete(); mag8.delete();
    high = Math.max(30, Math.round(otsu));
    low  = Math.max(5, Math.round(0.5 * high));
    if (els.edgeHigh) els.edgeHigh.value = String(high);
    if (els.edgeLow)  els.edgeLow.value  = String(low);
  } else { high = +els.edgeHigh.value; low = +els.edgeLow.value; }
  const edges = new cv.Mat(); cv.Canny(gray, edges, low, high, 3, true);
  const closeK = (els.bClose ? +els.bClose.value : 5) || 5;
  const ksize = Math.max(1, closeK | 0);
  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(ksize, ksize));
  const closed = new cv.Mat(); cv.morphologyEx(edges, closed, cv.MORPH_CLOSE, kernel);
  kernel.delete(); edges.delete(); gray.delete();
  const contours = new cv.MatVector(); const hierarchy = new cv.Mat();
  cv.findContours(closed, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  closed.delete(); hierarchy.delete();
  const rows = work.rows, cols = work.cols;
  const imgArea = rows * cols;
  const minAreaPct = (els.bMinArea ? +els.bMinArea.value : 2);
  const minArea = (minAreaPct / 100) * imgArea;
  const centerR = (els.bCenterR ? +els.bCenterR.value : 40) / 100;
  const cx0 = cols / 2, cy0 = rows / 2;
  const diag = Math.hypot(cols, rows);
  let bestIdx = -1, bestScore = -1, bestArea = 0;
  for (let i = 0; i < contours.size(); i++) {
    const cnt = contours.get(i);
    const area = cv.contourArea(cnt, false);
    if (area < minArea) { cnt.delete(); continue; }
    const m = cv.moments(cnt, false);
    if (m.m00 === 0) { cnt.delete(); continue; }
    const cx = m.m10 / m.m00, cy = m.m01 / m.m00;
    const dist = Math.hypot(cx - cx0, cy - cy0) / diag;
    const centerOK = dist <= centerR;
    const score = area * (centerOK ? 1.2 : 1.0) * (1.0 - 0.6 * dist);
    if (score > bestScore) { bestScore = score; bestIdx = i; bestArea = area; }
    cnt.delete();
  }
  const out = new cv.Mat(); cv.cvtColor(work, out, cv.COLOR_RGBA2RGB);
  if (bestIdx < 0) {
    contours.delete(); overlayModeLabel('BOUNDARY – NONE'); work.delete();
    return { out, info: 'no contour' };
  }
  const best = contours.get(bestIdx);
  const perim = cv.arcLength(best, true);
  const epsPct = (els.bEps ? +els.bEps.value : 1.5) / 100.0;
  const epsilon = Math.max(0.5, epsPct * perim);
  const approx = new cv.Mat(); cv.approxPolyDP(best, approx, epsilon, true);
  const mask = cv.Mat.zeros(rows, cols, cv.CV_8U);
  const mv = new cv.MatVector(); mv.push_back(approx);
  cv.fillPoly(mask, mv, new cv.Scalar(255)); mv.delete();
  if (els.edgeBinary && els.edgeBinary.checked) {
    const maskRGB = new cv.Mat(); cv.cvtColor(mask, maskRGB, cv.COLOR_GRAY2RGB);
    mask.delete(); work.delete(); best.delete(); contours.delete();
    return { out: maskRGB, info: `area=${Math.round(bestArea)} px, verts=${approx.rows}` };
  } else {
    const fillRGB = new cv.Mat(rows, cols, cv.CV_8UC3, new cv.Scalar(0, 255, 0, 0));
    const filled = new cv.Mat(); fillRGB.copyTo(filled, mask);
    const blended = new cv.Mat(); cv.addWeighted(out, 0.65, filled, 0.35, 0, blended);
    cv.polylines(blended, approx, true, new cv.Scalar(0, 255, 0, 255), 2, cv.LINE_AA);
    out.delete(); fillRGB.delete(); filled.delete(); mask.delete(); best.delete(); contours.delete(); work.delete();
    return { out: blended, info: `area=${Math.round(bestArea)} px, verts=${approx.rows}` };
  }
}

// ---------- Task 4: Segmentation – ArUco ----------
function arucoAvailable() {
  // Detect whether ArUco module exists in this build.
  return !!(cv.aruco &&
           (cv.aruco.getPredefinedDictionary || cv.aruco.Dictionary_get) &&
           (cv.aruco.detectMarkers || (cv.aruco.ArucoDetector && cv.aruco.DetectorParameters)));
}
function getArucoDictByName(name) {
  // Resolve dictionary by string (supports both newer & older JS bindings)
  if (!cv.aruco) throw new Error('ArUco module not available');
  const key = name in cv.aruco ? name : (('aruco' in cv) && (name in cv) ? name : null);
  const dictId = cv.aruco[name] ?? cv[name];
  if (cv.aruco.getPredefinedDictionary && dictId !== undefined) {
    return cv.aruco.getPredefinedDictionary(dictId);
  }
  if (cv.aruco.Dictionary_get && dictId !== undefined) {
    return cv.aruco.Dictionary_get(dictId);
  }
  throw new Error('Cannot create ArUco dictionary');
}
function detectAruco(gray, dictName) {
  // Returns { cornersVec, idsMat } OR null if none
  const corners = new cv.MatVector();
  const ids = new cv.Mat();
  try {
    const dict = getArucoDictByName(dictName);
    if (cv.aruco.ArucoDetector && cv.aruco.DetectorParameters) {
      // New API path
      const params = new cv.aruco.DetectorParameters();
      const detector = new cv.aruco.ArucoDetector(dict, params);
      detector.detectMarkers(gray, corners, ids);
    } else if (cv.aruco.detectMarkers) {
      // Classic API path
      const params = new cv.aruco.DetectorParameters();
      cv.aruco.detectMarkers(gray, dict, corners, ids, params);
    } else {
      throw new Error('No detectMarkers API found');
    }
  } catch (e) {
    corners.delete(); ids.delete();
    throw e;
  }
  if (ids.rows === 0) { corners.delete(); ids.delete(); return null; }
  return { cornersVec: corners, idsMat: ids };
}
function cornersToPointList(cornersVec, useCorners=true) {
  // Collect points either as marker centers or all 4 corners
  const pts = [];
  for (let i = 0; i < cornersVec.size(); i++) {
    const c = cornersVec.get(i);
    const f = c.data32F || c.data32S || c.data;
    if (useCorners) {
      // push 4 corners
      for (let k = 0; k < 4; k++) pts.push({ x: Math.round(f[k*2]), y: Math.round(f[k*2 + 1]) });
    } else {
      // center
      const x = (f[0] + f[2] + f[4] + f[6]) * 0.25;
      const y = (f[1] + f[3] + f[5] + f[7]) * 0.25;
      pts.push({ x: Math.round(x), y: Math.round(y) });
    }
    c.delete();
  }
  return pts;
}
function pointsToContourMat(pts) {
  // Order points around centroid by angle for a non-self-crossing polygon
  const cx = pts.reduce((s,p)=>s+p.x,0)/pts.length;
  const cy = pts.reduce((s,p)=>s+p.y,0)/pts.length;
  pts.sort((a,b)=> Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx));
  const contour = new cv.Mat(pts.length, 1, cv.CV_32SC2);
  for (let i = 0; i < pts.length; i++) {
    const ptr = contour.intPtr(i, 0);
    ptr[0] = pts[i].x; ptr[1] = pts[i].y;
  }
  return contour;
}
function segAruco(src) {
  // Build mask from detected ArUco markers and overlay boundary
  const work = maybeResize(src); // allows half-res preview; exports use full-res path
  const rgb = new cv.Mat(); cv.cvtColor(work, rgb, cv.COLOR_RGBA2RGB);
  const gray = new cv.Mat(); cv.cvtColor(work, gray, cv.COLOR_RGBA2GRAY);

  if (!arucoAvailable()) {
    const out = rgb.clone();
    overlayModeLabel('ARUCO MODULE MISSING');
    gray.delete(); work.delete(); rgb.delete();
    return { out, info: 'aruco-missing' };
  }

  const dictName = els.arucoDict?.value || 'DICT_4X4_50';
  const det = detectAruco(gray, dictName);
  if (!det) {
    const out = rgb.clone();
    overlayModeLabel('ARUCO – NONE');
    gray.delete(); work.delete(); rgb.delete();
    return { out, info: 'no-markers' };
  }
  const { cornersVec, idsMat } = det;

  // Draw IDs (optional)
  if (!els.edgeBinary?.checked && els.arucoShowIds?.checked) {
    for (let i = 0; i < cornersVec.size(); i++) {
      const c = cornersVec.get(i);
      const f = c.data32F || c.data32S || c.data;
      const id = idsMat.intAt ? idsMat.intAt(i, 0) : (idsMat.data32S ? idsMat.data32S[i] : i);
      const cx = Math.round((f[0] + f[2] + f[4] + f[6]) * 0.25);
      const cy = Math.round((f[1] + f[3] + f[5] + f[7]) * 0.25);
      cv.putText(rgb, String(id), new cv.Point(cx, cy), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(0,255,255,255), 2, cv.LINE_AA);
      c.delete();
    }
  } else {
    // still need to free corner mats later if not already
    for (let i = 0; i < cornersVec.size(); i++) cornersVec.get(i).delete();
  }

  // Re-detect (corner mats were deleted above); detect again to harvest points
  const det2 = detectAruco(gray, dictName);
  gray.delete();

  if (!det2) {
    const out = rgb.clone();
    overlayModeLabel('ARUCO – NONE');
    work.delete(); rgb.delete();
    return { out, info: 'no-markers' };
  }

  const pts = cornersToPointList(det2.cornersVec, !!(els.arucoUseCorners?.checked));
  det2.idsMat.delete(); det2.cornersVec.delete();

  if (pts.length < 3) {
    const out = rgb.clone();
    overlayModeLabel('ARUCO – TOO FEW PTS');
    work.delete(); rgb.delete();
    return { out, info: 'few-points' };
  }

  const contour = pointsToContourMat(pts);

  // Mask from polygon
  const rows = work.rows, cols = work.cols;
  let mask = cv.Mat.zeros(rows, cols, cv.CV_8U);
  const mv = new cv.MatVector(); mv.push_back(contour);
  cv.fillPoly(mask, mv, new cv.Scalar(255)); mv.delete(); contour.delete();

  // Optional dilation to make mask snugger
  const dil = Math.max(0, +(els.arucoDilate?.value || 0));
  if (dil > 0) {
    const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(dil, dil));
    const tmp = new cv.Mat(); cv.dilate(mask, tmp, k); mask.delete(); mask = tmp; k.delete();
  }

  // Binary or overlay
  if (els.edgeBinary && els.edgeBinary.checked) {
    const maskRGB = new cv.Mat(); cv.cvtColor(mask, maskRGB, cv.COLOR_GRAY2RGB);
    mask.delete(); work.delete(); rgb.delete();
    return { out: maskRGB, info: `aruco pts=${pts.length}` };
  } else {
    const fillRGB = new cv.Mat(rows, cols, cv.CV_8UC3, new cv.Scalar(0, 255, 0, 0));
    const filled = new cv.Mat(); fillRGB.copyTo(filled, mask);
    const blended = new cv.Mat(); cv.addWeighted(rgb, 0.65, filled, 0.35, 0, blended);

    // Outline (polyline from ordered points)
    const outline = pointsToContourMat(pts);
    cv.polylines(blended, outline, true, new cv.Scalar(0,255,0,255), 2, cv.LINE_AA);
    outline.delete();

    // cleanup
    fillRGB.delete(); filled.delete(); mask.delete(); work.delete(); rgb.delete();

    return { out: blended, info: `aruco pts=${pts.length}` };
  }
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
        const { out: E, count } = edgeSimple(src, true);
        out = E; imshowFit(out); overlayModeLabel('EDGE – NMS + HYST'); stats = ` | edges: ${count}`;
      } else if (mode === 'corners_harris') {
        const { out: C, count } = cornersHarris(src);
        out = C; imshowFit(out); overlayModeLabel('CORNERS – HARRIS'); stats = ` | corners: ${count}`;
      } else if (mode === 'boundary_contour') {
        const { out: B, info } = boundaryContour(src);
        out = B; imshowFit(out); overlayModeLabel('BOUNDARY – CONTOUR'); stats = info ? ` | ${info}` : '';
      } else if (mode === 'seg_aruco') {
        const { out: S, info } = segAruco(src);
        out = S; imshowFit(out); overlayModeLabel('SEG – ARUCO'); stats = info ? ` | ${info}` : '';
      } else {
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
  } catch (err) { console.error(err); setStatus('Failed to load images. Try JPG/PNG (not HEIC).'); }
};
els.loadDemo.onclick = async () => {
  const demoDataURL =
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABkCAYAAABoQy0/AAAACXBIWXMAAAsSAAALEgHS3X78AAABtUlEQVR4nO3RMQEAIAzAsFv/0m0kF3m4JY2nQbdk2iEJ8Qe3GgAAAAAAAAAAAAAAAAAAgL3M3wqf9y3iXWg+0l7Yf8m1oG6D2wE1qj8aXy9w8g7dD3vQb1m4R6k9sBNao/Gn8vcPJO3Q971G9ZuEepPbATWqPxpfL3DyDt0Pe9BvWbhHqT2wE1qj8aXy9w8g7dD3vQb1m4R6k9sBNao/Gn8vcPJO3Q971G9ZuEepPbATWqPxpfL3DyDs3C1k2iEJ8Qe3GgAAAAAAAAAAAAAAAOA3GUM1c7Jt0AAAAABJRU5ErkJggg==";
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

// re-render on any knob change
function re(){ renderCurrent(); }
if (els.mode)        els.mode.onchange     = () => { state.mode = els.mode.value; re(); };
['edgeBoost','halfRes','edgeLow','edgeHigh','edgeAuto','edgeBinary','hWin','hK','hTh','hHeat','bClose','bMinArea','bEps','bCenterR','arucoDict','arucoUseCorners','arucoShowIds','arucoDilate']
  .forEach(id => { const el = els[id]; if (!el) return; if (el.type === 'range') el.oninput = re; else el.onchange = re; });
if (els.edgeLow)  els.edgeLow.oninput  = () => { if (els.edgeAuto) els.edgeAuto.checked = false; re(); };
if (els.edgeHigh) els.edgeHigh.oninput = () => { if (els.edgeAuto) els.edgeAuto.checked = false; re(); };

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
    a.download = filename; a.href = off.toDataURL('image/png'); try { a.click(); } catch {}
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
    } catch (e) { console.error('Export error on', item.name, e); setStatus(`Export error on ${item.name}: ${e?.message || e}`); }
    finally { srcRGBA.delete(); }
    await new Promise(r => setTimeout(r, 120));
  }
  setStatus(`Exported Task 1 outputs for ${state.images.length} image(s). If nothing downloaded, enable multiple downloads.`);
};

// ---------- Export All (Task 2) ----------
els.exportAll2.onclick = async () => {
  await waitForCV();
  if (!state.images.length) { setStatus('Load images first, then try Export All (Task 2).'); return; }
  const off = document.createElement('canvas');
  function saveMatAsPNG(mat, filename) {
    off.width = mat.cols; off.height = mat.rows;
    cv.imshow(off, mat);
    const a = document.createElement('a');
    a.download = filename; a.href = off.toDataURL('image/png'); try { a.click(); } catch {}
  }
  setStatus('Exporting Task 2… your browser may ask to allow multiple downloads.');
  for (let i = 0; i < state.images.length; i++) {
    const item = state.images[i];
    const srcRGBA = getSrcMatFromBitmap(item.bitmap);
    const stem = item.name.replace(/\.[^.]+$/, '');
    try {
      const { out: E } = edgeSimple(srcRGBA, true);  saveMatAsPNG(E, `${stem}__edge.png`); E.delete();
      const { out: C } = cornersHarris(srcRGBA);    saveMatAsPNG(C, `${stem}__corners.png`); C.delete();
    } catch (e) { console.error('Task 2 export error on', item.name, e); setStatus(`Task 2 export error on ${item.name}: ${e?.message || e}`); }
    finally { srcRGBA.delete(); }
    await new Promise(r => setTimeout(r, 120));
  }
  setStatus(`Exported Task 2 outputs for ${state.images.length} image(s). If nothing downloaded, enable multiple downloads for this site.`);
};

// ---------- Task 3 helper for Export ----------
function computeBoundaryMaskAndOverlay(srcRGBA) {
  const rows = srcRGBA.rows, cols = srcRGBA.cols;
  const rgb = new cv.Mat(); cv.cvtColor(srcRGBA, rgb, cv.COLOR_RGBA2RGB);
  const gray = new cv.Mat(); cv.cvtColor(srcRGBA, gray, cv.COLOR_RGBA2GRAY);
  cv.GaussianBlur(gray, gray, new cv.Size(5,5), 0.8, 0.8);
  let low = 20, high = 60;
  const dx = new cv.Mat(), dy = new cv.Mat(), mag = new cv.Mat(), mag8 = new cv.Mat(), tmp = new cv.Mat();
  cv.Sobel(gray, dx, cv.CV_32F, 1, 0, 3, 1, 0); cv.Sobel(gray, dy, cv.CV_32F, 0, 1, 3, 1, 0);
  cv.magnitude(dx, dy, mag); cv.normalize(mag, mag, 0, 255, cv.NORM_MINMAX); mag.convertTo(mag8, cv.CV_8U);
  const otsu = cv.threshold(mag8, tmp, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
  high = Math.max(30, Math.round(otsu)); low = Math.max(5, Math.round(0.5 * high));
  dx.delete(); dy.delete(); mag.delete(); mag8.delete(); tmp.delete();
  const edges = new cv.Mat(); cv.Canny(gray, edges, low, high, 3, true); gray.delete();
  const closeK = (els.bClose ? +els.bClose.value : 5) || 5;
  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(closeK, closeK));
  const closed = new cv.Mat(); cv.morphologyEx(edges, closed, cv.MORPH_CLOSE, kernel); kernel.delete(); edges.delete();
  const contours = new cv.MatVector(); const hierarchy = new cv.Mat();
  cv.findContours(closed, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  closed.delete(); hierarchy.delete();
  const imgArea = rows * cols; const minArea = ((els.bMinArea ? +els.bMinArea.value : 2) / 100) * imgArea;
  const cx0 = cols / 2, cy0 = rows / 2; const diag = Math.hypot(cols, rows); const centerR = (els.bCenterR ? +els.bCenterR.value : 40) / 100;
  let bestIdx = -1, bestScore = -1, bestArea = 0;
  for (let i = 0; i < contours.size(); i++) {
    const cnt = contours.get(i);
    const area = cv.contourArea(cnt, false);
    if (area < minArea) { cnt.delete(); continue; }
    const m = cv.moments(cnt, false);
    if (m.m00 === 0) { cnt.delete(); continue; }
    const cx = m.m10 / m.m00, cy = m.m01 / m.m00;
    const dist = Math.hypot(cx - cx0, cy - cy0) / diag;
    const score = area * (dist <= centerR ? 1.2 : 1.0) * (1.0 - 0.6 * dist);
    if (score > bestScore) { bestScore = score; bestIdx = i; bestArea = area; }
    cnt.delete();
  }
  let mask = cv.Mat.zeros(rows, cols, cv.CV_8U); let overlay = rgb.clone();
  if (bestIdx >= 0) {
    const best = contours.get(bestIdx);
    const perim = cv.arcLength(best, true);
    const epsPct = (els.bEps ? +els.bEps.value : 1.5) / 100.0;
    const epsilon = Math.max(0.5, epsPct * perim);
    const approx = new cv.Mat(); cv.approxPolyDP(best, approx, epsilon, true);
    const mv = new cv.MatVector(); mv.push_back(approx); cv.fillPoly(mask, mv, new cv.Scalar(255)); mv.delete();
    const fillRGB = new cv.Mat(rows, cols, cv.CV_8UC3, new cv.Scalar(0, 255, 0, 0));
    const filled = new cv.Mat(); fillRGB.copyTo(filled, mask);
    const blended = new cv.Mat(); cv.addWeighted(overlay, 0.65, filled, 0.35, 0, blended);
    cv.polylines(blended, approx, true, new cv.Scalar(0, 255, 0, 255), 2, cv.LINE_AA);
    overlay.delete(); fillRGB.delete(); filled.delete(); overlay = blended;
    best.delete(); approx.delete();
  }
  contours.delete(); rgb.delete();
  return { mask, overlay };
}

// ---------- Export All (Task 3) ----------
els.exportAll3.onclick = async () => {
  await waitForCV();
  if (!state.images.length) { setStatus('Load images first, then try Export All (Task 3).'); return; }
  const off = document.createElement('canvas');
  function saveMatAsPNG(mat, filename) {
    off.width = mat.cols; off.height = mat.rows;
    cv.imshow(off, mat);
    const a = document.createElement('a');
    a.download = filename; a.href = off.toDataURL('image/png'); try { a.click(); } catch {}
  }
  setStatus('Exporting Task 3… your browser may ask to allow multiple downloads.');
  for (let i = 0; i < state.images.length; i++) {
    const item = state.images[i];
    const srcRGBA = getSrcMatFromBitmap(item.bitmap);
    const stem = item.name.replace(/\.[^.]+$/, '');
    try {
      const { mask, overlay } = computeBoundaryMaskAndOverlay(srcRGBA);
      saveMatAsPNG(mask,    `${stem}__mask.png`);
      saveMatAsPNG(overlay, `${stem}__boundary.png`);
      mask.delete(); overlay.delete();
    } catch (e) { console.error('Task 3 export error on', item.name, e); setStatus(`Task 3 export error on ${item.name}: ${e?.message || e}`); }
    finally { srcRGBA.delete(); }
    await new Promise(r => setTimeout(r, 120));
  }
  setStatus(`Exported Task 3 outputs for ${state.images.length} image(s). If nothing downloaded, allow multiple downloads for this site.`);
};

// ---------- Export All (Task 4): ArUco mask + overlay ----------
els.exportAll4.onclick = async () => {
  await waitForCV();
  if (!state.images.length) { setStatus('Load images first, then try Export All (Task 4).'); return; }
  if (!arucoAvailable()) { setStatus('This OpenCV.js build does not include ArUco. Please switch to a build with aruco module.'); return; }

  const off = document.createElement('canvas');
  function saveMatAsPNG(mat, filename) {
    off.width = mat.cols; off.height = mat.rows;
    cv.imshow(off, mat);
    const a = document.createElement('a');
    a.download = filename; a.href = off.toDataURL('image/png'); try { a.click(); } catch {}
  }

  setStatus('Exporting Task 4… your browser may ask to allow multiple downloads.');
  for (let i = 0; i < state.images.length; i++) {
    const item = state.images[i];
    const srcRGBA = getSrcMatFromBitmap(item.bitmap);
    const stem = item.name.replace(/\.[^.]+$/, '');
    try {
      // Full-resolution ArUco segmentation (no half-res here)
      const result = (function segFull(srcRGBA) {
        const rgb = new cv.Mat(); cv.cvtColor(srcRGBA, rgb, cv.COLOR_RGBA2RGB);
        const gray = new cv.Mat(); cv.cvtColor(srcRGBA, gray, cv.COLOR_RGBA2GRAY);
        if (!arucoAvailable()) { gray.delete(); rgb.delete(); return null; }
        const det = detectAruco(gray, els.arucoDict?.value || 'DICT_4X4_50');
        gray.delete();
        if (!det) { rgb.delete(); return { mask: cv.Mat.zeros(srcRGBA.rows, srcRGBA.cols, cv.CV_8U), overlay: rgb }; }
        const pts = cornersToPointList(det.cornersVec, !!(els.arucoUseCorners?.checked));
        det.idsMat.delete(); det.cornersVec.delete();
        const mask = cv.Mat.zeros(srcRGBA.rows, srcRGBA.cols, cv.CV_8U);
        let overlay = rgb.clone();
        if (pts.length >= 3) {
          const contour = pointsToContourMat(pts);
          const mv = new cv.MatVector(); mv.push_back(contour);
          cv.fillPoly(mask, mv, new cv.Scalar(255)); mv.delete(); contour.delete();
          const dil = Math.max(0, +(els.arucoDilate?.value || 0));
          if (dil > 0) { const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(dil, dil)); const tmp = new cv.Mat(); cv.dilate(mask, tmp, k); mask.delete(); k.delete(); mask = tmp; }
          const fillRGB = new cv.Mat(srcRGBA.rows, srcRGBA.cols, cv.CV_8UC3, new cv.Scalar(0, 255, 0, 0));
          const filled = new cv.Mat(); fillRGB.copyTo(filled, mask);
          const blended = new cv.Mat(); cv.addWeighted(overlay, 0.65, filled, 0.35, 0, blended);
          const outline = pointsToContourMat(pts);
          cv.polylines(blended, outline, true, new cv.Scalar(0,255,0,255), 2, cv.LINE_AA);
          overlay.delete(); fillRGB.delete(); filled.delete(); outline.delete();
          overlay = blended;
        }
        rgb.delete();
        return { mask, overlay };
      })(srcRGBA);

      if (result) {
        saveMatAsPNG(result.mask,    `${stem}__aruco_mask.png`);
        saveMatAsPNG(result.overlay, `${stem}__aruco_boundary.png`);
        result.mask.delete(); result.overlay.delete();
      } else {
        setStatus('ArUco not available in this OpenCV build.');
      }
    } catch (e) { console.error('Task 4 export error on', item.name, e); setStatus(`Task 4 export error on ${item.name}: ${e?.message || e}`); }
    finally { srcRGBA.delete(); }
    await new Promise(r => setTimeout(r, 140));
  }
  setStatus(`Exported Task 4 outputs for ${state.images.length} image(s). If nothing downloaded, enable multiple downloads.`);
};
