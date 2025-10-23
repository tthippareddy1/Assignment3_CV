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
    save:        document.getElementById('save'),
    exportAll1:  document.getElementById('exportAll1'),
    video:       document.getElementById('video'),
    canvas:      document.getElementById('canvas'),
    status:      document.getElementById('status')
  };
  
  // ---------- App state ----------
  let state = {
    images: [],          // [{name, bitmap}]
    index: 0,
    stream: null,        // MediaStream
    source: 'none',      // 'dataset' | 'webcam' | 'none'
    mode: 'original'
  };
  
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
  }
  function showVideo(show) { els.video.style.display = show ? 'block' : 'none'; }
  function showCanvas(show) { els.canvas.style.display = show ? 'block' : 'none'; }
  
  // ---------- Robust image loading ----------
  // Tries createImageBitmap; falls back to HTMLImageElement + canvas;
  // downscales huge images to avoid memory issues; rejects HEIC/HEIF.
  const MAX_SIDE = 4096; // prevent huge GPU bitmaps
  
  async function fileToBitmap(file) {
    // Reject formats that browsers typically can't decode
    if (/hei[cf]/i.test(file.type) || /\.heic$|\.heif$/i.test(file.name)) {
      throw new Error('HEIC/HEIF is not supported in browsers. Please convert to JPG/PNG.');
    }
  
    // 1) Fast path
    try {
      // some browsers choke on the orientation hint; omit it for compatibility
      return await createImageBitmap(file);
    } catch (_) {
      // 2) Fallback via Image + canvas
      const url = URL.createObjectURL(file);
      try {
        const img = await new Promise((resolve, reject) => {
          const el = new Image();
          el.onload = () => resolve(el);
          el.onerror = () => reject(new Error('Image decode failed'));
          el.src = url;
        });
  
        // Scale if too large
        const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
        const w = Math.max(1, Math.round(img.naturalWidth * scale));
        const h = Math.max(1, Math.round(img.naturalHeight * scale));
        const c = document.createElement('canvas');
        c.width = w; c.height = h;
        const ctx = c.getContext('2d');
        ctx.drawImage(img, 0, 0, w, h);
        const bmp = await createImageBitmap(c);
        return bmp;
      } finally {
        URL.revokeObjectURL(url);
      }
    }
  }
  
  async function filesToBitmaps(files) {
    const out = [];
    const errors = [];
    for (const f of files) {
      try {
        const bmp = await fileToBitmap(f);
        out.push({ name: f.name, bitmap: bmp });
      } catch (e) {
        errors.push(`${f.name}: ${e.message}`);
      }
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
    if (!els.halfRes.checked) return src.clone();
    const dst = new cv.Mat();
    cv.resize(src, dst, new cv.Size(Math.round(src.cols/2), Math.round(src.rows/2)), 0, 0, cv.INTER_AREA);
    return dst;
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
  
    if (+els.edgeBoost.value > 0) {
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
  
    // suppress weak gradients to make edges visible
    const mask = new cv.Mat();
    cv.threshold(mag8, mask, 20, 255, cv.THRESH_BINARY);
    if (+els.edgeBoost.value > 0) {
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
  
    if (+els.edgeBoost.value > 0) {
      const kernel = cv.Mat.ones(3,3,cv.CV_8U);
      cv.dilate(lap8, lap8, kernel);
      kernel.delete();
      if (+els.edgeBoost.value === 2) cv.equalizeHist(lap8, lap8);
    }
  
    work.delete(); gray.delete(); blur.delete(); lap.delete();
    return lap8;
  }
  
  // ---------- Rendering ----------
  async function renderCurrent() {
    if (state.source === 'dataset' && state.images.length) {
      showVideo(false); showCanvas(true);
      await waitForCV();
  
      const bmp = state.images[state.index].bitmap;
      const mode = state.mode;
      const src = getSrcMatFromBitmap(bmp); // RGBA
      let out = null;
  
      try {
        if (mode === 'original') {
          drawBitmapToCanvas(bmp);
          overlayModeLabel('ORIGINAL');
        } else if (mode === 'grad_mag') {
          out = gradMagnitudeMat(src); imshowFit(out); overlayModeLabel('GRADIENT MAG');
        } else if (mode === 'grad_angle') {
          out = gradAngleMat(src);    imshowFit(out); overlayModeLabel('GRADIENT ANGLE');
        } else if (mode === 'log') {
          out = logMat(src);          imshowFit(out); overlayModeLabel('LoG');
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
  
      setStatus(`Image ${state.index+1}/${state.images.length} — Mode: ${state.mode} — Source: dataset`);
      els.save.disabled = true; // Save button saves exactly the canvas, but we'll enable only for processed views
      if (mode !== 'original') els.save.disabled = false;
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
    // A small built-in PNG so you can verify the pipeline immediately
    const demoDataURL =
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABkCAYAAABoQy0/AAAACXBIWXMAAAsSAAALEgHS3X78AAABtUlEQVR4nO3RMQEAIAzAsFv/0m0kF3m4JY2nQbdk2iEJ8Qe3GgAAAAAAAAAAAAAAAAAAgL3M3wqf9y3iXWg+0l7Yf8m1oG6D2wE1qj8aXy9w8g7dD3vQb1m4R6k9sBNao/Gn8vcPJO3Q971G9ZuEepPbATWqPxpfL3DyDt0Pe9BvWbhHqT2wE1qj8aXy9w8g7dD3vQb1m4R6k9sBNao/Gn8vcPJO3Q971G9ZuEepPbATWqPxpfL3DyDt0Pe9BvWbhHqT2wE1qj8aXy9w8g7dD3vQb1m4R6k9sBNao/Gn8vcPJO3Q971G9ZuEepPbATWqPxpfL3DyDt0Pe9BvWbhHqT2wE1qj8aXy9w8g7dD3vQb1m4R6k9sBNao/Gn8vcPJO3Q971G9ZuEepPbATWqPxpfL3DyDt0Pe9BvWbhHqT2wE1qj8aXy9w8g7dD3vQb1m4R6k9sBNao/Gn8vcPJO3Q971G9ZuEepPbATWqPxpfL3DyDs3C1k2iEJ8Qe3GgAAAAAAAAAAAAAAAOA3GUM1c7Jt0AAAAABJRU5ErkJggg==";
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
  
  els.mode.onchange   = async () => { state.mode = els.mode.value; await renderCurrent(); };
  els.edgeBoost.oninput = async () => { await renderCurrent(); };
  els.halfRes.onchange  = async () => { await renderCurrent(); };
  
  // ---------- Save current canvas ----------
  els.save.onclick = () => {
    const srcW = els.canvas.width, srcH = els.canvas.height;
    if (!srcW || !srcH) { setStatus('Nothing to save.'); return; }
    const link = document.createElement('a');
    const base = (state.images[state.index]?.name || 'capture').replace(/\.[^.]+$/, '');
    link.download = `${base}__${state.mode}.png`;
    link.href = els.canvas.toDataURL('image/png'); // export exactly what you see
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
  