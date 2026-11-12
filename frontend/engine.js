/* BlightSpot on-device engine: runs the exported U-Net in the browser with
 * onnxruntime-web and reproduces the server's report (backend/app/inference.py)
 * so the UI can work with no backend at all — photos never leave the device.
 */
"use strict";

const BlightEngine = (() => {
  const ORT_VERSION = "1.20.1";
  const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  let session = null;
  let meta = null;
  let loading = null;

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.crossOrigin = "anonymous";
      s.onload = resolve;
      s.onerror = () => reject(new Error(`failed to load ${src}`));
      document.head.append(s);
    });
  }

  async function loadMeta() {
    if (!meta) {
      const r = await fetch("model/meta.json");
      if (!r.ok) throw new Error(`model/meta.json: HTTP ${r.status}`);
      meta = await r.json();
    }
    return meta;
  }

  /** Load onnxruntime-web + the model once; later calls reuse the same promise. */
  function load() {
    loading ??= (async () => {
      await loadMeta();
      if (!window.ort) await loadScript(`${ORT_BASE}ort.min.js`);
      ort.env.wasm.wasmPaths = ORT_BASE;
      // Threads need cross-origin isolation, which static hosts like GitHub Pages lack.
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      session = await ort.InferenceSession.create("model/leaf_unet.onnx", {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      return meta;
    })().catch((e) => { loading = null; throw e; });
    return loading;
  }

  /* ------------------------------------------------------------ pixels */

  /** Centre-crop any drawable to an n×n RGB Uint8 array (mirrors preprocess()). */
  function cropToRGB(src, n, ctx) {
    const w = src.videoWidth || src.naturalWidth || src.width;
    const h = src.videoHeight || src.naturalHeight || src.height;
    const s = Math.min(w, h);
    ctx.canvas.width = ctx.canvas.height = n;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(src, (w - s) / 2, (h - s) / 2, s, s, 0, 0, n, n);
    const rgba = ctx.getImageData(0, 0, n, n).data;
    const rgb = new Uint8Array(n * n * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
      rgb[j] = rgba[i]; rgb[j + 1] = rgba[i + 1]; rgb[j + 2] = rgba[i + 2];
    }
    return rgb;
  }

  // Test-time augmentation views. Each maps a view pixel (y, x) to the source
  // pixel it samples, so the same map un-warps predictions back onto the image.
  const VIEWS = {
    identity: (y, x) => [y, x],
    hflip: (y, x, n) => [y, n - 1 - x],
    vflip: (y, x, n) => [n - 1 - y, x],
    rot90: (y, x, n) => [x, n - 1 - y], // torch.rot90(k=1, dims=(-2, -1))
  };

  function toTensor(rgb, n, views) {
    const plane = n * n;
    const data = new Float32Array(views.length * 3 * plane);
    views.forEach((map, v) => {
      const base = v * 3 * plane;
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          const [sy, sx] = map(y, x, n);
          const src = (sy * n + sx) * 3;
          const dst = y * n + x;
          data[base + dst] = rgb[src] / 255;
          data[base + plane + dst] = rgb[src + 1] / 255;
          data[base + 2 * plane + dst] = rgb[src + 2] / 255;
        }
      }
    });
    return new ort.Tensor("float32", data, [views.length, 3, n, n]);
  }

  /** Softmax each view's logits, un-warp, and average → probs[C][n*n]. */
  function averageProbs(logits, n, views, C) {
    const plane = n * n;
    const probs = Array.from({ length: C }, () => new Float32Array(plane));
    const e = new Float32Array(C);
    views.forEach((map, v) => {
      const base = v * C * plane;
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          const p = y * n + x;
          let max = -Infinity;
          for (let c = 0; c < C; c++) max = Math.max(max, logits[base + c * plane + p]);
          let sum = 0;
          for (let c = 0; c < C; c++) sum += (e[c] = Math.exp(logits[base + c * plane + p] - max));
          const [sy, sx] = map(y, x, n);
          const q = sy * n + sx;
          for (let c = 0; c < C; c++) probs[c][q] += e[c] / sum / views.length;
        }
      }
    });
    return probs;
  }

  function argmax(probs, plane) {
    const mask = new Uint8Array(plane);
    for (let p = 0; p < plane; p++) {
      let best = 0;
      for (let c = 1; c < probs.length; c++) if (probs[c][p] > probs[best][p]) best = c;
      mask[p] = best;
    }
    return mask;
  }

  /** Logits → argmax mask directly (live mode: one view, no probabilities kept). */
  function argmaxLogits(logits, plane, C) {
    const mask = new Uint8Array(plane);
    for (let p = 0; p < plane; p++) {
      let best = 0;
      for (let c = 1; c < C; c++) if (logits[c * plane + p] > logits[best * plane + p]) best = c;
      mask[p] = best;
    }
    return mask;
  }

  /* ------------------------------------------------------------ report (port of inference.report) */

  function components(binary, n, minPixels = 3) {
    const seen = new Uint8Array(binary.length);
    const queue = new Int32Array(binary.length);
    let count = 0;
    for (let start = 0; start < binary.length; start++) {
      if (!binary[start] || seen[start]) continue;
      let head = 0, tail = 0, size = 0;
      queue[tail++] = start;
      seen[start] = 1;
      while (head < tail) {
        const p = queue[head++];
        size++;
        const y = (p / n) | 0, x = p % n;
        const nbrs = [y + 1 < n && p + n, y > 0 && p - n, x + 1 < n && p + 1, x > 0 && p - 1];
        for (const q of nbrs) {
          if (q !== false && binary[q] && !seen[q]) { seen[q] = 1; queue[tail++] = q; }
        }
      }
      if (size >= minPixels) count++;
    }
    return count;
  }

  const round = (v, d) => Math.round(v * 10 ** d) / 10 ** d;

  function severityLabel(pct) {
    if (pct < 0.5) return "healthy";
    if (pct < 5) return "mild";
    if (pct < 15) return "moderate";
    return "severe";
  }

  function report(mask, probs, n) {
    const { classes, disease_ids, disease_info, healthy_info } = meta;
    const LEAF = classes.indexOf("leaf");
    const plane = n * n;
    let leafTotal = 0;
    for (let p = 0; p < plane; p++) if (mask[p] >= LEAF) leafTotal++;
    const leafFrac = leafTotal / plane;

    const diseases = [];
    for (const id of disease_ids) {
      const c = classes.indexOf(id);
      const binary = new Uint8Array(plane);
      let px = 0, confSum = 0;
      for (let p = 0; p < plane; p++) {
        if (mask[p] === c) { binary[p] = 1; px++; if (probs) confSum += probs[c][p]; }
      }
      if (!px) continue;
      const entry = {
        id, name: disease_info[id].name, pixels: px,
        affected_pct: round((100 * px) / Math.max(leafTotal, 1), 2),
        lesion_count: components(binary, n),
      };
      if (probs) entry.confidence = round(confSum / px, 3);
      diseases.push(entry);
    }
    diseases.sort((a, b) => b.pixels - a.pixels);
    const affected = diseases.reduce((s, d) => s + d.pixels, 0);
    const totalPct = leafTotal ? (100 * affected) / leafTotal : 0;
    const significant = diseases.filter((d) => d.affected_pct >= 0.5);
    const primary = significant.length ? significant[0].id : "healthy";

    const warnings = [];
    if (leafFrac < 0.04) warnings.push("Very little leaf detected. Fill the frame with a single leaf.");
    let confidence = null;
    if (probs && leafTotal) {
      let s = 0;
      for (let p = 0; p < plane; p++) {
        if (mask[p] < LEAF) continue;
        let m = 0;
        for (let c = 0; c < probs.length; c++) m = Math.max(m, probs[c][p]);
        s += m;
      }
      confidence = round(s / leafTotal, 3);
      if (confidence < 0.8) warnings.push("Low model confidence: this photo looks unlike the synthetic training data.");
    }

    return {
      diagnosis: leafTotal && leafFrac >= 0.01 ? primary : "no_leaf",
      diagnosis_info: disease_info[primary] || healthy_info,
      severity: leafTotal ? severityLabel(totalPct) : "unknown",
      affected_pct: round(totalPct, 2),
      leaf_pixels: leafTotal,
      leaf_coverage_pct: round(100 * leafFrac, 1),
      lesion_pixels: affected,
      lesion_count: diseases.reduce((s, d) => s + d.lesion_count, 0),
      diseases,
      confidence,
      warnings,
    };
  }

  /* ------------------------------------------------------------ rendering (port of overlay/mask/confidence_map) */

  function toPngB64(rgb, n, ctx) {
    ctx.canvas.width = ctx.canvas.height = n;
    const img = ctx.createImageData(n, n);
    for (let p = 0, j = 0; p < n * n; p++, j += 3) {
      img.data[4 * p] = rgb[j]; img.data[4 * p + 1] = rgb[j + 1]; img.data[4 * p + 2] = rgb[j + 2]; img.data[4 * p + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return ctx.canvas.toDataURL("image/png").split(",")[1];
  }

  function renderOverlay(rgb, mask, n) {
    const { palette, classes, disease_ids } = meta;
    const disease = new Set(disease_ids.map((id) => classes.indexOf(id)));
    const out = new Uint8Array(rgb);
    const leafish = (y, x) => mask[((y + n) % n) * n + ((x + n) % n)] !== 0; // np.roll wraps
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const p = y * n + x, j = 3 * p, c = mask[p];
        if (disease.has(c)) for (let k = 0; k < 3; k++) out[j + k] = Math.min(255, 0.3 * rgb[j + k] + 0.7 * palette[c][k]);
        if (c !== 0 && !(leafish(y - 1, x) && leafish(y + 1, x) && leafish(y, x - 1) && leafish(y, x + 1))) {
          out[j] = out[j + 1] = out[j + 2] = 255;
        }
      }
    }
    return out;
  }

  function renderMask(mask) {
    const out = new Uint8Array(mask.length * 3);
    mask.forEach((c, p) => out.set(meta.palette[c], 3 * p));
    return out;
  }

  const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  function renderConfidence(probs, plane) {
    const C = probs.length, out = new Uint8Array(plane * 3);
    for (let p = 0; p < plane; p++) {
      let m = 0;
      for (let c = 0; c < C; c++) m = Math.max(m, probs[c][p]);
      const t = Math.min(1, Math.max(0, (m - 1 / C) / (1 - 1 / C))) * (VIRIDIS.length - 1);
      const i = Math.min(Math.floor(t), VIRIDIS.length - 2), f = t - i;
      for (let k = 0; k < 3; k++) out[3 * p + k] = VIRIDIS[i][k] * (1 - f) + VIRIDIS[i + 1][k] * f;
    }
    return out;
  }

  /* ------------------------------------------------------------ public API */

  const scratch = document.createElement("canvas").getContext("2d", { willReadFrequently: true });

  /** Full analysis with TTA. Returns the same JSON shape as POST /analyze. */
  async function analyze(source, { tta = true } = {}) {
    await load();
    const t0 = performance.now();
    const n = meta.img_size, C = meta.classes.length;
    const rgb = cropToRGB(source, n, scratch);
    const views = tta ? Object.values(VIEWS) : [VIEWS.identity];
    const out = await session.run({ [session.inputNames[0]]: toTensor(rgb, n, views) });
    const probs = averageProbs(out[session.outputNames[0]].data, n, views, C);
    const mask = argmax(probs, n * n);
    return {
      ...report(mask, probs, n),
      images: {
        original: toPngB64(rgb, n, scratch),
        overlay: toPngB64(renderOverlay(rgb, mask, n), n, scratch),
        mask: toPngB64(renderMask(mask), n, scratch),
        confidence: toPngB64(renderConfidence(probs, n * n), n, scratch),
      },
      inference_ms: Math.round(performance.now() - t0),
      tta,
      engine: "on-device",
      disclaimer: meta.disclaimer,
    };
  }

  const liveScratch = document.createElement("canvas").getContext("2d", { willReadFrequently: true });

  /** Fast single-view segmentation for live camera frames. */
  async function segment(source, n = 128) {
    await load();
    const rgb = cropToRGB(source, n, liveScratch);
    const out = await session.run({ [session.inputNames[0]]: toTensor(rgb, n, [VIEWS.identity]) });
    const mask = argmaxLogits(out[session.outputNames[0]].data, n * n, meta.classes.length);
    return { mask, n, summary: report(mask, null, n) };
  }

  return {
    load, loadMeta, analyze, segment,
    get ready() { return !!session; },
    get meta() { return meta; },
    // exposed for tests / debugging
    _internals: { VIEWS, components, report, severityLabel },
  };
})();
