/**
 * LuminaPV - Charting & Canvas Visualizers
 * Handles HiDPI canvas rendering for the batch damage sparkline, defect distribution histogram, and floating tooltips.
 */

/* ---------- canvas & tooltip helpers ---------- */
var tt = null;

function showTip(text, x, y) {
  if (!tt) tt = $("#tt");
  if (!tt) return;
  tt.textContent = text;
  tt.classList.add("show");
  var pad = 10, tw = tt.offsetWidth, th = tt.offsetHeight;
  var left = x + 12, top = y + 12;
  if (left + tw + pad > window.innerWidth) left = x - tw - 12;
  if (top + th + pad > window.innerHeight) top = y - th - 12;
  tt.style.left = Math.max(pad, left) + "px";
  tt.style.top = Math.max(pad, top) + "px";
}

function hideTip() {
  if (!tt) tt = $("#tt");
  if (tt) tt.classList.remove("show");
}

function prepCanvas(c) {
  var dpr = window.devicePixelRatio || 1;
  var w = c.clientWidth || c.width, h = c.clientHeight || c.height;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
  }
  var ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx: ctx, w: w, h: h };
}

/* ---------- sparkline (batch damage profile) ---------- */
var sparkPts = [];

function drawSpark() {
  var c = $("#spark");
  if (!c) return;
  var p = prepCanvas(c);
  var vals = state.modules.map(function(m) { return +m.module_damage_percentage || 0; });
  sparkPts = [];
  var padL = 8, padR = 8, padT = 8, padB = 10;
  p.ctx.strokeStyle = cssVar("--border-strong");
  p.ctx.lineWidth = 1;
  p.ctx.beginPath();
  p.ctx.moveTo(padL, p.h - padB + 0.5);
  p.ctx.lineTo(p.w - padR, p.h - padB + 0.5);
  p.ctx.stroke();

  if (!vals.length) {
    p.ctx.fillStyle = cssVar("--muted2");
    p.ctx.font = "10px " + cssVar("--mono");
    p.ctx.fillText("aguardando payload", padL, p.h / 2);
    return;
  }

  var max = Math.max.apply(null, vals.concat([10]));
  var stepX = vals.length > 1 ? (p.w - padL - padR) / (vals.length - 1) : 0;
  var accent = cssVar("--accent"), defect = cssVar("--defect");

  p.ctx.beginPath();
  vals.forEach(function(v, i) {
    var x = padL + i * stepX;
    var y = (p.h - padB) - (v / max) * (p.h - padT - padB);
    sparkPts.push({ x: x, y: y, label: state.modules[i].module_id, value: pct(v) });
    if (i) p.ctx.lineTo(x, y); else p.ctx.moveTo(x, y);
  });
  p.ctx.strokeStyle = accent;
  p.ctx.lineWidth = 1.5;
  p.ctx.stroke();

  p.ctx.lineTo(padL + (vals.length - 1) * stepX, p.h - padB);
  p.ctx.lineTo(padL, p.h - padB);
  p.ctx.closePath();
  p.ctx.fillStyle = hexToRgba(accent, 0.18);
  p.ctx.fill();

  sparkPts.forEach(function(pt) {
    p.ctx.beginPath();
    p.ctx.rect(pt.x - 1.5, pt.y - 1.5, 3, 3);
    p.ctx.fillStyle = (pt.value !== "—" && parseFloat(String(pt.value).replace(",", ".")) >= 15) ? defect : accent;
    p.ctx.fill();
  });
}

/* ---------- defect distribution chart ---------- */
var distPts = [];

function distData() {
  var byType = {};
  DEFECT_KEYS.forEach(function(k) { byType[k] = 0; });
  state.modules.forEach(function(m) {
    defects(m).forEach(function(c) {
      if (byType[c.defect_type] === undefined) byType[c.defect_type] = 0;
      byType[c.defect_type]++;
    });
  });
  var keys = DEFECT_KEYS.filter(function(k) { return byType[k] > 0; });
  Object.keys(byType).forEach(function(k) {
    if (keys.indexOf(k) < 0 && byType[k] > 0) keys.push(k);
  });
  return keys.map(function(k) {
    return {
      key: k,
      label: DEFECT_LABEL[k] || k.toUpperCase(),
      value: byType[k],
      color: cssVar(DEFECT_VAR[k] || "--d1")
    };
  });
}

function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

function drawDist(progress) {
  var c = $("#distChart");
  if (!c) return;
  var p = prepCanvas(c);
  var data = state.distValues || distData();
  distPts = [];
  var legend = $("#distLegend");

  if (!data.length) {
    p.ctx.fillStyle = cssVar("--muted2");
    p.ctx.font = "11px " + cssVar("--mono");
    p.ctx.fillText("nenhuma anomalia no lote atual", 10, p.h / 2);
    if (legend) legend.innerHTML = '<div style="color:var(--muted2)">sem dados</div>';
    return;
  }

  var max = Math.max.apply(null, data.map(function(d) { return d.value; }));
  var x0 = 10, y0 = p.h - 22, barH = p.h - 42;
  var gap = 10, bw = Math.max(18, Math.min(88, (p.w - x0 * 2 - (data.length - 1) * gap) / data.length));

  p.ctx.strokeStyle = cssVar("--border-strong");
  p.ctx.beginPath();
  p.ctx.moveTo(x0, y0 + 0.5);
  p.ctx.lineTo(p.w - 10, y0 + 0.5);
  p.ctx.stroke();

  var total = data.reduce(function(a, b) { return a + b.value; }, 0);
  data.forEach(function(d, i) {
    var hh = (d.value / max) * barH * progress;
    var x = x0 + i * (bw + gap);
    p.ctx.fillStyle = d.color;
    p.ctx.fillRect(Math.round(x) + 0.5, Math.round(y0 - hh) + 0.5, Math.round(bw), Math.max(1, Math.round(hh)));
    p.ctx.fillStyle = cssVar("--fg-strong");
    p.ctx.font = "10px " + cssVar("--mono");
    p.ctx.fillText(String(Math.round(d.value * progress)), x, y0 - hh - 5);
    p.ctx.fillStyle = cssVar("--muted2");
    p.ctx.fillText(d.label.length > 13 ? d.label.slice(0, 12) + "." : d.label, x, y0 + 14);
    distPts.push({
      x: x + bw / 2,
      y: y0 - hh,
      rect: [x, y0 - hh, bw, Math.max(1, hh)],
      label: d.label,
      value: d.value,
      share: total ? d.value / total * 100 : 0,
      idx: i
    });
  });

  if (legend) {
    legend.innerHTML = data.map(function(d) {
      return '<div><i style="background:' + d.color + '"></i>' + esc(d.label) + '<b>' + int(d.value) + '</b></div>';
    }).join("") + '<div style="border-top:1px solid var(--border);padding-top:5px;color:var(--muted2)"><span>TOTAL</span><b style="margin-left:auto;color:var(--fg-strong)">' + int(total) + '</b></div>';
  }
}

function animateDist() {
  var dur = 800, start = performance.now();
  function frame(ts) {
    var t = Math.min((ts - start) / dur, 1);
    drawDist(easeOutCubic(t));
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
