/**
 * LuminaPV - State Management & Geometry Utilities
 * Holds active application state, formatting filters, topological calculators, and system logger.
 */

var state = {
  modules: [],
  summary: null,
  activeId: null,
  filter: "all",
  files: [],
  pendingFiles: [],
  retries: 0,
  errorId: null,
  lastAttempt: null,
  backoffTimer: null,
  lastError: null,
  controller: null,
  loading: false,
  startedAt: 0,
  log: [],
  cellsByIndex: {},
  selectedCell: null,
  zoom: { scale: 1, tx: 0, ty: 0, fitScale: 1, natW: 0, natH: 0, raster: false },
  distAnimated: false,
  distValues: null,
  cluster: null,
  clusterPollingInterval: null,
  lastLeaderId: null,
  lastNodesState: {},
  auth: {
    token: null,
    user: null,
    authenticated: false,
    loading: false,
    mode: "login",
    error: null
  }
};

/* ---------- formatting & string helpers ---------- */
function nf(n, d) {
  if (n === null || n === undefined || !isFinite(n)) return "—";
  return Number(n).toFixed(d === undefined ? 1 : d).replace(".", ",");
}

function pct(n, d) {
  if (n === null || n === undefined || !isFinite(n)) return "—";
  return nf(n, d) + "%";
}

function int(n) {
  if (n === null || n === undefined || !isFinite(n)) return "—";
  return new Intl.NumberFormat("pt-BR").format(n);
}

function pad2(n) {
  return String(n).padStart(2, "0");
}

function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function now() {
  var d = new Date();
  return pad2(d.getHours()) + ":" + pad2(d.getMinutes()) + ":" + pad2(d.getSeconds());
}

function debounce(fn, ms) {
  var t;
  return function() {
    var a = arguments, c = this;
    clearTimeout(t);
    t = setTimeout(function() { fn.apply(c, a); }, ms);
  };
}

function live(msg) {
  var el = $("#live");
  if (!el) return;
  el.textContent = "";
  setTimeout(function() { el.textContent = msg; }, 30);
}

function hexToRgba(hex, a) {
  var h = String(hex || "").replace("#", "").trim();
  if (h.length === 3) h = h.charAt(0) + h.charAt(0) + h.charAt(1) + h.charAt(1) + h.charAt(2) + h.charAt(2);
  if (h.length !== 6) return "rgba(0,122,204," + a + ")";
  var n = parseInt(h, 16);
  return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
}

function cssVar(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

/* ---------- logging ---------- */
function log(level, msg) {
  state.log.push({ t: now(), level: level, msg: msg });
  renderLog();
}

function renderLog() {
  var out = $("#logOut");
  if (!out) return;
  if (!state.log.length) {
    out.innerHTML = '<div><span class="msg" style="color:var(--muted2)">Nenhuma operação registrada nesta sessão.</span></div>';
    return;
  }
  out.innerHTML = state.log.slice().reverse().map(function(e) {
    var lv = e.level === "error" ? "err" : (e.level === "ok" ? "ok" : (e.level === "req" ? "req" : (e.level === "warn" ? "warn" : (e.level === "info" ? "info" : ""))));
    return '<div><time class="mono">' + e.t + '</time><span class="lvl ' + lv + '">' + esc(e.level.toUpperCase()) + '</span><span class="msg">' + esc(e.msg) + '</span></div>';
  }).join("");
}

/* ---------- grid geometry & module queries ---------- */
function gridDims(m) {
  if (!m) return { rows: 1, cols: 1 };
  if (m.rows && m.cols) return { rows: +m.rows, cols: +m.cols };
  var total = +m.total_cells || 0;
  var cells = m.cells || [];
  if (cells.length === total && total > 0) {
    var mr = 0, mc = 0;
    cells.forEach(function(c) {
      mr = Math.max(mr, +c.row || 0);
      mc = Math.max(mc, +c.col || 0);
    });
    if (mc + 1 > 0 && (mr + 1) * (mc + 1) === total) return { rows: mr + 1, cols: mc + 1 };
  }
  if (total === 60) return { rows: 6, cols: 10 };
  if (total === 72) return { rows: 6, cols: 12 };
  if (total === 120 || total === 144) return { rows: 12, cols: 12 };
  if (total > 0 && total % 6 === 0) return { rows: 6, cols: total / 6 };
  if (total > 0 && total % 10 === 0) return { rows: 10, cols: total / 10 };
  var cols = Math.max(1, Math.min(12, total || 1));
  return { rows: Math.max(1, Math.ceil((total || 1) / cols)), cols: cols };
}

function defects(m) {
  return (m && m.cells ? m.cells : []).filter(function(c) { return c.is_defective; });
}

function severity(m) {
  if (!m) return { key: "—", cls: "", label: "—" };
  var d = +m.module_damage_percentage;
  if (!isFinite(d)) return { key: "—", cls: "", label: "—" };
  if (d >= 15) return { key: "CRITICO", cls: "critical", label: "CRÍTICO" };
  if (d >= 5) return { key: "DEGRADADO", cls: "degraded", label: "DEGRADADO" };
  return { key: "NOMINAL", cls: "nominal", label: "NOMINAL" };
}

function moduleById(id) {
  for (var i = 0; i < state.modules.length; i++) {
    if (state.modules[i].module_id === id) return state.modules[i];
  }
  return null;
}

function activeModule() {
  return state.activeId ? moduleById(state.activeId) : null;
}

function visibleModules() {
  return state.modules.filter(function(m) {
    return state.filter === "all" || (state.filter === "fail" ? m.status === "FAIL" : m.status !== "FAIL");
  });
}

function summaryOr(value, fallback) {
  return value === undefined || value === null ? fallback : value;
}

function computeCellTotals() {
  var total = 0, def = 0;
  state.modules.forEach(function(m) {
    total += (+m.total_cells || 0);
    def += (+m.defective_cells_count || 0);
  });
  return { total: total, defective: def };
}
