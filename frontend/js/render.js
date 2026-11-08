/**
 * LuminaPV - DOM Render Pipeline
 * Orchestrates viewport layout, optical grid transformation, inspector panels, tree explorer, and status updates.
 */

/* ---------- render: metrics ---------- */
function renderMetrics() {
  var s = state.summary;
  var has = !state.loading && !!s;
  var mIns = $("#mInspected");
  if (mIns) mIns.textContent = has ? pad2(+summaryOr(s.total_modules, 0)) : "—";
  var mInsSub = $("#mInspectedSub");
  if (mInsSub) mInsSub.textContent = has ? "módulos no lote inspecionado" : (state.loading ? "analisando..." : "aguardando lote");

  var dm = has ? +summaryOr(s.defective_modules_count, 0) : null;
  var tm = has ? (+summaryOr(s.total_modules, 0) || 0) : null;
  var rate = has ? summaryOr(s.batch_failure_percentage, tm ? (dm / tm * 100) : null) : null;
  var fv = $("#mFailure");
  if (fv) {
    fv.textContent = has ? pct(rate) : "—";
    fv.className = "metric-v" + (has ? (rate > 0 ? " is-fail" : " is-pass") : "");
  }
  var mFailSub = $("#mFailureSub");
  if (mFailSub) mFailSub.textContent = has ? (int(dm) + "/" + int(tm) + " módulos reprovados") : (state.loading ? "analisando..." : "aguardando lote");

  var ct = computeCellTotals();
  var mCells = $("#mCells");
  if (mCells) mCells.textContent = has ? int(ct.defective) : "—";
  var mCellsSub = $("#mCellsSub");
  if (mCellsSub) mCellsSub.textContent = has ? (int(ct.defective) + " de " + int(ct.total) + " células (" + pct(ct.total ? ct.defective / ct.total * 100 : 0) + ")") : (state.loading ? "analisando..." : "aguardando lote");

  var mDegr = $("#mDegr");
  if (mDegr) mDegr.textContent = has ? pct(summaryOr(s.average_cell_degradation_percentage, ct.total ? ct.defective / ct.total * 100 : 0)) : "—";
  var mDegrSub = $("#mDegrSub");
  if (mDegrSub) mDegrSub.textContent = has ? "perda média de área por célula" : (state.loading ? "analisando..." : "aguardando lote");

  var exp = $("#exportBtn");
  if (exp) exp.disabled = !has || state.loading;

  var worst = 0;
  state.modules.forEach(function(m) { worst = Math.max(worst, +m.module_damage_percentage || 0); });
  var bar = $("#globalBar");
  if (bar) bar.style.width = has ? Math.min(100, worst) + "%" : "0%";
}

/* ---------- render: explorer ---------- */
function renderTree() {
  var ul = $("#tree");
  if (!ul) return;
  var countEl = $("#treeCount");
  if (countEl) countEl.textContent = state.modules.length ? String(visibleModules().length) + "/" + String(state.modules.length) : "0";

  if (!state.modules.length) {
    ul.innerHTML = '<li class="empty-mini" role="presentation" style="padding:16px 10px">Fila vazia.<br>Nenhum módulo ingerido no lote.</li>';
    return;
  }
  var list = visibleModules();
  if (!list.length) {
    ul.innerHTML = '<li class="empty-mini" role="presentation" style="padding:16px 10px">Nenhum módulo com o filtro "' + esc(state.filter.toUpperCase()) + '".</li>';
    return;
  }
  ul.innerHTML = list.map(function(m) {
    var fail = m.status === "FAIL";
    var tag = '<span class="tag ' + (fail ? "tag-fail" : "tag-pass") + '">' + (fail ? "FALHA" : "APROVADO") + '</span>';
    var nodeTag = m.processed_by_node ? '<span class="tag-node" title="Processado por ' + esc(m.processed_by_node) + '">' + esc(m.processed_by_node) + '</span>' : '';
    var repTag = m.cache_hit ? '<span class="tag-cache" title="Inferência recuperada do cache distribuído">CACHE</span>' : (m.replicated ? '<span class="tag-rep" title="Resultado replicado com consistência Read-Your-Writes">REPLICADO</span>' : '');
    return '<li role="option" aria-selected="' + (m.module_id === state.activeId) + '" data-id="' + esc(m.module_id) + '" class="mod' + (m.module_id === state.activeId ? " is-active" : "") + '" tabindex="' + (m.module_id === state.activeId ? "0" : "-1") + '">' +
      '<span class="mod-top"><span class="mod-name" title="' + esc(m.filename || m.module_id) + '">' + esc(m.filename || m.module_id) + '</span>' + nodeTag + repTag + tag + '</span>' +
      '<span class="mod-meta"><span>' + esc(m.module_id) + '</span><span>' + (+m.defective_cells_count || 0) + '/' + (+m.total_cells || 0) + ' cél.</span><span>dano ' + pct(m.module_damage_percentage) + '</span></span></li>';
  }).join("");

  var act = ul.querySelector('.mod.is-active');
  if (act) act.setAttribute("aria-selected", "true");
}

/* ---------- render: viewport ---------- */
function renderViewport() {
  var m = activeModule();
  var stage = $("#stage");
  var plate = $("#plate");
  if (!stage || !plate) return;

  var vpFile = $("#vpFile");
  if (vpFile) vpFile.textContent = m ? (m.filename || m.module_id) : "—";
  var vpGrid = $("#vpGrid");
  if (vpGrid) vpGrid.textContent = m ? ("GRADE " + gridDims(m).rows + "×" + gridDims(m).cols) : "GRADE —";
  var vpStatus = $("#vpStatus");
  if (vpStatus) vpStatus.innerHTML = m ? (m.status === "FAIL" ? '<span class="tag tag-fail">REPROVADO</span>' : '<span class="tag tag-pass">APROVADO</span>') : "";

  stage.classList.toggle("no-plate", state.loading || !m);
  if (state.loading || !m) {
    plate.hidden = true;
    applyTransform();
    return;
  }
  plate.hidden = false;

  var g = gridDims(m);
  plate.style.setProperty("--cols", g.cols);
  plate.style.setProperty("--rows", g.rows);

  // Determine true aspect ratio:
  // 1. From API metadata (image_width / image_height)
  // 2. Or from loaded natural image dimensions (state.zoom.natW / natH)
  // 3. Or fallback to topological cell ratio (cols / rows)
  var ar;
  if (m.image_width && m.image_height) {
    ar = m.image_width / m.image_height;
  } else if (state.zoom.natW && state.zoom.natH) {
    ar = state.zoom.natW / state.zoom.natH;
  } else {
    ar = g.cols / g.rows;
  }
  plate.style.aspectRatio = String(ar);

  // Build cell lookup by coordinate "r:c"
  var cellMap = {};
  (m.cells || []).forEach(function(c) {
    cellMap[(+c.row) + ":" + (+c.col)] = c;
  });

  var dmap = {};
  defects(m).forEach(function(c) { dmap[(+c.row) + ":" + (+c.col)] = c; });
  state.cellsByIndex = {};

  var html = "";
  var cw = 100 / g.cols, ch = 100 / g.rows;

  // Track column horizontal bounds and row vertical bounds for precise ruler alignment
  var colBounds = [];
  var rowBounds = [];

  for (var cIdx = 0; cIdx < g.cols; cIdx++) {
    var cLeft, cWidth;
    if (m.x_bounds && m.image_width && m.x_bounds.length > cIdx + 1) {
      cLeft = (m.x_bounds[cIdx] / m.image_width) * 100;
      cWidth = ((m.x_bounds[cIdx + 1] - m.x_bounds[cIdx]) / m.image_width) * 100;
    } else {
      var sampleCell = cellMap["0:" + cIdx];
      if (sampleCell && sampleCell.bbox) {
        cLeft = sampleCell.bbox.x_pct;
        cWidth = sampleCell.bbox.width_pct;
      } else {
        cLeft = cIdx * cw;
        cWidth = cw;
      }
    }
    colBounds.push({ left: cLeft, width: cWidth });
  }

  for (var rIdx = 0; rIdx < g.rows; rIdx++) {
    var rTop, rHeight;
    if (m.y_bounds && m.image_height && m.y_bounds.length > rIdx + 1) {
      rTop = (m.y_bounds[rIdx] / m.image_height) * 100;
      rHeight = ((m.y_bounds[rIdx + 1] - m.y_bounds[rIdx]) / m.image_height) * 100;
    } else {
      var sampleRowCell = cellMap[rIdx + ":0"];
      if (sampleRowCell && sampleRowCell.bbox) {
        rTop = sampleRowCell.bbox.y_pct;
        rHeight = sampleRowCell.bbox.height_pct;
      } else {
        rTop = rIdx * ch;
        rHeight = ch;
      }
    }
    rowBounds.push({ top: rTop, height: rHeight });
  }

  // Generate accurately aligned ruler spans
  var colsHtml = colBounds.map(function(cb, i) {
    return '<span style="left:' + cb.left.toFixed(3) + '%;width:' + cb.width.toFixed(3) + '%">' + pad2(i + 1) + '</span>';
  }).join("");

  var rowsHtml = rowBounds.map(function(rb, r) {
    return '<span style="top:' + rb.top.toFixed(3) + '%;height:' + rb.height.toFixed(3) + '%">' + pad2(r + 1) + '</span>';
  }).join("");

  var rc = $("#rulerCols"), rr = $("#rulerRows");
  if (rc) rc.innerHTML = colsHtml;
  if (rr) rr.innerHTML = rowsHtml;

  // Generate cell overlays mapped with bounding box coordinates
  for (var rowIdx = 0; rowIdx < g.rows; rowIdx++) {
    for (var colIdx = 0; colIdx < g.cols; colIdx++) {
      var cellData = cellMap[rowIdx + ":" + colIdx];
      var d = (cellData && cellData.is_defective) ? cellData : dmap[rowIdx + ":" + colIdx];
      var idx = (cellData && cellData.index !== undefined) ? +cellData.index : (rowIdx * g.cols + colIdx);
      var cls = "cell";
      if (d) {
        cls += (d.defect_type === "dead" ? " is-dead" : " is-defect");
        state.cellsByIndex[idx] = d;
      }

      var leftPct, topPct, widthPct, heightPct;
      if (cellData && cellData.bbox) {
        leftPct = cellData.bbox.x_pct;
        topPct = cellData.bbox.y_pct;
        widthPct = cellData.bbox.width_pct;
        heightPct = cellData.bbox.height_pct;
      } else if (colBounds[colIdx] && rowBounds[rowIdx]) {
        leftPct = colBounds[colIdx].left;
        topPct = rowBounds[rowIdx].top;
        widthPct = colBounds[colIdx].width;
        heightPct = rowBounds[rowIdx].height;
      } else {
        leftPct = colIdx * cw;
        topPct = rowIdx * ch;
        widthPct = cw;
        heightPct = ch;
      }

      var style = "left:" + leftPct.toFixed(3) + "%;top:" + topPct.toFixed(3) + "%;width:" + widthPct.toFixed(3) + "%;height:" + heightPct.toFixed(3) + "%";
      var tag = d ? '<span class="cell-tag">C[' + pad2(rowIdx + 1) + ',' + pad2(colIdx + 1) + ']</span>' : "";
      var ti = d ? ("C[" + pad2(rowIdx + 1) + "," + pad2(colIdx + 1) + "] · #" + pad2(idx) + " · " + (DEFECT_LABEL[d.defect_type] || d.defect_type) + " · " + pct((+d.confidence || 0) * 100)) : "";
      html += '<div class="' + cls + '" data-r="' + rowIdx + '" data-c="' + colIdx + '" data-i="' + idx + '" style="' + style + '" title="' + esc(ti) + '">' + tag + '</div>';
    }
  }

  var cellsHost = $("#cells");
  if (cellsHost) {
    cellsHost.innerHTML = html;
    var zGrid = $("#zGrid");
    cellsHost.hidden = zGrid && zGrid.getAttribute("aria-pressed") === "false";
  }

  // Handle raster image display (rectified image from backend or original uploaded file)
  var raster = m.filename ? findFile(m.filename) : null;
  var ga = $(".grid-area", plate);
  var rb = $("#rasterBadge");

  if (m.rectified_image && ga) {
    state.zoom.raster = true;
    if (rb) rb.hidden = true;
    ga.style.backgroundImage = "url('" + m.rectified_image + "')";
    ga.dataset.for = m.module_id;
    if (m.image_width && m.image_height) {
      state.zoom.natW = m.image_width;
      state.zoom.natH = m.image_height;
      plate.style.aspectRatio = String(m.image_width / m.image_height);
    }
  } else if (raster && ga) {
    state.zoom.raster = true;
    if (rb) rb.hidden = true;
    if (ga.dataset.for !== m.filename) {
      if (ga.dataset.url) URL.revokeObjectURL(ga.dataset.url);
      ga.dataset.url = URL.createObjectURL(raster);
      ga.dataset.for = m.filename;
      var url = ga.dataset.url;
      var img = new Image();
      img.onload = function() {
        if (ga.dataset.url !== url) return;
        state.zoom.natW = img.naturalWidth;
        state.zoom.natH = img.naturalHeight;
        plate.style.aspectRatio = String(img.naturalWidth / img.naturalHeight);
        fitPlate();
        updateZoomChip();
      };
      img.src = url;
    }
    ga.style.backgroundImage = "url('" + ga.dataset.url + "')";
  } else if (ga) {
    state.zoom.raster = false;
    if (rb) rb.hidden = false;
    if (ga.dataset.url) {
      URL.revokeObjectURL(ga.dataset.url);
      delete ga.dataset.url;
    }
    delete ga.dataset.for;
    ga.style.backgroundImage = "none";
    state.zoom.natW = 0;
    state.zoom.natH = 0;
  }

  applyCellSelection();
  fitPlate();
  applyTransform();
  updateZoomChip();
}

function findFile(name) {
  if (!name) return null;
  var base = String(name).toLowerCase();
  for (var i = 0; i < state.files.length; i++) {
    var fn = state.files[i].name.toLowerCase();
    if (fn === base || fn.endsWith("/" + base) || fn.endsWith("\\" + base)) return state.files[i];
  }
  return null;
}

function fitPlate() {
  var plate = $("#plate"), stage = $("#stage");
  if (!plate || !stage || plate.hidden) return;
  var colHead = 14, rowHead = 18, pad = 30;
  var availW = Math.max(140, stage.clientWidth - pad * 2) - rowHead;
  var availH = Math.max(100, stage.clientHeight - pad * 2) - colHead;
  var ar = parseFloat(plate.style.aspectRatio) || 1.6;
  var w = availW, h = availW / ar;
  if (h > availH) { h = availH; w = availH * ar; }
  plate.style.width = Math.round(w + rowHead) + "px";
  plate.style.height = Math.round(h + colHead) + "px";
  state.zoom.fitScale = state.zoom.natW ? w / state.zoom.natW : 1;
}

function applyTransform() {
  var plate = $("#plate");
  if (!plate) return;
  var z = state.zoom;
  plate.style.transform = "translate(calc(-50% + " + z.tx + "px), calc(-50% + " + z.ty + "px)) scale(" + z.scale + ")";
  plate.classList.toggle("compact", z.scale < 0.62);
}

function updateZoomChip() {
  var z = state.zoom;
  var pctv = z.raster ? (z.scale / (z.fitScale || 1)) * 100 : z.scale * 100;
  var zv = $("#zoomVal");
  if (zv) zv.textContent = Math.round(pctv) + "%";
}

function setScale(scale) {
  state.zoom.scale = Math.max(0.25, Math.min(8, scale));
  applyTransform();
  updateZoomChip();
}

function zoomFit() {
  state.zoom.scale = 1;
  state.zoom.tx = 0;
  state.zoom.ty = 0;
  applyTransform();
  updateZoomChip();
}

function zoom11() {
  state.zoom.scale = state.zoom.fitScale ? 1 / Math.max(state.zoom.fitScale, 0.001) : 1;
  state.zoom.tx = 0;
  state.zoom.ty = 0;
  applyTransform();
  updateZoomChip();
}

function renderViewportState() {
  var host = $("#vpState");
  if (!host) return;
  var m = activeModule();
  if (state.loading) {
    host.innerHTML = loadingMarkup();
    return;
  }
  if (!m) {
    host.innerHTML = '<div class="statebox">' +
      '<svg class="ico" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="square" aria-hidden="true"><path d="M3 4h18v16H3zM3 9h18M9 9v11"/></svg>' +
      '<h2>Nenhuma captura carregada</h2>' +
      '<p>Ingira capturas de eletroluminescência (.png, .jpg, .tif) para o endpoint <span class="mono">' + API_ENDPOINT + '</span>. A grade de células, a matriz de coordenadas e o payload bruto só são renderizados a partir da resposta real da API — nada é simulado no cliente.</p>' +
      '<div class="acts"><button class="btn" type="button" id="emptyIngest">Ingerir lote de capturas</button><button class="linkbtn" type="button" id="emptySample">carregar payload de exemplo (.json)</button></div>' +
      '<div class="hint">POST multipart/form-data · múltipla seleção · timeout 60 s</div>' +
      '</div>';
    var ei = $("#emptyIngest");
    if (ei) ei.addEventListener("click", function() { $("#fileInput").click(); });
    var es = $("#emptySample");
    if (es) es.addEventListener("click", loadSample);
    return;
  }
  host.innerHTML = "";
}

function loadingMarkup() {
  return '<div class="statebox">' +
    '<svg class="ico" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="square" aria-hidden="true"><path d="M4 12h16M4 6h16M4 18h10"/></svg>' +
    '<h2 id="loadTitle">Analisando lote pela API</h2>' +
    '<p id="loadSub">Enviando ' + state.pendingFiles.length + ' arquivo(s) para <span class="mono">' + API_ENDPOINT + '</span>.</p>' +
    '<div class="progwrap"><div class="prog" id="prog"><i id="progFill"></i></div>' +
    '<div class="progmeta"><span id="progPhase">UPLINK</span><span id="progPct">0%</span><span id="progTime">0,0 s</span></div></div>' +
    '<div class="acts"><button class="btn" type="button" id="cancelBtn">Cancelar</button></div>' +
    '<div class="hint" id="slowNote" hidden>a análise está demorando mais que o esperado — o serviço ainda não respondeu</div>' +
    '</div>';
}

/* ---------- render: error ---------- */
function errorMarkup(err) {
  var retriable = err.retriable !== false;
  var acts = "";
  if (retriable && state.retries < 3) {
    acts += '<button class="btn" type="button" id="retryBtn">Tentar novamente</button>';
  } else if (retriable) {
    acts += '<button class="btn" type="button" id="copyErrBtn">Copiar ID do erro</button>';
  }
  acts += '<button class="btn" type="button" id="pickBtn">Selecionar outros arquivos</button>';
  return '<div class="statebox"><div class="alert" role="alert">' +
    '<svg class="ico" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="square" aria-hidden="true"><path d="M12 8v5m0 3h.01M10.3 4.3 2.6 18a1 1 0 0 0 .9 1.5h17a1 1 0 0 0 .9-1.5L13.7 4.3a1 1 0 0 0-1.7 0Z"/></svg>' +
    '<div><h3>' + esc(err.title) + '</h3><p>' + esc(err.detail) + '</p>' +
    '<div class="acts">' + acts + '</div>' +
    (state.retries >= 3 ? '<div class="err-id">tentativas esgotadas · ' + esc(state.retries) + '/3 · acione o suporte com o ID ' + esc(state.errorId || "—") + '</div>' : '') +
    '<div class="err-id">última tentativa ' + esc(state.lastAttempt || "—") + ' · ID ' + esc(state.errorId || "—") + ' · ' + state.pendingFiles.length + ' arquivo(s) preservados em memória</div>' +
    '</div></div></div>';
}

function renderErrorState(err) {
  var vps = $("#vpState");
  if (vps) vps.innerHTML = errorMarkup(err);
  var plate = $("#plate");
  if (plate) plate.hidden = true;
  var stage = $("#stage");
  if (stage) stage.classList.add("no-plate");

  var rb = $("#retryBtn");
  if (rb) rb.addEventListener("click", retry);
  var pb = $("#pickBtn");
  if (pb) pb.addEventListener("click", function() { $("#fileInput").click(); });
  var cb = $("#copyErrBtn");
  if (cb) cb.addEventListener("click", function() {
    copyText(state.errorId || "");
    log("ok", "ID do erro copiado: " + (state.errorId || "—"));
  });

  if (state.backoffTimer) { clearInterval(state.backoffTimer); state.backoffTimer = null; }
  if (rb && state.retries >= 1) {
    var wait = Math.min(8, Math.pow(2, state.retries));
    rb.disabled = true;
    var left = wait;
    rb.textContent = "Tentar novamente em " + left + "s";
    log("info", "Recuo exponencial antes da próxima tentativa: " + wait + " s");
    state.backoffTimer = setInterval(function() {
      left -= 1;
      if (left <= 0) {
        clearInterval(state.backoffTimer);
        state.backoffTimer = null;
        rb.disabled = false;
        rb.textContent = "Tentar novamente";
      } else {
        rb.textContent = "Tentar novamente em " + left + "s";
      }
    }, 1000);
  }
  setStatusBar("erro");
}

/* ---------- render: inspector ---------- */
function renderInspector() {
  var m = activeModule();
  var insId = $("#insId");
  if (insId) insId.textContent = m ? (m.module_id) : "Nenhum módulo selecionado";

  var sev = m ? severity(m) : null;
  var insStatus = $("#insStatus");
  if (insStatus) {
    insStatus.innerHTML = m ? ((m.status === "FAIL" ? '<span class="tag tag-fail">REPROVADO</span>' : '<span class="tag tag-pass">APROVADO</span>') + ' <span class="tag tag-open">' + sev.label + '</span>') : "";
  }

  var dmg = $("#insDamage");
  if (dmg) {
    if (m) {
      dmg.textContent = pct(m.module_damage_percentage);
      dmg.className = "damage" + (isFinite(+m.module_damage_percentage) && +m.module_damage_percentage < 5 ? " ok" : "");
    } else {
      dmg.textContent = "—";
      dmg.className = "damage";
    }
  }

  var g = m ? gridDims(m) : null;
  var insTally = $("#insTally");
  if (insTally) insTally.innerHTML = m ? (int(+m.defective_cells_count || 0) + '<small> / ' + int(+m.total_cells || 0) + '</small>') : "—";
  var insGrid = $("#insGrid");
  if (insGrid) insGrid.textContent = m ? ("grade " + g.rows + "×" + g.cols + " · " + int(g.rows * g.cols) + " células") : "grade —";

  var dl = m ? defects(m) : [];
  var byType = {};
  dl.forEach(function(c) { byType[c.defect_type] = (byType[c.defect_type] || 0) + 1; });
  var dom = null, dc = -1;
  Object.keys(byType).forEach(function(k) {
    if (byType[k] > dc) { dc = byType[k]; dom = k; }
  });
  var insDom = $("#insDominant");
  if (insDom) insDom.textContent = dom ? (DEFECT_LABEL[dom] || dom) : (m ? "NENHUM" : "—");

  var confs = dl.map(function(c) { return +c.confidence || 0; });
  var insConf = $("#insConf");
  if (insConf) insConf.textContent = confs.length ? pct(confs.reduce(function(a, b) { return a + b; }, 0) / confs.length) : "—";

  var insNode = $("#insNode");
  if (insNode) {
    if (m) {
      var nName = m.processed_by_node || "worker-1";
      var isL = state.cluster && state.cluster.leader_id && (nName.indexOf(state.cluster.leader_id) >= 0 || nName === state.cluster.leader_id);
      insNode.innerHTML = '<span class="mono">' + esc(nName) + '</span>' + (isL ? ' <span class="tag-leader" style="font-size:8.5px">LÍDER</span>' : ' <span class="tag-replica" style="font-size:8.5px">RÉPLICA</span>');
    } else {
      insNode.textContent = "—";
    }
  }

  var insRep = $("#insRep");
  if (insRep) {
    if (m) {
      var rHtml = m.cache_hit ? '<span class="tag-cache">CACHE HIT</span>' : (m.replicated ? '<span class="tag-rep">REPLICADO (RYW)</span>' : '<span class="tag-replica">SÍNCRONO</span>');
      if (m.model_sha256) {
        rHtml += ' <span class="mono" style="font-size:9.5px;color:var(--muted2)" title="SHA-256 do best.pt: ' + esc(m.model_sha256) + '">[' + m.model_sha256.slice(0, 6) + '…]</span>';
      }
      insRep.innerHTML = rHtml;
    } else {
      insRep.textContent = "—";
    }
  }

  var countSpan = $("#cellPanelCount");
  if (countSpan) countSpan.textContent = dl.length ? (dl.length + " anomalias") : "0";

  var body = $("#cellBody");
  if (body) {
    if (!m) {
      body.innerHTML = '<tr><td colspan="5" class="empty-mini">Selecione um módulo no explorador para ver a matriz de coordenadas.</td></tr>';
    } else if (!dl.length) {
      body.innerHTML = '<tr><td colspan="5" class="empty-mini">Nenhuma célula defeituosa detectada neste módulo.<br>' + int(+m.total_cells || 0) + ' células aprovadas na inspeção.</td></tr>';
    } else {
      body.innerHTML = dl.slice().sort(function(a, b) { return (+a.index || 0) - (+b.index || 0); }).map(function(c) {
        var r = +c.row || 0, cc = +c.col || 0;
        return '<tr tabindex="0" data-i="' + esc(c.index) + '" data-r="' + r + '" data-c="' + cc + '">' +
          '<td>#' + pad2(+c.index || 0) + '</td><td>' + pad2(r + 1) + '</td><td>' + pad2(cc + 1) + '</td>' +
          '<td class="c-status">' + esc(DEFECT_LABEL[c.defect_type] || String(c.defect_type || "—").toUpperCase()) + '</td>' +
          '<td class="c-conf">' + pct((+c.confidence || 0) * 100) + '</td></tr>';
      }).join("");
    }
  }
  drawSpark();
}

/* ---------- render: dock ---------- */
function renderJson() {
  var m = activeModule();
  var out = m ? JSON.stringify(m, null, 2) : "{ }";
  var jsonEl = $("#jsonOut");
  if (jsonEl) jsonEl.innerHTML = highlightJson(out);
}

function highlightJson(text) {
  return esc(text).replace(/("(?:\\u[\da-fA-F]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(?:true|false)\b|\bnull\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g, function(match) {
    var cls = "j-num";
    if (match.charAt(0) === '"') cls = /:\s*$/.test(match) ? "j-key" : "j-str";
    else if (match === "true" || match === "false") cls = "j-bool";
    else if (match === "null") cls = "j-null";
    return '<span class="' + cls + '">' + match + '</span>';
  });
}

function setDockTab(name) {
  $$(".dock-tabs button").forEach(function(b) {
    var on = b.dataset.tab === name;
    b.setAttribute("aria-selected", on ? "true" : "false");
  });
  var dp = $("#dockPayload"), dd = $("#dockDist"), dl = $("#dockLog"), dt = $("#dockTopology"), cb = $("#copyBtn");
  if (dp) dp.hidden = name !== "payload";
  if (dd) dd.hidden = name !== "dist";
  if (dl) dl.hidden = name !== "log";
  if (dt) dt.hidden = name !== "topology";
  if (cb) cb.style.visibility = name === "payload" ? "visible" : "hidden";

  try { if (history.replaceState) history.replaceState(null, "", "#" + name); } catch (e) {}
  if (name === "dist") {
    if (!state.distAnimated) {
      state.distValues = distData();
      if (state.distValues.length) animateDist(); else drawDist(1);
      state.distAnimated = true;
    } else {
      drawDist(1);
    }
  } else if (name === "topology") {
    renderTopology();
  }
}

/* ---------- render: cluster topology ---------- */
function renderTopology() {
  var c = state.cluster;
  var lId = $("#topoLeaderId");
  var lTerm = $("#topoLeaderTerm");
  var lBadge = $("#topoLeaderBadge");
  var sName = $("#topoServiceName");
  var dDomain = $("#topoDnsDomain");
  var cModel = $("#topoConsistency");
  var rSync = $("#topoReplicaSync");
  var rCount = $("#topoReplicationCount");
  var mInteg = $("#topoModelIntegrity");
  var nBody = $("#topoNodesBody");
  var aLocks = $("#topoActiveLocks");
  var cTime = $("#topoClusterTime");

  if (!c) {
    if (lId) lId.textContent = "STANDALONE / LOCAL";
    if (lTerm) lTerm.textContent = "Aguardando telemetria do cluster";
    if (nBody) {
      nBody.innerHTML = '<tr>' +
        '<td><strong>worker-1</strong></td>' +
        '<td class="mono">worker-1.luminapv.local</td>' +
        '<td><span class="tag-leader">LÍDER</span></td>' +
        '<td><span class="tag tag-pass">ONLINE</span></td>' +
        '<td class="mono" style="color:var(--muted2)">http://localhost:8001</td>' +
        '<td><span class="mono">best.pt (local)</span></td>' +
        '<td class="mono">ativo</td>' +
        '</tr>';
    }
    return;
  }

  var leaderId = c.leader_id || "worker-1";
  var term = c.term || 1;
  if (lId) lId.textContent = leaderId;
  if (lTerm) lTerm.textContent = "Termo #" + term + " · Concessão Ativa (Redis Lease)";
  if (lBadge) lBadge.textContent = "LEADER";

  var naming = c.naming_scheme || {};
  if (sName) sName.textContent = naming.service_name || "service://luminapv-vision-worker";
  if (dDomain) dDomain.textContent = "Domínio ." + (naming.dns_domain || "luminapv.local") + " · Catálogo Dinâmico";

  var repStats = c.replication_stats || {};
  if (cModel) cModel.textContent = (repStats.consistency_model || c.consistency_model || "Read-Your-Writes").toUpperCase();
  if (rSync) rSync.textContent = "Sincronização Réplicas: " + (repStats.secondary_replica_sync || "Eventual");
  if (rCount) rCount.textContent = int(repStats.total_replicated_modules !== undefined ? repStats.total_replicated_modules : 0);
  if (mInteg) {
    var consistent = repStats.model_consistent !== false;
    mInteg.innerHTML = consistent ? '<span style="color:var(--pass)">Pesos IA: Íntegros (SHA-256)</span>' : '<span style="color:var(--defect-ink)">Divergência de Pesos IA</span>';
  }

  var nodes = Array.isArray(c.nodes) ? c.nodes : [];
  if (nBody) {
    if (!nodes.length) {
      nBody.innerHTML = '<tr><td colspan="7" class="empty-mini">Nenhum nó registrado no catálogo no momento.</td></tr>';
    } else {
      nBody.innerHTML = nodes.map(function(n) {
        var isLeader = n.role === "LEADER" || n.node_id === leaderId;
        var isOnline = (n.status || "ONLINE").toUpperCase() === "ONLINE";
        var roleBadge = isLeader ? '<span class="tag-leader">LÍDER</span>' : '<span class="tag-replica">RÉPLICA</span>';
        var statusBadge = isOnline ? '<span class="tag tag-pass">ONLINE</span>' : '<span class="tag tag-fail">OFFLINE</span>';
        var shaShort = n.model_sha256 ? ('<span title="' + esc(n.model_sha256) + '" class="mono">' + esc(n.model_sha256.slice(0, 10)) + '…</span>') : '<span class="mono" style="color:var(--muted2)">—</span>';
        var lastSeen = n.last_seen ? new Date(n.last_seen * 1000).toLocaleTimeString("pt-BR") : "ativo agora";
        return '<tr>' +
          '<td><strong>' + esc(n.node_id || "—") + '</strong></td>' +
          '<td class="mono">' + esc(n.dns_name || (n.node_id + ".luminapv.local")) + '</td>' +
          '<td>' + roleBadge + '</td>' +
          '<td>' + statusBadge + '</td>' +
          '<td class="mono" style="color:var(--muted2)">' + esc(n.endpoint || "—") + '</td>' +
          '<td>' + shaShort + '</td>' +
          '<td class="mono">' + esc(lastSeen) + '</td>' +
          '</tr>';
      }).join("");
    }
  }

  var locks = Array.isArray(c.active_locks) ? c.active_locks : [];
  if (aLocks) {
    if (!locks.length) {
      aLocks.innerHTML = 'Locks Distribuídos Ativos: <span style="color:var(--pass)">Nenhum (seção crítica livre)</span>';
    } else {
      var lockNames = locks.map(function(lk) { return esc(lk.resource || lk); }).join(", ");
      aLocks.innerHTML = 'Locks Distribuídos Ativos: <span style="color:var(--warn)">' + locks.length + ' em vigor (' + lockNames + ')</span>';
    }
  }

  if (cTime) {
    var ts = c.timestamp ? new Date(c.timestamp).toLocaleTimeString("pt-BR") : now();
    cTime.textContent = "Última sincronização: " + ts;
  }
}

/* ---------- render: status bar ---------- */
function setStatusBar(mode, http) {
  var dot = $("#sbDot");
  var st = $("#sbState");
  if (dot) dot.className = "dot" + (mode === "busy" ? " busy" : (mode === "erro" ? " err" : ""));
  if (st) st.textContent = mode === "busy" ? "ANALISANDO" : (mode === "erro" ? "FALHA" : "PRONTO");
  if (http !== undefined) {
    var sbH = $("#sbHttp");
    if (sbH) sbH.textContent = "HTTP " + (http || "—");
  }
  var m = activeModule();
  var sbA = $("#sbActive");
  if (sbA) sbA.textContent = m ? ("MÓDULO " + m.module_id + (m.status === "FAIL" ? " · REPROVADO" : " · APROVADO")) : "MÓDULO —";
  var ct = computeCellTotals();
  var sbB = $("#sbBatch");
  if (sbB) sbB.textContent = state.modules.length ? ("LOTE " + state.modules.length + " MÓD · " + int(ct.defective) + " CÉL. ANÔMALAS") : "LOTE —";

  var c = state.cluster;
  var sbLeader = $("#sbLeaderBadge");
  if (sbLeader) {
    sbLeader.textContent = c && c.leader_id ? ("LÍDER: " + c.leader_id) : "LÍDER: —";
  }
  var sbNodes = $("#sbClusterNodes");
  if (sbNodes) {
    var count = (c && Array.isArray(c.nodes)) ? c.nodes.length : 0;
    sbNodes.textContent = count ? (count + " NÓ" + (count > 1 ? "S" : "")) : "CLUSTER —";
  }
  var sbCons = $("#sbConsistency");
  if (sbCons) {
    var cm = (c && c.consistency_model) ? (c.consistency_model === "read-your-writes" ? "RYW / EVENTUAL" : c.consistency_model.toUpperCase()) : "RYW";
    sbCons.textContent = "CONSISTÊNCIA: " + cm;
  }
}

/* ---------- master render ---------- */
function renderAll() {
  renderMetrics();
  renderTree();
  renderViewportState();
  renderViewport();
  renderInspector();
  renderJson();
  renderTopology();
  renderAuthUI();
  setStatusBar(state.loading ? "busy" : (state.errorId ? "erro" : "idle"));
  var dd = $("#dockDist");
  if (state.distAnimated && dd && !dd.hidden) {
    state.distValues = distData();
    drawDist(1);
  }
}

/* ---------- render: authentication UI & modal ---------- */
function renderAuthUI() {
  var a = state.auth || {};
  var badge = $("#authUserBadge");
  var btnLabel = $("#authBtnLabel");
  var alertBox = $("#authAlert");
  var submitBtn = $("#authSubmitBtn");
  var nameField = $("#authNameField");
  var confirmField = $("#authConfirmField");
  var roleField = $("#authRoleField");
  var demoRow = $("#authDemoRow");
  var tabLogin = $("#authTabLogin");
  var tabReg = $("#authTabRegister");

  var isReg = a.mode === "register";
  if (tabLogin) tabLogin.setAttribute("aria-selected", isReg ? "false" : "true");
  if (tabReg) tabReg.setAttribute("aria-selected", isReg ? "true" : "false");
  if (nameField) nameField.hidden = !isReg;
  if (confirmField) confirmField.hidden = !isReg;
  if (roleField) roleField.hidden = !isReg;
  if (demoRow) demoRow.hidden = isReg;

  if (badge) {
    if (a.authenticated && a.user) {
      var roleTag = String(a.user.role || "operator").toUpperCase();
      badge.className = "tag-auth-ok";
      badge.textContent = "OPERADOR: " + a.user.name + " [" + roleTag + "]";
      badge.title = "Autenticado como " + a.user.email + " (storage: " + (a.user.storage_backend || "memory") + ")";
    } else {
      badge.className = "tag-replica";
      badge.textContent = "SESSÃO: NÃO AUTENTICADO";
      badge.title = "Clique em Autenticar para iniciar sessão de operador";
    }
  }

  if (btnLabel) {
    btnLabel.textContent = (a.authenticated && a.user) ? "Sair" : "Autenticar";
  }

  if (alertBox) {
    if (a.error) {
      alertBox.hidden = false;
      alertBox.textContent = a.error;
    } else {
      alertBox.hidden = true;
      alertBox.textContent = "";
    }
  }

  if (submitBtn) {
    submitBtn.disabled = !!a.loading;
    if (a.loading) {
      submitBtn.textContent = isReg ? "Cadastrando..." : "Autenticando...";
    } else {
      submitBtn.textContent = isReg ? "Concluir Cadastro" : "Entrar no Console";
    }
  }
}

function setAuthMode(mode) {
  if (!state.auth) return;
  state.auth.mode = mode === "register" ? "register" : "login";
  state.auth.error = null;
  renderAuthUI();
  var targetInput = state.auth.mode === "register" ? $("#authName") : $("#authEmail");
  if (targetInput) setTimeout(function() { targetInput.focus(); }, 20);
}

function openAuthModal(mode, errorMsg) {
  var backdrop = $("#authModalBackdrop");
  if (!backdrop) return;
  state.auth.mode = mode === "register" ? "register" : "login";
  state.auth.error = errorMsg || null;
  backdrop.hidden = false;
  renderAuthUI();
  var firstInput = state.auth.mode === "register" ? $("#authName") : $("#authEmail");
  if (firstInput) setTimeout(function() { firstInput.focus(); }, 30);
}

function closeAuthModal() {
  var backdrop = $("#authModalBackdrop");
  if (!backdrop) return;
  backdrop.hidden = true;
  state.auth.error = null;
  renderAuthUI();
}

/* ---------- selection ---------- */
function selectModule(id, opts) {
  if (!moduleById(id)) return;
  if (state.activeId !== id) {
    state.zoom.scale = 1;
    state.zoom.tx = 0;
    state.zoom.ty = 0;
    var zf = $("#zFit");
    if (zf) zf.setAttribute("aria-pressed", "true");
  }
  state.activeId = id;
  state.selectedCell = null;
  try { localStorage.setItem("luminapv.module", id); } catch (e) {}
  renderTree();
  renderViewport();
  renderInspector();
  renderJson();
  setStatusBar(state.loading ? "busy" : (state.errorId ? "erro" : "idle"));
  if (opts && opts.focus) focusTreeItem(id);
  var m = moduleById(id);
  log("info", "Módulo " + id + " (" + (m.filename || "—") + ") carregado no inspetor · " + defects(m).length + " anomalias");
}

function applyCellSelection() {
  $$(".cell.is-sel").forEach(function(el) { el.classList.remove("is-sel"); });
  if (state.selectedCell === null) return;
  var el = $('#cells .cell[data-i="' + state.selectedCell + '"]');
  if (el) el.classList.add("is-sel");
}

function selectCell(index, row, col) {
  state.selectedCell = index;
  applyCellSelection();
  $$("#cellBody tr").forEach(function(tr) {
    var on = String(tr.dataset.i) === String(index);
    tr.classList.toggle("is-sel", on);
    if (on) {
      var wrap = $("#cellTableWrap");
      if (wrap) {
        var top = tr.offsetTop - wrap.clientHeight / 2 + tr.clientHeight / 2;
        wrap.scrollTop = Math.max(0, top);
      }
    }
  });
}
