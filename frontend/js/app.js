/**
 * LuminaPV - App Entry Point & Event Controller
 * Registers event listeners for pan/zoom, keyboard shortcuts, drag & drop, dock collapsing, theme management, and boot sequence.
 */

/* ---------- tree focus helper ---------- */
function focusTreeItem(id) {
  var tree = $("#tree");
  if (!tree) return;
  var items = $$(".mod", tree);
  for (var i = 0; i < items.length; i++) {
    var on = items[i].dataset.id === id;
    items[i].tabIndex = on ? 0 : -1;
    if (on) items[i].focus();
  }
}

/* ---------- dock collapse ---------- */
var dockCollapsed = false;

function toggleDockCollapse() {
  dockCollapsed = !dockCollapsed;
  document.body.classList.toggle("dock-collapsed", dockCollapsed);
  var b = $("#dockCollapse");
  if (b) {
    b.setAttribute("aria-expanded", dockCollapsed ? "false" : "true");
    b.setAttribute("aria-label", dockCollapsed ? "Expandir doca de diagnóstico" : "Recolher doca de diagnóstico");
    b.setAttribute("title", dockCollapsed ? "Expandir doca" : "Recolher doca");
  }
  var dcv = $("#dockChevron");
  if (dcv) dcv.setAttribute("d", dockCollapsed ? "M6 15l6-6 6 6" : "M6 9l6 6 6-6");
  setTimeout(function() {
    fitPlate();
    applyTransform();
    drawSpark();
    if (state.distAnimated) {
      state.distValues = distData();
      drawDist(1);
    }
  }, 0);
  log("info", "Doca de diagnóstico " + (dockCollapsed ? "recolhida" : "expandida"));
}

/* ---------- panel zoom ---------- */
function toggleZoom(sel) {
  var el = $(sel);
  if (!el) return;
  var backdrop = $(".backdrop");
  if (el.classList.contains("zoomed")) {
    el.classList.remove("zoomed");
    if (backdrop) backdrop.remove();
    return;
  }
  if (backdrop) backdrop.remove();
  var b = document.createElement("div");
  b.className = "backdrop";
  b.addEventListener("click", function() {
    el.classList.remove("zoomed");
    b.remove();
  });
  document.body.appendChild(b);
  el.classList.add("zoomed");
  live("Painel expandido. Pressione Escape para restaurar.");
}

/* ---------- theme ---------- */
function applyTheme(light) {
  document.body.classList.toggle("theme-light", !!light);
  var tb = $("#themeBtn");
  if (tb) tb.setAttribute("aria-pressed", light ? "true" : "false");
  try { localStorage.setItem("luminapv.theme", light ? "light" : "dark"); } catch (e) {}
  renderViewport();
  renderInspector();
  drawSpark();
  if (state.distAnimated) {
    state.distValues = distData();
    drawDist(1);
  }
}

/* ---------- event registrations ---------- */
function initEvents() {
  var ingestBtn = $("#ingestBtn");
  if (ingestBtn) ingestBtn.addEventListener("click", function() { $("#fileInput").click(); });

  var emptyIngest = $("#emptyIngest");
  if (emptyIngest) emptyIngest.addEventListener("click", function() { $("#fileInput").click(); });

  var drop = $("#drop");
  if (drop) {
    drop.addEventListener("click", function() { $("#fileInput").click(); });
    drop.addEventListener("keydown", function(e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        $("#fileInput").click();
      }
    });
  }

  var fileInput = $("#fileInput");
  if (fileInput) {
    fileInput.addEventListener("change", function(e) {
      ingest(e.target.files);
      e.target.value = "";
    });
  }

  var insZoom = $("#insZoom");
  if (insZoom) {
    insZoom.addEventListener("click", function() { toggleZoom("#inspector"); });
  }

  ["dragenter", "dragover"].forEach(function(ev) {
    document.addEventListener(ev, function(e) {
      e.preventDefault();
      var d = $("#drop");
      if (d) d.classList.add("drag");
    });
  });

  ["dragleave", "drop"].forEach(function(ev) {
    document.addEventListener(ev, function(e) {
      e.preventDefault();
      var d = $("#drop");
      if (d) d.classList.remove("drag");
    });
  });

  document.addEventListener("drop", function(e) {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
      ingest(e.dataTransfer.files);
    }
  });

  $$(".seg button").forEach(function(b) {
    b.addEventListener("click", function() {
      state.filter = b.dataset.filter;
      $$(".seg button").forEach(function(x) {
        x.setAttribute("aria-pressed", x === b ? "true" : "false");
      });
      renderTree();
      log("info", "Filtro do explorador: " + state.filter.toUpperCase());
    });
  });

  var tree = $("#tree");
  if (tree) {
    tree.addEventListener("click", function(e) {
      var li = e.target.closest(".mod");
      if (li) selectModule(li.dataset.id);
    });

    tree.addEventListener("keydown", function(e) {
      var li = e.target.closest(".mod");
      if (!li) return;
      var items = $$(".mod", tree);
      var i = items.indexOf(li);
      if (e.key === "ArrowDown") {
        e.preventDefault();
        var n = items[Math.min(i + 1, items.length - 1)];
        selectModule(n.dataset.id, { focus: true });
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        var p = items[Math.max(i - 1, 0)];
        selectModule(p.dataset.id, { focus: true });
      } else if (e.key === "Home") {
        e.preventDefault();
        selectModule(items[0].dataset.id, { focus: true });
      } else if (e.key === "End") {
        e.preventDefault();
        selectModule(items[items.length - 1].dataset.id, { focus: true });
      } else if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectModule(li.dataset.id, { focus: true });
      }
    });
  }

  var cellHost = $("#cells");
  if (cellHost) {
    cellHost.addEventListener("mousemove", function(e) {
      var c = e.target.closest(".cell");
      if (!c) return;
      var idx = +c.dataset.i, d = state.cellsByIndex[idx];
      var r = +c.dataset.r, cc = +c.dataset.c;
      if (d) {
        showTip("C[" + pad2(r + 1) + "," + pad2(cc + 1) + "]  #" + pad2(idx) + "\n" + (DEFECT_LABEL[d.defect_type] || d.defect_type) + "  ·  confiança " + pct((+d.confidence || 0) * 100), e.clientX, e.clientY);
      } else {
        showTip("C[" + pad2(r + 1) + "," + pad2(cc + 1) + "]  #" + pad2(idx) + "\ncélula nominal", e.clientX, e.clientY);
      }
    });
    cellHost.addEventListener("mouseleave", hideTip);
    cellHost.addEventListener("click", function(e) {
      var c = e.target.closest(".cell");
      if (!c) return;
      selectCell(+c.dataset.i, +c.dataset.r, +c.dataset.c);
    });
  }

  var cellBody = $("#cellBody");
  if (cellBody) {
    cellBody.addEventListener("click", function(e) {
      var tr = e.target.closest("tr[data-i]");
      if (tr) selectCell(tr.dataset.i, +tr.dataset.r, +tr.dataset.c);
    });
    cellBody.addEventListener("keydown", function(e) {
      var tr = e.target.closest("tr[data-i]");
      if (!tr) return;
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectCell(tr.dataset.i, +tr.dataset.r, +tr.dataset.c);
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        var n = tr.nextElementSibling;
        if (n && n.dataset.i !== undefined) {
          n.focus();
          selectCell(n.dataset.i, +n.dataset.r, +n.dataset.c);
        }
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        var prev = tr.previousElementSibling;
        if (prev && prev.dataset.i !== undefined) {
          prev.focus();
          selectCell(prev.dataset.i, +prev.dataset.r, +prev.dataset.c);
        }
      }
    });
  }

  var stage = $("#stage");
  if (stage) {
    stage.addEventListener("wheel", function(e) {
      var pl = $("#plate");
      if (!pl || pl.hidden) return;
      e.preventDefault();
      var rect = stage.getBoundingClientRect();
      var cx = e.clientX - rect.left - rect.width / 2, cy = e.clientY - rect.top - rect.height / 2;
      var z = state.zoom, factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      var next = Math.max(0.25, Math.min(8, z.scale * factor));
      var k = next / z.scale;
      z.tx = cx - (cx - z.tx) * k;
      z.ty = cy - (cy - z.ty) * k;
      z.scale = next;
      applyTransform();
      updateZoomChip();
    }, { passive: false });

    var drag = null;
    stage.addEventListener("pointerdown", function(e) {
      var pl = $("#plate");
      if (!pl || pl.hidden) return;
      if (e.target.closest(".viewtools")) return;
      drag = { x: e.clientX, y: e.clientY, tx: state.zoom.tx, ty: state.zoom.ty };
      pl.classList.add("grabbing");
      if (stage.setPointerCapture) stage.setPointerCapture(e.pointerId);
    });

    stage.addEventListener("pointermove", function(e) {
      if (!drag) return;
      state.zoom.tx = drag.tx + (e.clientX - drag.x);
      state.zoom.ty = drag.ty + (e.clientY - drag.y);
      applyTransform();
    });

    stage.addEventListener("pointerup", function() {
      drag = null;
      var pl = $("#plate");
      if (pl) pl.classList.remove("grabbing");
    });

    stage.addEventListener("pointercancel", function() {
      drag = null;
      var pl = $("#plate");
      if (pl) pl.classList.remove("grabbing");
    });
  }

  var z11 = $("#z11");
  if (z11) z11.addEventListener("click", function() { zoom11(); var zf = $("#zFit"); if (zf) zf.setAttribute("aria-pressed", "false"); });

  var zFit = $("#zFit");
  if (zFit) zFit.addEventListener("click", function() { zoomFit(); zFit.setAttribute("aria-pressed", "true"); });

  var zGrid = $("#zGrid");
  if (zGrid) {
    zGrid.addEventListener("click", function() {
      var on = zGrid.getAttribute("aria-pressed") === "true";
      zGrid.setAttribute("aria-pressed", on ? "false" : "true");
      var ch = $("#cells");
      if (ch) ch.hidden = on;
      log("info", "Sobreposição de grade " + (on ? "ocultada" : "exibida"));
    });
  }

  $$(".dock-tabs button").forEach(function(b) {
    b.addEventListener("click", function() { setDockTab(b.dataset.tab); });
    b.addEventListener("keydown", function(e) {
      var tabs = $$(".dock-tabs button");
      var i = tabs.indexOf(b);
      if (e.key === "ArrowRight") {
        e.preventDefault();
        var n = tabs[(i + 1) % tabs.length];
        n.focus();
        setDockTab(n.dataset.tab);
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        var prev = tabs[(i - 1 + tabs.length) % tabs.length];
        prev.focus();
        setDockTab(prev.dataset.tab);
      }
    });
  });

  var copyBtn = $("#copyBtn");
  if (copyBtn) {
    copyBtn.addEventListener("click", function() {
      copyText($("#jsonOut").textContent);
      log("ok", "Payload do módulo ativo copiado para a área de transferência");
    });
  }

  var dockCollapse = $("#dockCollapse");
  if (dockCollapse) dockCollapse.addEventListener("click", toggleDockCollapse);

  var exportBtn = $("#exportBtn");
  if (exportBtn) exportBtn.addEventListener("click", exportCsv);

  var distChart = $("#distChart");
  if (distChart) {
    distChart.addEventListener("mousemove", function(e) {
      if (!distPts.length) return;
      var r = distChart.getBoundingClientRect();
      var x = (e.clientX - r.left) * (distChart.width / (window.devicePixelRatio || 1)) / r.width;
      var y = (e.clientY - r.top) * (distChart.height / (window.devicePixelRatio || 1)) / r.height;
      var hit = null;
      distPts.forEach(function(p) {
        if (x >= p.rect[0] - 4 && x <= p.rect[0] + p.rect[2] + 4 && y >= p.rect[1] - 8 && y <= p.rect[1] + p.rect[3] + 6) {
          hit = p;
        }
      });
      if (hit) {
        showTip(hit.label + " · " + int(hit.value) + " células\n" + pct(hit.share) + " das anomalias do lote", e.clientX, e.clientY);
      } else {
        hideTip();
      }
    });
    distChart.addEventListener("mouseleave", hideTip);
  }

  var spark = $("#spark");
  if (spark) {
    spark.addEventListener("mousemove", function(e) {
      if (!sparkPts.length) return;
      var r = spark.getBoundingClientRect();
      var x = (e.clientX - r.left) * (spark.width / (window.devicePixelRatio || 1)) / r.width;
      var y = (e.clientY - r.top) * (spark.height / (window.devicePixelRatio || 1)) / r.height;
      var best = null, bd = 1e9;
      sparkPts.forEach(function(p) {
        var d = Math.hypot(p.x - x, p.y - y);
        if (d < bd) { bd = d; best = p; }
      });
      if (best && bd < 24) {
        showTip(best.label + " · dano " + best.value, e.clientX, e.clientY);
      } else {
        hideTip();
      }
    });
    spark.addEventListener("mouseleave", hideTip);
  }

  document.addEventListener("keydown", function(e) {
    if (e.key === "Escape") {
      var authBg = $("#authModalBackdrop");
      if (authBg && !authBg.hidden) {
        closeAuthModal();
        return;
      }
      var z = $(".zoomed");
      if (z) {
        z.classList.remove("zoomed");
        var b = $(".backdrop");
        if (b) b.remove();
        hideTip();
      }
      if (state.controller && state.loading) cancel();
    }
  });

  var authOpenBtn = $("#authOpenBtn");
  if (authOpenBtn) {
    authOpenBtn.addEventListener("click", function() {
      if (state.auth && state.auth.authenticated) {
        authLogout();
      } else {
        openAuthModal("login");
      }
    });
  }

  var authCloseBtn = $("#authCloseBtn");
  if (authCloseBtn) authCloseBtn.addEventListener("click", closeAuthModal);

  var authCancelBtn = $("#authCancelBtn");
  if (authCancelBtn) authCancelBtn.addEventListener("click", closeAuthModal);

  var authModalBackdrop = $("#authModalBackdrop");
  if (authModalBackdrop) {
    authModalBackdrop.addEventListener("click", function(e) {
      if (e.target === authModalBackdrop) closeAuthModal();
    });
  }

  var authTabLogin = $("#authTabLogin");
  if (authTabLogin) authTabLogin.addEventListener("click", function() { setAuthMode("login"); });

  var authTabRegister = $("#authTabRegister");
  if (authTabRegister) authTabRegister.addEventListener("click", function() { setAuthMode("register"); });

  var authDemoFillBtn = $("#authDemoFillBtn");
  if (authDemoFillBtn) {
    authDemoFillBtn.addEventListener("click", function() {
      var em = $("#authEmail"), pw = $("#authPassword");
      if (em) em.value = "operador@luminapv.local";
      if (pw) pw.value = "LuminaPV@2026";
      state.auth.error = null;
      renderAuthUI();
    });
  }

  var authForm = $("#authForm");
  if (authForm) {
    authForm.addEventListener("submit", function(e) {
      e.preventDefault();
      var mode = (state.auth && state.auth.mode) || "login";
      var email = ($("#authEmail") && $("#authEmail").value || "").trim();
      var password = ($("#authPassword") && $("#authPassword").value || "");
      var name = ($("#authName") && $("#authName").value || "").trim();
      var confirmPw = ($("#authPasswordConfirm") && $("#authPasswordConfirm").value || "");
      var role = ($("#authRole") && $("#authRole").value || "operator");

      if (!email || email.indexOf("@") < 1) {
        state.auth.error = "Informe um endereço de e-mail válido.";
        renderAuthUI();
        return;
      }
      if (!password || password.length < 8) {
        state.auth.error = "A senha deve possuir pelo menos 8 caracteres.";
        renderAuthUI();
        return;
      }

      if (mode === "register") {
        if (!name || name.length < 2) {
          state.auth.error = "Informe o nome completo do operador (mín. 2 caracteres).";
          renderAuthUI();
          return;
        }
        if (password !== confirmPw) {
          state.auth.error = "As senhas informadas não coincidem.";
          renderAuthUI();
          return;
        }
        authRegister(name, email, password, role).catch(function() {});
      } else {
        authLogin(email, password).catch(function() {});
      }
    });
  }

  var themeBtn = $("#themeBtn");
  if (themeBtn) {
    themeBtn.addEventListener("click", function() {
      applyTheme(!document.body.classList.contains("theme-light"));
    });
  }

  var historyRefreshBtn = $("#historyRefreshBtn");
  if (historyRefreshBtn) {
    historyRefreshBtn.addEventListener("click", function() {
      if (typeof fetchUserHistory === "function") fetchUserHistory();
    });
  }

  var historyGrid = $("#historyGrid");
  if (historyGrid) {
    historyGrid.addEventListener("click", function(e) {
      var loadBtn = e.target.closest("[data-load-module]");
      if (loadBtn) {
        var modId = loadBtn.getAttribute("data-load-module");
        if (modId && typeof loadHistoricalModule === "function") {
          loadHistoricalModule(modId);
        }
        return;
      }
      var delBtn = e.target.closest("[data-delete-module]");
      if (delBtn) {
        var delId = delBtn.getAttribute("data-delete-module");
        if (delId && typeof deleteHistoricalModule === "function") {
          deleteHistoricalModule(delId);
        }
      }
    });
  }

  window.addEventListener("resize", debounce(function() {
    fitPlate();
    applyTransform();
    var pl = $("#plate");
    if (pl && !pl.hidden) {
      state.zoom.tx = 0;
      state.zoom.ty = 0;
    }
    drawSpark();
    if (state.distAnimated) {
      state.distValues = distData();
      drawDist(1);
    }
  }, 140));
}

/* ---------- boot sequence ---------- */
function boot() {
  initEvents();

  var light = false;
  try {
    var t = localStorage.getItem("luminapv.theme");
    light = t ? t === "light" : false;
  } catch (e) {}
  document.body.classList.toggle("theme-light", light);
  var tb = $("#themeBtn");
  if (tb) tb.setAttribute("aria-pressed", light ? "true" : "false");

  var hash = (window.location.hash || "#payload").slice(1);
  setDockTab(["payload", "dist", "topology", "history", "log"].indexOf(hash) >= 0 ? hash : "payload");

  try {
    var savedFilter = localStorage.getItem("luminapv.filter");
    if (savedFilter && ["all", "fail", "pass"].indexOf(savedFilter) >= 0) state.filter = savedFilter;
  } catch (e) {}
  $$(".seg button").forEach(function(x) {
    x.setAttribute("aria-pressed", x.dataset.filter === state.filter ? "true" : "false");
  });

  log("info", "Console iniciado · endpoint " + API_ENDPOINT + " · nenhuma simulação no cliente");
  renderAll();
  if (typeof authVerifySession === "function") authVerifySession();
  var sbH = $("#sbHttp");
  if (sbH) sbH.textContent = "HTTP —";
  setStatusBar("idle");

  // Inicia polling de telemetria e coordenação do cluster em tempo real
  if (typeof startClusterPolling === "function") {
    startClusterPolling();
    document.addEventListener("visibilitychange", function() {
      if (document.hidden) {
        stopClusterPolling();
      } else {
        startClusterPolling();
      }
    });
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
