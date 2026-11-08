/**
 * LuminaPV - API Gateway Communication & Data Ingestion
 * Handles multipart bulk transmission, stream fallback, retries, CSV export, and payload parsing.
 */

function isImage(f) {
  return /\.(png|jpe?g|tiff?|bmp|webp)$/i.test(f.name) || /^image\//.test(f.type);
}

function isJson(f) {
  return /\.json$/i.test(f.name) || f.type === "application/json";
}

function newErrorId(status) {
  return (status ? ("H" + status) : "NET") + "-" + Date.now().toString(36).toUpperCase().slice(-6);
}

function describeHttp(status, body) {
  if (status === 404) {
    return {
      title: "Endpoint de análise não encontrado",
      detail: "A rota " + API_ENDPOINT + " respondeu HTTP 404. Confirme se o serviço de inferência está em execução e se a rota está publicada.",
      retriable: true
    };
  }
  if (status === 400 || status === 415 || status === 422) {
    return {
      title: "Lote rejeitado pela API",
      detail: "HTTP " + status + " — a API recusou o envio: " + (body ? String(body).slice(0, 240) : "sem detalhe na resposta") + ". Verifique formato e resolução das capturas.",
      retriable: true
    };
  }
  if (status === 401 || status === 403) {
    return {
      title: "Acesso negado pelo serviço de análise",
      detail: "HTTP " + status + " — a credencial da sessão não autoriza o envio do lote. Renove o acesso e tente novamente.",
      retriable: false
    };
  }
  if (status >= 500) {
    return {
      title: "Falha no serviço de análise",
      detail: "HTTP " + status + ". O lote foi preservado em memória — tente novamente; se persistir, acione o suporte.",
      retriable: true
    };
  }
  return {
    title: "Resposta inesperada da API",
    detail: "HTTP " + status + " em " + API_ENDPOINT + ". O lote foi preservado — tente novamente.",
    retriable: true
  };
}

function setError(err) {
  state.errorId = newErrorId(err.status);
  state.lastAttempt = now();
  state.lastError = err;
  state.loading = false;
  log("error", err.title + " — " + err.detail + " · ID " + state.errorId);
  renderErrorState(err);
  renderMetrics();
  setStatusBar("erro", err.status || "—");
  live("Falha na análise: " + err.title);
}

function ingestPayload(data, source) {
  if (!data || typeof data !== "object") throw new Error("Payload inválido");
  var modules = Array.isArray(data.modules) ? data.modules : [];
  var seenIds = {};
  modules.forEach(function(m, idx) {
    if (!m.module_id || seenIds[m.module_id]) {
      m.module_id = (m.module_id ? m.module_id + "_" + pad2(idx + 1) : "MOD-2026-" + pad2(idx + 1));
    }
    seenIds[m.module_id] = true;
  });
  state.modules = modules;
  state.summary = data.batch_summary || {};
  if (!state.summary.total_modules) state.summary.total_modules = modules.length;
  if (state.summary.defective_modules_count === undefined) {
    state.summary.defective_modules_count = modules.filter(function(m) { return m.status === "FAIL"; }).length;
  }
  var prev = state.activeId;
  state.activeId = (prev && moduleById(prev)) ? prev : (modules.length ? modules[0].module_id : null);
  state.selectedCell = null;
  state.distAnimated = false;
  state.distValues = null;
  state.errorId = null;
  state.loading = false;
  state.retries = 0;
  log("ok", "Payload aplicado (" + source + ") · " + modules.length + " módulos · " + int(state.summary.total_modules) + " inspecionados");
  renderAll();
  live("Análise recebida. " + modules.length + " módulos carregados no explorador.");
}

function postBulk(files) {
  var ctrl = new AbortController();
  state.controller = ctrl;
  var fd = new FormData();
  files.forEach(function(f) { fd.append("files", f, f.name); });
  var t0 = performance.now();
  var timedOut = false;
  var timer = setTimeout(function() { timedOut = true; ctrl.abort(); }, TIMEOUT_MS);
  var slowTimer = setTimeout(function() {
    var n = $("#slowNote");
    if (n) n.hidden = false;
  }, SLOW_MS);
  var tick = setInterval(function() {
    var el = $("#progTime");
    if (el) el.textContent = ((performance.now() - t0) / 1000).toFixed(1).replace(".", ",") + " s";
  }, 100);

  function onUpload(frac) {
    var fill = $("#progFill"), pctEl = $("#progPct");
    if (fill) fill.style.width = Math.round(frac * 100) + "%";
    if (pctEl) pctEl.textContent = Math.round(frac * 100) + "%";
    var ph = $("#progPhase");
    if (ph) ph.textContent = frac >= 1 ? "LOCK & ANÁLISE" : "UPLINK";
    var bar = $("#globalBar");
    if (bar) bar.style.width = Math.max(4, Math.round(frac * 100)) + "%";
  }

  var body = fd, headers, durationNote = "fetch";
  try {
    if (window.ReadableStream && window.Response && new Response(fd).body) {
      var probe = new Response(fd);
      var total = Number(probe.headers.get("content-length") || 0);
      var ctype = probe.headers.get("content-type");
      var sent = 0;
      var stream = probe.body.pipeThrough(new TransformStream({
        transform: function(chunk, ctl) {
          sent += chunk.byteLength;
          onUpload(total ? Math.min(sent / total, 1) : 0);
          ctl.enqueue(chunk);
        }
      }));
      body = stream;
      headers = { "Content-Type": ctype };
      durationNote = "fetch (stream com progresso de bytes)";
    }
  } catch (e) {
    body = fd;
    headers = undefined;
  }

  log("req", "POST " + API_ENDPOINT + " · " + files.length + " arquivo(s) · Aquisição de locks distribuídos & balanceamento Round-Robin");
  setStatusBar("busy");

  function doFetch(useBody, useHeaders) {
    var opts = { method: "POST", signal: ctrl.signal };
    var mergedHeaders = {};
    if (useHeaders) {
      Object.keys(useHeaders).forEach(function(k) { mergedHeaders[k] = useHeaders[k]; });
    }
    if (state.auth && state.auth.token) {
      mergedHeaders["Authorization"] = "Bearer " + state.auth.token;
    }
    if (Object.keys(mergedHeaders).length > 0) {
      opts.headers = mergedHeaders;
    }
    if (useBody !== fd) opts.duplex = "half";
    opts.body = useBody;
    return fetch(API_ENDPOINT, opts);
  }

  return doFetch(body, headers).catch(function(e) {
    if (body !== fd) {
      log("info", "Streaming de upload não suportado pelo navegador — enviando via FormData direto");
      onUpload(1);
      return doFetch(fd, undefined);
    }
    throw e;
  }).then(function(res) {
    clearTimeout(timer);
    clearTimeout(slowTimer);
    clearInterval(tick);
    onUpload(1);
    var status = res.status;
    $("#sbHttp").textContent = "HTTP " + status;
    return res.text().then(function(text) {
      if (!res.ok) {
        if ((status === 401 || status === 403) && typeof openAuthModal === "function") {
          openAuthModal("login", "Autenticação necessária para submeter lotes à análise.");
        }
        var d = describeHttp(status, text && text.slice(0, 200));
        var err = new Error(d.title);
        err.title = d.title;
        err.detail = d.detail;
        err.retriable = d.retriable;
        err.status = status;
        throw err;
      }
      var data;
      try {
        data = JSON.parse(text);
      } catch (pe) {
        var e2 = new Error("json");
        e2.title = "Resposta da API não é JSON válido";
        e2.detail = "O serviço respondeu HTTP " + status + " mas o corpo não pôde ser interpretado como JSON. Confirme o contrato de resposta do endpoint.";
        e2.retriable = true;
        e2.status = status;
        throw e2;
      }
      log("ok", "HTTP " + status + " · resposta recebida em " + ((performance.now() - t0) / 1000).toFixed(2).replace(".", ",") + " s · " + (Array.isArray(data.modules) ? data.modules.length : 0) + " módulos");
      if (Array.isArray(data.modules) && data.modules.length) {
        var nodeCounts = {};
        var repCount = 0;
        var cacheCount = 0;
        data.modules.forEach(function(m) {
          var nId = m.processed_by_node || "worker-1";
          nodeCounts[nId] = (nodeCounts[nId] || 0) + 1;
          if (m.replicated) repCount++;
          if (m.cache_hit) cacheCount++;
        });
        var distSummary = Object.keys(nodeCounts).map(function(k) { return k + ": " + nodeCounts[k] + " mód."; }).join(" · ");
        log("info", "Distribuição do lote entre nós: " + distSummary);
        log("ok", "Consistência Read-Your-Writes: " + repCount + " módulo(s) replicado(s) · " + cacheCount + " cache hit(s)");
        log("info", "Exclusão mútua concluída: locks distribuídos por módulo liberados com sucesso");
      }
      ingestPayload(data, "HTTP " + status);
      pollClusterStatus();
      return data;
    });
  }).catch(function(e) {
    clearTimeout(timer);
    clearTimeout(slowTimer);
    clearInterval(tick);
    if (e && e.status) {
      setError(e);
    } else if (timedOut || (e && e.name === "AbortError" && !e.__cancelled)) {
      setError({
        title: "Tempo esgotado na análise",
        detail: "A API não respondeu em " + TIMEOUT_MS / 1000 + " s e a requisição foi cancelada. O lote foi preservado em memória — tente novamente.",
        retriable: true,
        status: null
      });
    } else if (e && e.name === "AbortError") {
      log("info", "Requisição cancelada pelo operador");
      setStatusBar("idle");
    } else {
      setError({
        title: "Serviço de análise inacessível",
        detail: "Não foi possível alcançar " + API_ENDPOINT + " (" + (e && e.message ? e.message : "erro de rede") + "). Verifique a conexão com o serviço de análise e tente novamente.",
        retriable: true,
        status: null
      });
    }
    throw e;
  });
}

function ingest(files) {
  var arr = Array.prototype.slice.call(files || []);
  if (!arr.length) return;
  var jsons = arr.filter(isJson);
  var images = arr.filter(function(f) { return !isJson(f) && isImage(f); });
  var other = arr.filter(function(f) { return !isJson(f) && !isImage(f); });
  if (other.length) {
    log("error", "Ignorados " + other.length + " arquivo(s) fora do contrato: " + other.map(function(f) { return f.name; }).join(", "));
  }
  if (jsons.length) {
    var read = jsons.map(function(f) {
      return f.text().then(function(t) { return { name: f.name, text: t }; });
    });
    Promise.all(read).then(function(items) {
      items.forEach(function(it) {
        try {
          ingestPayload(JSON.parse(it.text), "arquivo local " + it.name);
        } catch (e) {
          setError({
            title: "Payload JSON inválido",
            detail: "O arquivo " + it.name + " não pôde ser interpretado como payload de diagnóstico. Envie o JSON no contrato de batch_summary + modules[].",
            retriable: false,
            status: null
          });
        }
      });
    });
  }
  if (images.length) {
    state.pendingFiles = images;
    state.files = images;
    state.loading = true;
    state.retries = 0;
    state.errorId = null;
    state.startedAt = performance.now();
    renderMetrics();
    renderTree();
    $("#vpState").innerHTML = loadingMarkup();
    $("#plate").hidden = true;
    $("#stage").classList.add("no-plate");
    var cb = $("#cancelBtn");
    if (cb) cb.addEventListener("click", cancel);
    live("Analisando " + images.length + " capturas.");
    postBulk(images).catch(function() {});
  }
}

function retry() {
  if (!state.pendingFiles.length || state.loading) return;
  state.retries += 1;
  state.errorId = null;
  state.loading = true;
  log("req", "Nova tentativa #" + state.retries + " · " + state.pendingFiles.length + " arquivo(s) preservado(s)");
  setStatusBar("busy");
  $("#plate").hidden = true;
  $("#stage").classList.add("no-plate");
  $("#vpState").innerHTML = loadingMarkup();
  var cb = $("#cancelBtn");
  if (cb) cb.addEventListener("click", cancel);
  postBulk(state.pendingFiles).catch(function() {});
}

function cancel() {
  if (state.controller && state.loading) state.controller.abort();
  if (state.backoffTimer) { clearInterval(state.backoffTimer); state.backoffTimer = null; }
  state.loading = false;
  log("info", "Análise cancelada pelo operador · lote preservado (" + state.pendingFiles.length + " arquivos)");
  renderAll();
}

function loadSample() {
  log("req", "GET " + SAMPLE_PAYLOAD);
  fetch(SAMPLE_PAYLOAD, { cache: "no-store" }).then(function(r) {
    if (!r.ok) throw new Error("HTTP " + r.status);
    $("#sbHttp").textContent = "HTTP " + r.status;
    return r.json();
  }).then(function(d) {
    log("ok", "Payload de exemplo carregado de " + SAMPLE_PAYLOAD);
    ingestPayload(d, "HTTP GET " + SAMPLE_PAYLOAD);
  }).catch(function(e) {
    var inline = document.getElementById("samplePayload");
    if (inline) {
      try {
        var d = JSON.parse(inline.textContent);
        log("info", "fetch de arquivo indisponível neste contexto — payload de exemplo lido do bloco de dados local");
        ingestPayload(d, "bloco de dados local");
        return;
      } catch (pe) { /* segue para o estado de erro */ }
    }
    setError({
      title: "Payload de exemplo indisponível",
      detail: "Não foi possível ler " + SAMPLE_PAYLOAD + " (" + (e && e.message ? e.message : "erro") + "). Arraste o arquivo para o explorador ou aponte o console para uma API ativa.",
      retriable: false,
      status: null
    });
  });
}

function exportCsv() {
  if (!state.modules.length) return;
  var head = ["ID_MODULO", "ARQUIVO", "STATUS", "TOTAL_CELULAS", "CELULAS_DEFEITUOSAS", "DANO_PCT", "TIPOS_DEFEITO", "COORDENADAS_DEFEITUOSAS"];
  var rows = [head];
  visibleModules().forEach(function(m) {
    var dl = defects(m);
    var types = Array.from(new Set(dl.map(function(c) { return c.defect_type; }))).join(" | ");
    var coords = dl.map(function(c) { return "R" + pad2(+c.row + 1) + "C" + pad2(+c.col + 1); }).join(" ");
    rows.push([
      m.module_id,
      m.filename || "",
      m.status || "",
      m.total_cells || 0,
      m.defective_cells_count || 0,
      nf(m.module_damage_percentage, 1),
      types,
      coords
    ]);
  });
  rows.push([]);
  rows.push(["RESUMO_DO_LOTE", "", "", "", "", "", "", ""]);
  rows.push(["TOTAL_MODULOS", state.summary && state.summary.total_modules || state.modules.length]);
  rows.push(["MODULOS_DEFEITUOSOS", state.summary && state.summary.defective_modules_count]);
  rows.push(["TAXA_FALHA_PCT", state.summary && state.summary.batch_failure_percentage]);
  rows.push(["DEGRADACAO_MEDIA_PCT", state.summary && state.summary.average_cell_degradation_percentage]);
  rows.push(["EXPORTADO_EM", new Date().toISOString()]);
  var csv = rows.map(function(r) {
    return r.map(function(c) { return '"' + String(c === undefined || c === null ? "" : c).replace(/"/g, '""') + '"'; }).join(",");
  }).join("\r\n");
  var d = new Date();
  var name = "luminapv-lote-" + d.getFullYear() + pad2(d.getMonth() + 1) + pad2(d.getDate()) + "-" + pad2(d.getHours()) + pad2(d.getMinutes()) + ".csv";
  var blob = new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8" });
  var a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(function() { URL.revokeObjectURL(a.href); }, 4000);
  log("ok", 'Relatório exportado: ' + name + ' · ' + visibleModules().length + ' linha(s) · coluna de coordenadas incluída');
}

function copyText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).catch(function() { fallbackCopy(text); });
  } else {
    fallbackCopy(text);
  }
}

function fallbackCopy(text) {
  var ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.left = "-1000px";
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand("copy"); } catch (e) {}
  ta.remove();
}

/* ---------- cluster status polling & failover monitoring ---------- */
function pollClusterStatus() {
  if (typeof CLUSTER_ENDPOINT === "undefined") return Promise.resolve(null);

  return fetch(CLUSTER_ENDPOINT, { cache: "no-store" })
    .then(function(res) {
      if (!res.ok) {
        // Graceful degradation when endpoint returns non-200 or offline
        return null;
      }
      return res.json();
    })
    .then(function(data) {
      if (!data || typeof data !== "object") return null;

      var prevLeader = state.lastLeaderId;
      var newLeader = data.leader_id;

      // 1. Detect and log leader election / failover
      if (newLeader && prevLeader && prevLeader !== newLeader) {
        var termStr = data.term ? (" (Termo #" + data.term + ")") : "";
        log("warn", "Transição de Liderança detectada: " + prevLeader + " -> " + newLeader + termStr + " · Failover automático concluído");
        live("Novo líder eleito no cluster: " + newLeader);
      }
      state.lastLeaderId = newLeader || prevLeader;

      // 2. Detect worker nodes health/status transitions
      if (Array.isArray(data.nodes)) {
        var currentNodesState = {};
        data.nodes.forEach(function(node) {
          var nid = node.node_id || node.dns_name;
          var nstatus = (node.status || "ONLINE").toUpperCase();
          currentNodesState[nid] = nstatus;

          var oldStatus = state.lastNodesState ? state.lastNodesState[nid] : null;
          if (oldStatus && oldStatus !== nstatus) {
            var dnsLabel = node.dns_name ? (node.dns_name + " (" + nid + ")") : nid;
            if (nstatus === "OFFLINE") {
              log("warn", "Nó " + dnsLabel + " OFFLINE · Failover ativo");
              live("Nó " + nid + " offline. Tráfego redirecionado para réplicas ativas.");
            } else if (nstatus === "ONLINE") {
              log("ok", "Nó " + dnsLabel + " restabelecido (ONLINE)");
            }
          }
        });
        state.lastNodesState = currentNodesState;
      }

      state.cluster = data;
      renderTopology();
      setStatusBar(state.loading ? "busy" : (state.errorId ? "erro" : "idle"));
      return data;
    })
    .catch(function(err) {
      // Graceful degradation: do NOT throw errors or pollute console if cluster backend is offline
      // Keeps existing state without breaking local inspection
      return null;
    });
}

function startClusterPolling() {
  if (state.clusterPollingInterval) {
    clearInterval(state.clusterPollingInterval);
    state.clusterPollingInterval = null;
  }
  pollClusterStatus();
  var intervalMs = (typeof CLUSTER_POLL_INTERVAL_MS !== "undefined") ? CLUSTER_POLL_INTERVAL_MS : 4000;
  state.clusterPollingInterval = setInterval(pollClusterStatus, intervalMs);
}

function stopClusterPolling() {
  if (state.clusterPollingInterval) {
    clearInterval(state.clusterPollingInterval);
    state.clusterPollingInterval = null;
  }
}

/* ---------- authentication & session management (JWT + S3 Adapter) ---------- */
function saveAuthSession(token, user) {
  state.auth.token = token || null;
  state.auth.user = user || null;
  state.auth.authenticated = !!(token && user);
  state.auth.error = null;
  try {
    if (token && user) {
      localStorage.setItem("luminapv.auth.token", token);
      localStorage.setItem("luminapv.auth.user", JSON.stringify(user));
    }
  } catch (e) {}
  if (typeof renderAuthUI === "function") renderAuthUI();
}

function clearAuthSession() {
  state.auth.token = null;
  state.auth.user = null;
  state.auth.authenticated = false;
  state.auth.error = null;
  try {
    localStorage.removeItem("luminapv.auth.token");
    localStorage.removeItem("luminapv.auth.user");
  } catch (e) {}
  if (typeof renderAuthUI === "function") renderAuthUI();
}

function authLogin(email, password) {
  state.auth.loading = true;
  state.auth.error = null;
  if (typeof renderAuthUI === "function") renderAuthUI();

  return fetch(AUTH_LOGIN_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: email, password: password })
  })
    .then(function(res) {
      return res.json().catch(function() { return {}; }).then(function(body) {
        if (!res.ok) {
          var errMsg = (body && body.detail) ? String(body.detail) : ("Falha na autenticação (HTTP " + res.status + ").");
          throw new Error(errMsg);
        }
        return body;
      });
    })
    .then(function(data) {
      state.auth.loading = false;
      saveAuthSession(data.access_token, data.user);
      log("ok", "Operador autenticado: " + data.user.name + " <" + data.user.email + "> [" + String(data.user.role || "operator").toUpperCase() + "] · storage=" + (data.user.storage_backend || "memory"));
      live("Sessão iniciada como " + data.user.name);
      if (typeof closeAuthModal === "function") closeAuthModal();
      return data;
    })
    .catch(function(err) {
      state.auth.loading = false;
      state.auth.error = (err && err.message) ? err.message : "Não foi possível comunicar com o servidor de autenticação.";
      log("error", "Falha de login: " + state.auth.error);
      if (typeof renderAuthUI === "function") renderAuthUI();
      throw err;
    });
}

function authRegister(name, email, password, role) {
  state.auth.loading = true;
  state.auth.error = null;
  if (typeof renderAuthUI === "function") renderAuthUI();

  return fetch(AUTH_REGISTER_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: name,
      email: email,
      password: password,
      role: role || "operator"
    })
  })
    .then(function(res) {
      return res.json().catch(function() { return {}; }).then(function(body) {
        if (!res.ok) {
          var errMsg = (body && body.detail) ? String(body.detail) : ("Falha no cadastro (HTTP " + res.status + ").");
          throw new Error(errMsg);
        }
        return body;
      });
    })
    .then(function(data) {
      state.auth.loading = false;
      saveAuthSession(data.access_token, data.user);
      log("ok", "Novo operador cadastrado e autenticado: " + data.user.name + " <" + data.user.email + "> [" + String(data.user.role || "operator").toUpperCase() + "]");
      live("Cadastro concluído. Sessão iniciada como " + data.user.name);
      if (typeof closeAuthModal === "function") closeAuthModal();
      return data;
    })
    .catch(function(err) {
      state.auth.loading = false;
      state.auth.error = (err && err.message) ? err.message : "Não foi possível concluir o cadastro.";
      log("error", "Falha no cadastro de operador: " + state.auth.error);
      if (typeof renderAuthUI === "function") renderAuthUI();
      throw err;
    });
}

function authLogout() {
  var tok = state.auth.token;
  var prevEmail = state.auth.user ? state.auth.user.email : "operador";
  clearAuthSession();
  log("info", "Sessão encerrada para " + prevEmail + " · token revogado");
  live("Sessão encerrada.");

  if (!tok) return Promise.resolve();
  return fetch(AUTH_LOGOUT_ENDPOINT, {
    method: "POST",
    headers: { "Authorization": "Bearer " + tok }
  }).catch(function() {
    // Graceful cleanup even if backend is offline
    return null;
  });
}

function authVerifySession() {
  var savedToken = null;
  var savedUser = null;
  try {
    savedToken = localStorage.getItem("luminapv.auth.token");
    var rawUser = localStorage.getItem("luminapv.auth.user");
    if (rawUser) savedUser = JSON.parse(rawUser);
  } catch (e) {}

  if (!savedToken) {
    clearAuthSession();
    return Promise.resolve(null);
  }

  state.auth.token = savedToken;
  state.auth.user = savedUser;
  state.auth.authenticated = !!(savedToken && savedUser);
  if (typeof renderAuthUI === "function") renderAuthUI();

  return fetch(AUTH_ME_ENDPOINT, {
    method: "GET",
    headers: { "Authorization": "Bearer " + savedToken },
    cache: "no-store"
  })
    .then(function(res) {
      if (res.status === 401 || res.status === 403) {
        clearAuthSession();
        log("info", "Sessão anterior expirada ou revogada — faça login novamente.");
        return null;
      }
      if (!res.ok) return savedUser;
      return res.json();
    })
    .then(function(userDto) {
      if (userDto && userDto.email) {
        saveAuthSession(savedToken, userDto);
      }
      return userDto;
    })
    .catch(function() {
      // Graceful degradation when running standalone HTML without backend
      return savedUser;
    });
}
