// Frontend controller for Local LLM Runner V2 Desktop App (pywebview bridge)

let state = {
  mode: "host",
  hostPeers: [],
  hostRunning: false,
  peerRunning: false,
  generating: false,
  downloadTarget: "host",
  activeAssistantBubble: null,
  activeAssistantRaw: "",
};

const $ = (id) => document.getElementById(id);

function showToast(message, kind = "info") {
  const container = $("toastContainer");
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => el.remove(), 3800);
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdownSimple(text) {
  const parts = text.split(/```([\s\S]*?)```/g);
  return parts
    .map((part, idx) => {
      if (idx % 2 === 1) {
        const lines = part.replace(/^\w*\n/, "");
        return `<pre><code>${escapeHtml(lines)}</code></pre>`;
      }
      return escapeHtml(part)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\n/g, "<br>");
    })
    .join("");
}

function syncGlobalPill() {
  const pill = $("globalStatusPill");
  const text = $("globalStatusText");
  if (state.mode === "host") {
    pill.dataset.state = $("hostStatePill").dataset.state;
    text.textContent = `HOST: ${$("hostStateBadge").textContent}`;
  } else {
    pill.dataset.state = $("peerStatePill").dataset.state;
    text.textContent = `PEER: ${$("peerStateBadge").textContent}`;
  }
}

// Mode switching (Host <-> Peer)
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const mode = btn.dataset.mode;
    state.mode = mode;
    document.querySelectorAll(".mode-btn").forEach((b) => b.classList.toggle("active", b === btn));
    $("hostSection").classList.toggle("active", mode === "host");
    $("peerSection").classList.toggle("active", mode === "peer");
    syncGlobalPill();
  });
});

// Host Subtabs
document.querySelectorAll(".subtab").forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.subtab;
    document.querySelectorAll(".subtab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".subview").forEach((v) => v.classList.toggle("active", v.id === `subview-${target}`));
  });
});

// ------------------------------------------------------------------
// Host Config & Peers
// ------------------------------------------------------------------

function applyHostConfig(cfg) {
  $("hostModel").value = cfg.model || "";
  $("hostListenPort").value = cfg.listen_port ?? 5000;
  $("hostAuthToken").value = cfg.auth_token || "";
  $("hostRouting").value = cfg.routing || "auto";
  $("hostWaitTimeout").value = cfg.peer_wait_timeout ?? 300;
  state.hostPeers = (cfg.peers || []).map((p) => ({ ...p }));
  renderPeersList();

  const g = cfg.generation || {};
  $("genMaxTokens").value = g.max_tokens ?? 512;
  $("genTemp").value = g.temperature ?? 0.5;
  $("genTopP").value = g.top_p ?? 0.9;
  $("genTopK").value = g.top_k ?? 40;
  $("genRepPen").value = g.repetition_penalty ?? 1.1;
  $("genHistory").value = g.history_turns ?? 6;
  $("genSystemPrompt").value = g.system_prompt ?? "";
}

function collectHostConfig() {
  return {
    model: $("hostModel").value.trim(),
    listen_port: Number($("hostListenPort").value || 5000),
    auth_token: $("hostAuthToken").value.trim(),
    routing: $("hostRouting").value,
    peer_wait_timeout: Number($("hostWaitTimeout").value || 300),
    peers: state.hostPeers,
    generation: {
      max_tokens: Number($("genMaxTokens").value || 512),
      temperature: Number($("genTemp").value || 0.5),
      top_p: Number($("genTopP").value || 0.9),
      top_k: Number($("genTopK").value || 40),
      repetition_penalty: Number($("genRepPen").value || 1.1),
      history_turns: Number($("genHistory").value || 6),
      system_prompt: $("genSystemPrompt").value,
    },
  };
}

function renderPeersList() {
  const container = $("hostPeersList");
  container.innerHTML = "";
  if (state.hostPeers.length === 0) {
    container.innerHTML = `<div class="muted" style="font-size:12px;">No peers added yet.</div>`;
    return;
  }
  state.hostPeers.forEach((p, idx) => {
    const item = document.createElement("div");
    item.className = "peer-item";
    const sub = p.mode === "outbound" ? `${p.address}:${p.port}` : "connects inbound to host";
    item.innerHTML = `
      <div>
        <div class="peer-item-title">${escapeHtml(p.name)} <span class="peer-tag">${p.mode}</span></div>
        <div class="peer-item-sub">${escapeHtml(sub)}</div>
      </div>
      <button class="btn-xs btn-danger" data-idx="${idx}">Remove</button>
    `;
    item.querySelector("button").addEventListener("click", () => {
      state.hostPeers.splice(idx, 1);
      renderPeersList();
    });
    container.appendChild(item);
  });
}

// ------------------------------------------------------------------
// Peer Config
// ------------------------------------------------------------------

function setPeerConnMode(mode) {
  $("peerConnWanBtn").classList.toggle("active", mode === "wan");
  $("peerConnLanBtn").classList.toggle("active", mode === "lan");
  $("peerWanFields").classList.toggle("hidden", mode !== "wan");
  $("peerLanFields").classList.toggle("hidden", mode === "wan");
}

$("peerConnWanBtn").addEventListener("click", () => setPeerConnMode("wan"));
$("peerConnLanBtn").addEventListener("click", () => setPeerConnMode("lan"));

function applyPeerConfig(cfg) {
  $("peerName").value = cfg.name || "peer-1";
  $("peerModel").value = cfg.model || "";
  $("peerDevice").value = cfg.device || "auto";
  $("peerLayerStart").value = cfg.layer_start ?? "";
  $("peerLayerEnd").value = cfg.layer_end ?? "";
  $("peerListenPort").value = cfg.listen_port ?? 5001;
  $("peerHostAddr").value = cfg.host_address || "";
  $("peerHostPort").value = cfg.host_port ?? 443;
  $("peerTransport").value = cfg.host_transport || "wss";
  $("peerSocks5").value = cfg.socks5_proxy || "";
  $("peerAuthToken").value = cfg.auth_token || "";
  setPeerConnMode(cfg.conn_mode === "lan" ? "lan" : "wan");
}

function collectPeerConfig() {
  const isWan = $("peerConnWanBtn").classList.contains("active");
  return {
    name: $("peerName").value.trim(),
    model: $("peerModel").value.trim(),
    device: $("peerDevice").value,
    layer_start: $("peerLayerStart").value.trim(),
    layer_end: $("peerLayerEnd").value.trim(),
    listen_port: Number($("peerListenPort").value || 5001),
    conn_mode: isWan ? "wan" : "lan",
    host_address: $("peerHostAddr").value.trim(),
    host_port: Number($("peerHostPort").value || 443),
    host_transport: $("peerTransport").value,
    socks5_proxy: $("peerSocks5").value.trim(),
    auth_token: $("peerAuthToken").value.trim(),
  };
}

// ------------------------------------------------------------------
// Backend Event Handler
// ------------------------------------------------------------------

window.__onBackendEvent = function ({ event, data }) {
  if (event === "host_status") {
    $("hostStatePill").dataset.state = data.state;
    $("hostStateBadge").textContent = data.state;
    $("hostStateDetail").textContent = data.detail || "";
    const running = data.state === "STARTING" || data.state === "READY";
    state.hostRunning = running;
    $("startHostBtn").classList.toggle("hidden", running);
    $("stopHostBtn").classList.toggle("hidden", !running);
    $("sendPromptBtn").disabled = data.state !== "READY";
    if (data.state === "ERROR") showToast(data.detail, "error");
    syncGlobalPill();
  } else if (event === "pipeline_update") {
    renderPipelineTopology(data);
  } else if (event === "probe_results") {
    renderProbeResults(data.rows || []);
    document.querySelector('.subtab[data-subtab="pipeline"]').click();
  } else if (event === "chat_chunk") {
    if (state.activeAssistantBubble) {
      state.activeAssistantRaw += data.text;
      state.activeAssistantBubble.innerHTML = renderMarkdownSimple(state.activeAssistantRaw);
      const scroller = $("chatMessages");
      scroller.scrollTop = scroller.scrollHeight;
    }
  } else if (event === "chat_done") {
    if (state.activeAssistantBubble) {
      state.activeAssistantBubble.classList.remove("streaming-cursor");
      const meta = document.createElement("div");
      meta.className = "msg-meta";
      const peerStr = data.peer_ms !== null ? ` • peer ${data.peer_ms}ms` : "";
      meta.textContent = `${data.steps} steps in ${data.elapsed_s}s • ${data.tok_s} tok/s • TTFT ${data.ttft_s}s • host ${data.host_ms}ms${peerStr} • tunnel ${data.tunnel_ms}ms • ${data.rx_mode} ${data.rx_bytes}B`;
      state.activeAssistantBubble.parentElement.appendChild(meta);
    }
    finishGenerationUI();
  } else if (event === "chat_error") {
    if (state.activeAssistantBubble) {
      state.activeAssistantBubble.classList.remove("streaming-cursor");
      state.activeAssistantBubble.innerHTML += `<br><span style="color:#dc2626">[Error: ${escapeHtml(data.error)}]</span>`;
    }
    showToast(data.error, "error");
    finishGenerationUI();
  } else if (event === "peer_status") {
    $("peerStatePill").dataset.state = data.state;
    $("peerStateBadge").textContent = data.state;
    $("peerStateDetail").textContent = data.detail || "";
    const active = !["IDLE", "STOPPED", "ERROR"].includes(data.state);
    state.peerRunning = active;
    $("startPeerBtn").classList.toggle("hidden", active);
    $("stopPeerBtn").classList.toggle("hidden", !active);
    if (["LOADING_SHARD", "READY", "STOPPED"].includes(data.state)) {
      $("peerShardRange").textContent = data.state === "STOPPED" ? "Waiting" : data.detail;
      $("peerShardSub").textContent = data.state;
    }
    appendLog($("peerLogConsole"), `[${data.state}] ${data.detail}`);
    syncGlobalPill();
  } else if (event === "log_line") {
    appendLog($("hostLogConsole"), data.line);
  } else if (event === "download_state") {
    if (data.status === "downloading") {
      $("dlProgressRow").classList.remove("hidden");
      $("confirmDlBtn").disabled = true;
      $("dlStatusText").textContent = `Downloading ${data.repo_id} into models/...`;
    } else if (data.status === "done") {
      $("dlProgressRow").classList.add("hidden");
      $("confirmDlBtn").disabled = false;
      $("downloadModal").classList.add("hidden");
      if (data.target_role === "peer") $("peerModel").value = data.path;
      else $("hostModel").value = data.path;
      showToast(`Model saved to ${data.path}`, "success");
    } else if (data.status === "error") {
      $("dlProgressRow").classList.add("hidden");
      $("confirmDlBtn").disabled = false;
      showToast(`Download failed: ${data.error}`, "error");
    }
  }
};

function appendLog(el, line) {
  el.textContent += line + "\n";
  el.scrollTop = el.scrollHeight;
}

function renderPipelineTopology(pipeline) {
  $("pipelineSummaryMeta").textContent = `— ${pipeline.total_layers} layers • ${pipeline.routing}`;
  const flow = $("topologyFlow");
  flow.innerHTML = "";
  (pipeline.nodes || []).forEach((n, idx) => {
    if (idx > 0) {
      const arrow = document.createElement("div");
      arrow.className = "flow-arrow";
      arrow.textContent = "→";
      flow.appendChild(arrow);
    }
    const card = document.createElement("div");
    card.className = `node-card ${n.is_host ? "host-node" : ""}`;
    card.innerHTML = `
      <div class="node-top">
        <span class="node-name">${escapeHtml(n.name)}</span>
        <span class="layer-pill">Layers ${escapeHtml(n.layers)}</span>
      </div>
      <div class="node-row"><span>Role</span><span>${escapeHtml(n.role)}</span></div>
      <div class="node-row"><span>Device</span><span>${escapeHtml(n.device)} (${escapeHtml(n.free_mem)})</span></div>
      <div class="node-row"><span>Link / RTT</span><span>${escapeHtml(n.connection)} • ${escapeHtml(n.rtt)}</span></div>
    `;
    flow.appendChild(card);
  });
}

function renderProbeResults(rows) {
  const grid = $("probeResults");
  grid.innerHTML = "";
  rows.forEach((r) => {
    const card = document.createElement("div");
    card.className = "node-card";
    const badgeColor = r.reachable ? "#15803d" : "#dc2626";
    card.innerHTML = `
      <div class="node-top">
        <span class="node-name">${escapeHtml(r.name)}</span>
        <span style="color:${badgeColor}; font-weight:700; font-size:11.5px;">${r.reachable ? "REACHABLE" : "OFFLINE"}</span>
      </div>
      <div class="node-row"><span>Endpoint</span><span>${escapeHtml(r.address)}</span></div>
      <div class="node-row"><span>Device / Mem</span><span>${escapeHtml(r.device)} (${escapeHtml(r.free_mem)})</span></div>
      <div class="node-row"><span>Detail</span><span>${escapeHtml(r.detail)}</span></div>
    `;
    grid.appendChild(card);
  });
}

// ------------------------------------------------------------------
// Chat Actions
// ------------------------------------------------------------------

function finishGenerationUI() {
  state.generating = false;
  state.activeAssistantBubble = null;
  state.activeAssistantRaw = "";
  $("stopGenBtn").classList.add("hidden");
  $("sendPromptBtn").disabled = !state.hostRunning;
}

async function sendPrompt(textOverride = null) {
  const input = $("promptInput");
  const prompt = (textOverride ?? input.value).trim();
  if (!prompt || state.generating) return;

  if (!state.hostRunning) {
    showToast("Start the Host pipeline first.", "error");
    return;
  }

  if (textOverride === null) input.value = "";
  $("chatEmptyState")?.classList.add("hidden");

  const scroller = $("chatMessages");
  const userMsg = document.createElement("div");
  userMsg.className = "msg user";
  userMsg.innerHTML = `<div class="msg-role">You</div><div class="msg-bubble">${escapeHtml(prompt).replace(/\n/g, "<br>")}</div>`;
  scroller.appendChild(userMsg);

  const asstMsg = document.createElement("div");
  asstMsg.className = "msg assistant";
  asstMsg.innerHTML = `<div class="msg-role">Assistant</div><div class="msg-bubble streaming-cursor"></div>`;
  scroller.appendChild(asstMsg);

  state.activeAssistantBubble = asstMsg.querySelector(".msg-bubble");
  state.activeAssistantRaw = "";
  state.generating = true;
  $("sendPromptBtn").disabled = true;
  $("stopGenBtn").classList.remove("hidden");
  scroller.scrollTop = scroller.scrollHeight;

  const res = await window.pywebview.api.send_prompt(prompt);
  if (!res.ok) {
    showToast(res.error || "Failed to send prompt", "error");
    finishGenerationUI();
  }
}

// ------------------------------------------------------------------
// Wire UI Buttons & Startup
// ------------------------------------------------------------------

function setupBindings() {
  // Host YAML & Model
  $("hostBrowseYamlBtn").onclick = async () => {
    const p = await window.pywebview.api.browse_yaml();
    if (p) {
      $("hostCfgPath").value = p;
      $("hostLoadYamlBtn").click();
    }
  };
  $("hostLoadYamlBtn").onclick = async () => {
    try {
      const res = await window.pywebview.api.load_host_yaml($("hostCfgPath").value);
      applyHostConfig(res.config);
      showToast("Host config loaded", "success");
    } catch (e) {
      showToast(String(e), "error");
    }
  };
  $("hostSaveYamlBtn").onclick = async () => {
    try {
      await window.pywebview.api.save_host_yaml($("hostCfgPath").value, collectHostConfig());
      showToast("Host config saved", "success");
    } catch (e) {
      showToast(String(e), "error");
    }
  };
  $("hostBrowseModelBtn").onclick = async () => {
    const p = await window.pywebview.api.browse_folder();
    if (p) $("hostModel").value = p;
  };

  // Host Controls
  $("probePeersBtn").onclick = async () => {
    try {
      await window.pywebview.api.probe_peers(collectHostConfig());
    } catch (e) {
      showToast(String(e), "error");
    }
  };
  $("startHostBtn").onclick = async () => {
    try {
      await window.pywebview.api.start_host(collectHostConfig());
    } catch (e) {
      showToast(String(e), "error");
    }
  };
  $("stopHostBtn").onclick = () => window.pywebview.api.stop_host();

  // Chat
  $("sendPromptBtn").onclick = () => sendPrompt();
  $("stopGenBtn").onclick = () => window.pywebview.api.stop_generation();
  $("clearChatBtn").onclick = async () => {
    await window.pywebview.api.clear_chat();
    $("chatMessages").innerHTML = "";
    showToast("Chat history cleared", "info");
  };
  $("promptInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendPrompt();
    }
  });
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.onclick = () => sendPrompt(chip.dataset.prompt);
  });

  // Logs
  $("clearLogsBtn").onclick = () => ($("hostLogConsole").textContent = "");
  $("clearPeerLogsBtn").onclick = () => ($("peerLogConsole").textContent = "");

  // Peer YAML & Model
  $("peerBrowseYamlBtn").onclick = async () => {
    const p = await window.pywebview.api.browse_yaml();
    if (p) {
      $("peerCfgPath").value = p;
      $("peerLoadYamlBtn").click();
    }
  };
  $("peerLoadYamlBtn").onclick = async () => {
    try {
      const res = await window.pywebview.api.load_peer_yaml($("peerCfgPath").value);
      applyPeerConfig(res.config);
      showToast("Peer config loaded", "success");
    } catch (e) {
      showToast(String(e), "error");
    }
  };
  $("peerSaveYamlBtn").onclick = async () => {
    try {
      await window.pywebview.api.save_peer_yaml($("peerCfgPath").value, collectPeerConfig());
      showToast("Peer config saved", "success");
    } catch (e) {
      showToast(String(e), "error");
    }
  };
  $("peerBrowseModelBtn").onclick = async () => {
    const p = await window.pywebview.api.browse_folder();
    if (p) $("peerModel").value = p;
  };
  $("peerDevice").onchange = async () => {
    const hw = await window.pywebview.api.detect_hardware($("peerDevice").value);
    $("peerHwName").textContent = hw.name;
    $("peerHwType").textContent = hw.type.toUpperCase();
    $("peerHwMem").textContent = hw.free_mem;
  };
  $("startPeerBtn").onclick = async () => {
    try {
      await window.pywebview.api.start_peer(collectPeerConfig());
    } catch (e) {
      showToast(String(e), "error");
    }
  };
  $("stopPeerBtn").onclick = () => window.pywebview.api.stop_peer();

  // Modals: Add Peer
  $("openAddPeerBtn").onclick = () => $("addPeerModal").classList.remove("hidden");
  $("cancelAddPeerBtn").onclick = () => $("addPeerModal").classList.add("hidden");
  $("newPeerMode").onchange = () => {
    $("newPeerOutboundFields").classList.toggle("hidden", $("newPeerMode").value !== "outbound");
  };
  $("confirmAddPeerBtn").onclick = () => {
    const name = $("newPeerName").value.trim();
    if (!name) return showToast("Peer name is required", "error");
    state.hostPeers.push({
      name,
      mode: $("newPeerMode").value,
      address: $("newPeerAddr").value.trim(),
      port: Number($("newPeerPort").value || 5001),
    });
    renderPeersList();
    $("newPeerName").value = "";
    $("addPeerModal").classList.add("hidden");
  };

  // Modals: Download Model
  $("hostDownloadBtn").onclick = () => {
    state.downloadTarget = "host";
    $("downloadModal").classList.remove("hidden");
  };
  $("peerDownloadBtn").onclick = () => {
    state.downloadTarget = "peer";
    $("downloadModal").classList.remove("hidden");
  };
  $("cancelDlBtn").onclick = () => $("downloadModal").classList.add("hidden");
  $("confirmDlBtn").onclick = async () => {
    try {
      await window.pywebview.api.download_model($("dlRepoId").value, state.downloadTarget);
    } catch (e) {
      showToast(String(e), "error");
    }
  };
}

window.addEventListener("pywebviewready", async () => {
  setupBindings();
  try {
    const init = await window.pywebview.api.get_initial_state();
    $("hostCfgPath").value = init.host_path;
    applyHostConfig(init.host_config);
    $("peerCfgPath").value = init.peer_path;
    applyPeerConfig(init.peer_config);
    if (init.hardware) {
      $("peerHwName").textContent = init.hardware.name;
      $("peerHwType").textContent = (init.hardware.type || "cpu").toUpperCase();
      $("peerHwMem").textContent = init.hardware.free_mem;
    }
  } catch (e) {
    showToast(`Failed loading initial state: ${e}`, "error");
  }
});
