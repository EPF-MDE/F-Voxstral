const $ = (id) => document.getElementById(id);
let ws, ctx, node, stream;

function setStatus(text, cls = "") {
  $("status").textContent = text;
  $("status").className = cls;
}

function row(id) {
  let el = document.querySelector(`[data-id="${id}"]`);
  if (!el) {
    el = document.createElement("article");
    el.dataset.id = id;
    el.innerHTML = `<p class="tr">…</p><p class="src"></p><p class="meta"></p>`;
    $("out").prepend(el);
  }
  return el;
}

function handle(msg) {
  if (msg.type === "segment") {
    const el = row(msg.id);
    el.querySelector(".src").textContent = msg.transcript;
    el.querySelector(".meta").textContent = `${msg.audio_s}s audio · ASR ${msg.asr_ms} ms`;
  } else if (msg.type === "translation") {
    const el = row(msg.id);
    el.querySelector(".tr").textContent = msg.translation;
    el.querySelector(".meta").textContent += ` · total ${msg.latency_ms} ms`;
  } else if (msg.type === "error" || msg.type === "warning") {
    setStatus(msg.message, "err");
  }
}

function sendConfig() {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({
      type: "config", source: $("src").value, target: $("tgt").value,
      sample_rate: ctx ? ctx.sampleRate : 16000,
    }));
  }
}

async function start() {
  $("start").disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (e) {
    setStatus("Micro refusé : " + e.message, "err");
    $("start").disabled = false;
    return;
  }
  ctx = new AudioContext({ sampleRate: 16000 });
  await ctx.audioWorklet.addModule("/static/pcm-worklet.js");
  node = new AudioWorkletNode(ctx, "pcm-writer", { numberOfOutputs: 0 });
  ctx.createMediaStreamSource(stream).connect(node);

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    sendConfig();
    node.port.onmessage = (e) => {
      if (ws.readyState === 1) ws.send(e.data.pcm);
      $("level").value = e.data.level;
    };
    setStatus("à l'écoute…", "ok");
    $("stop").disabled = false;
  };
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => { setStatus("déconnecté"); cleanup(); };
}

function cleanup() {
  if (node) node.port.onmessage = null;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (ctx) ctx.close();
  node = stream = ctx = null;
  $("start").disabled = false;
  $("stop").disabled = true;
}

function stop() {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ type: "flush" }));
    setTimeout(() => ws.close(), 15000); // leave time for the last segment
  }
  cleanup();
  setStatus("arrêt… (dernier segment en cours)");
}

$("start").onclick = start;
$("stop").onclick = stop;
$("src").onchange = sendConfig;
$("tgt").onchange = sendConfig;
