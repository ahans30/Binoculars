#!/usr/bin/env python3
"""
Mac MPS runner for the B1 blinded scoring experiment.
Mirrors the output schema of llm-pr-binoculars-runner (JosephAyo/llm-pr-binoculars-runner)
but runs on Apple GPU via MPS instead of CUDA.

Usage:
    python experiments/b1_mac_runner.py smoke   # score the 10 smoke records
    python experiments/b1_mac_runner.py full    # score all 5630 records (resumable)
    python experiments/b1_mac_runner.py verify  # check output completeness

Environment variables:
    B1_DATA_ROOT        path to folder with corpus.csv, smoke_ids.csv, manifest.json
                        default: AyoExperimentInput/llm-prs-b1-binoculars-input-v1
    B1_OUTPUT_ROOT      where results are written  (default: output/b1)
    B1_MIN_FREE_GB      abort if available unified memory falls below this (default: 4)
    B1_DASHBOARD_PORT   port for the live web dashboard (default: 7861)
    B1_SECONDARY_ROOT   optional second folder to mirror results into (e.g. iCloud path)
"""

import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get(
    "B1_DATA_ROOT",
    REPO_ROOT / "AyoExperimentInput" / "llm-prs-b1-binoculars-input-v1",
))
OUTPUT_ROOT = Path(os.environ.get("B1_OUTPUT_ROOT", REPO_ROOT / "output" / "b1"))
MIN_FREE_GB = float(os.environ.get("B1_MIN_FREE_GB", "4"))
DASHBOARD_PORT = int(os.environ.get("B1_DASHBOARD_PORT", "7861"))
SECONDARY_ROOT = os.environ.get("B1_SECONDARY_ROOT", "")

DETECTOR_NAME = "binoculars"
DETECTOR_VERSION = "macos-mps"
OBSERVER_MODEL = "tiiuae/falcon-7b"
PERFORMER_MODEL = "tiiuae/falcon-7b-instruct"
MAX_LENGTH = 512
MODE = "low-fpr"
THRESHOLD = 0.8536432310785527
THRESHOLD_SOURCE = "official-low-fpr"
PRECISION = "bfloat16"
MAX_ATTEMPTS = 2

_MEM_ERROR_PATTERNS = (
    "out of memory",
    "mps backend out of memory",
    "cuda out of memory",
    "memory error",
    "allocationerror",
    "outofmemoryerror",
)


def is_memory_error(exc: Exception) -> bool:
    msg = (type(exc).__name__ + " " + str(exc)).lower()
    return any(p in msg for p in _MEM_ERROR_PATTERNS)


# ── dashboard state ───────────────────────────────────────────────────────────

_state_lock = threading.Lock()
_state: dict = {
    "status": "idle",
    "run_id": "",
    "mode": "",
    "device": "mps",
    "total": 0,
    "done": 0,
    "ai": 0,
    "human": 0,
    "failed": 0,
    "mem_errors": 0,
    "mem_free_gb": 0.0,
    "mps_gb": 0.0,
    "total_mem_gb": 64.0,
    "elapsed_s": 0.0,
    "speed_rpm": 0.0,
    "eta_s": None,
    "started_at": None,
    "recent": [],
}


def state_update(**kwargs) -> None:
    with _state_lock:
        _state.update(kwargs)


def state_snapshot() -> dict:
    with _state_lock:
        return dict(_state)


# ── dashboard HTML ────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!doctype html>
<title>B1 Experiment Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap">
<style>
  :root {
    --bg:       #0e0d0b;
    --surface:  #181614;
    --border:   #2c2926;
    --text:     #ddd5c8;
    --muted:    #6b6258;
    --accent:   #c97b25;
    --accent-d: #7d4c16;
    --ai:       #c45c5c;
    --ai-bg:    #2a1414;
    --human:    #5a9e72;
    --human-bg: #102218;
    --fail:     #7a6040;
    --mono:     "JetBrains Mono", monospace;
    --sans:     "Inter", system-ui, sans-serif;
  }
  @media (prefers-color-scheme: light) {
    :root:not([data-theme="dark"]) {
      --bg:       #f5f2ee;
      --surface:  #ffffff;
      --border:   #ddd8d0;
      --text:     #2a2420;
      --muted:    #8a7f74;
      --accent:   #b56d1a;
      --accent-d: #e8d4b8;
      --ai:       #b03030;
      --ai-bg:    #fce8e8;
      --human:    #2e7a4e;
      --human-bg: #e4f5eb;
      --fail:     #7a5820;
    }
  }
  :root[data-theme="light"] {
    --bg:       #f5f2ee;
    --surface:  #ffffff;
    --border:   #ddd8d0;
    --text:     #2a2420;
    --muted:    #8a7f74;
    --accent:   #b56d1a;
    --accent-d: #e8d4b8;
    --ai:       #b03030;
    --ai-bg:    #fce8e8;
    --human:    #2e7a4e;
    --human-bg: #e4f5eb;
    --fail:     #7a5820;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.5;
    min-height: 100vh;
    padding: 0 0 40px;
  }

  /* ── header ── */
  .header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 16px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
  }
  .header-left { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .header h1 { font-family: var(--mono); font-size: 15px; font-weight: 600; color: var(--accent); letter-spacing: 0.04em; }
  .run-id { font-family: var(--mono); font-size: 12px; color: var(--muted); }
  .status-badge {
    font-family: var(--mono); font-size: 11px; font-weight: 600;
    padding: 3px 10px; border-radius: 20px; letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .status-badge.running  { background: var(--accent-d); color: var(--accent); }
  .status-badge.complete { background: var(--human-bg); color: var(--human); }
  .status-badge.idle     { background: var(--border);   color: var(--muted); }
  .status-badge.loading  { background: var(--border);   color: var(--muted); }
  .status-badge.aborted  { background: var(--ai-bg);    color: var(--ai); }

  /* ── main content ── */
  .content { max-width: 1100px; margin: 0 auto; padding: 28px 28px 0; display: flex; flex-direction: column; gap: 20px; }

  /* ── progress block ── */
  .progress-block { display: flex; flex-direction: column; gap: 8px; }
  .progress-label {
    display: flex; justify-content: space-between; align-items: baseline;
    font-family: var(--mono); font-size: 12px; color: var(--muted);
  }
  .progress-label .pct { font-size: 22px; font-weight: 600; color: var(--text); }
  .progress-label .counts { font-size: 13px; }
  .bar-track {
    height: 6px; background: var(--border); border-radius: 4px; overflow: hidden;
  }
  .bar-fill {
    height: 100%; background: var(--accent); border-radius: 4px;
    transition: width 0.6s ease;
  }

  /* ── stat tiles ── */
  .tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .tile {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 16px 18px;
  }
  .tile-label { font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
  .tile-value { font-family: var(--mono); font-size: 28px; font-weight: 600; color: var(--text); font-variant-numeric: tabular-nums; line-height: 1; }
  .tile-sub   { font-family: var(--mono); font-size: 12px; color: var(--muted); margin-top: 4px; }
  .tile.ai .tile-value { color: var(--ai); }
  .tile.human .tile-value { color: var(--human); }

  /* ── gauges ── */
  .gauges { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 16px 18px; display: flex; flex-direction: column; gap: 12px; }
  .gauge-row { display: flex; align-items: center; gap: 12px; }
  .gauge-label { font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); width: 110px; flex-shrink: 0; }
  .gauge-track { flex: 1; height: 4px; background: var(--border); border-radius: 4px; overflow: hidden; }
  .gauge-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
  .gauge-fill.mem  { background: var(--accent); }
  .gauge-fill.mps  { background: var(--human); }
  .gauge-val { font-family: var(--mono); font-size: 12px; color: var(--text); width: 110px; text-align: right; flex-shrink: 0; font-variant-numeric: tabular-nums; }
  .speed-row { display: flex; gap: 24px; padding-top: 4px; border-top: 1px solid var(--border); }
  .speed-item { display: flex; flex-direction: column; gap: 2px; }
  .speed-item .s-label { font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
  .speed-item .s-val   { font-family: var(--mono); font-size: 15px; font-weight: 600; color: var(--text); font-variant-numeric: tabular-nums; }

  /* ── recent records table ── */
  .table-block { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
  .table-header { padding: 12px 18px; border-bottom: 1px solid var(--border); font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
  .records-table { width: 100%; border-collapse: collapse; overflow-x: auto; display: block; }
  .records-table th {
    font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--muted); text-align: left; padding: 8px 14px;
    border-bottom: 1px solid var(--border); white-space: nowrap;
  }
  .records-table td {
    font-family: var(--mono); font-size: 12px; padding: 7px 14px;
    border-bottom: 1px solid var(--border); white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }
  .records-table tr:last-child td { border-bottom: none; }
  .records-table tr:hover td { background: var(--border); }
  .chip {
    display: inline-block; font-size: 10px; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; padding: 2px 7px; border-radius: 3px;
  }
  .chip.ai      { background: var(--ai-bg);    color: var(--ai); }
  .chip.human   { background: var(--human-bg); color: var(--human); }
  .chip.fail    { background: var(--border);   color: var(--fail); }
  .chip.mem-oom { background: #1e1400; color: #d48f00; border: 1px solid #4a3800; }

  /* ── empty state ── */
  .empty { text-align: center; padding: 40px; color: var(--muted); font-family: var(--mono); font-size: 13px; }

  /* ── pulse dot ── */
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
  .pulse { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--accent); margin-right: 6px; animation: pulse 1.4s ease-in-out infinite; vertical-align: middle; }
</style>

<div class="header">
  <div class="header-left">
    <h1>B1 EXPERIMENT MONITOR</h1>
    <span class="run-id" id="run-id">—</span>
  </div>
  <span class="status-badge idle" id="status-badge">idle</span>
</div>

<div class="content">

  <div class="progress-block">
    <div class="progress-label">
      <span class="pct" id="pct">0%</span>
      <span class="counts" id="counts">— / — records</span>
    </div>
    <div class="bar-track"><div class="bar-fill" id="bar" style="width:0%"></div></div>
  </div>

  <div class="tiles">
    <div class="tile">
      <div class="tile-label">Scored</div>
      <div class="tile-value" id="t-done">—</div>
      <div class="tile-sub" id="t-total">of — total</div>
    </div>
    <div class="tile ai">
      <div class="tile-label">AI detected</div>
      <div class="tile-value" id="t-ai">—</div>
      <div class="tile-sub" id="t-ai-pct">—%</div>
    </div>
    <div class="tile human">
      <div class="tile-label">Human</div>
      <div class="tile-value" id="t-human">—</div>
      <div class="tile-sub" id="t-human-pct">—%</div>
    </div>
    <div class="tile">
      <div class="tile-label">Failed</div>
      <div class="tile-value" id="t-failed">—</div>
      <div class="tile-sub" id="t-mem-errors">&nbsp;</div>
    </div>
  </div>

  <div class="gauges">
    <div class="gauge-row">
      <span class="gauge-label">Mem free</span>
      <div class="gauge-track"><div class="gauge-fill mem" id="g-mem" style="width:0%"></div></div>
      <span class="gauge-val" id="g-mem-val">— GB</span>
    </div>
    <div class="gauge-row">
      <span class="gauge-label">MPS used</span>
      <div class="gauge-track"><div class="gauge-fill mps" id="g-mps" style="width:0%"></div></div>
      <span class="gauge-val" id="g-mps-val">— GB</span>
    </div>
    <div class="speed-row">
      <div class="speed-item"><span class="s-label">Device</span><span class="s-val" id="s-device">—</span></div>
      <div class="speed-item"><span class="s-label">Speed</span><span class="s-val" id="s-speed">—</span></div>
      <div class="speed-item"><span class="s-label">Elapsed</span><span class="s-val" id="s-elapsed">—</span></div>
      <div class="speed-item"><span class="s-label">ETA</span><span class="s-val" id="s-eta">—</span></div>
      <div class="speed-item"><span class="s-label">Mode</span><span class="s-val" id="s-mode">—</span></div>
    </div>
  </div>

  <div class="table-block">
    <div class="table-header"><span class="pulse" id="pulse" hidden></span>Recent records</div>
    <div style="overflow-x:auto">
      <table class="records-table">
        <thead>
          <tr>
            <th>Record ID</th>
            <th>Label</th>
            <th>Raw score</th>
            <th>Tokens</th>
            <th>ms</th>
            <th>Time (UTC)</th>
          </tr>
        </thead>
        <tbody id="tbody">
          <tr><td colspan="6" class="empty">Waiting for data…</td></tr>
        </tbody>
      </table>
    </div>
  </div>

</div>

<script>
function fmt(n, d=0){ return n==null?'—':Number(n).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}); }
function fmtDur(s){
  if(s==null||s<0) return '—';
  const h=Math.floor(s/3600), m=Math.floor((s%3600)/60), sec=Math.floor(s%60);
  if(h>0) return h+'h '+m+'m';
  if(m>0) return m+'m '+sec+'s';
  return sec+'s';
}
function chip(label){
  if(!label) return '<span class="chip fail">—</span>';
  const cls = label==='AI'?'ai':label==='human'?'human':label==='mem_oom'?'mem-oom':'fail';
  const txt = label==='mem_oom'?'MEM OOM':label;
  return `<span class="chip ${cls}">${txt}</span>`;
}

async function poll(){
  try {
    const r = await fetch('/api/status');
    if(!r.ok) return;
    const d = await r.json();

    document.getElementById('run-id').textContent = d.run_id || '—';

    const badge = document.getElementById('status-badge');
    badge.textContent = d.status;
    badge.className = 'status-badge ' + (d.status||'idle');

    const done=d.done||0, total=d.total||0;
    const pct = total>0 ? (done/total*100) : 0;
    document.getElementById('pct').textContent = pct.toFixed(1)+'%';
    document.getElementById('counts').textContent = fmt(done)+' / '+fmt(total)+' records';
    document.getElementById('bar').style.width = pct.toFixed(2)+'%';

    document.getElementById('t-done').textContent   = fmt(done);
    document.getElementById('t-total').textContent  = 'of '+fmt(total)+' total';
    document.getElementById('t-ai').textContent     = fmt(d.ai);
    document.getElementById('t-human').textContent  = fmt(d.human);
    document.getElementById('t-failed').textContent = fmt(d.failed);
    const me = d.mem_errors||0;
    document.getElementById('t-mem-errors').textContent = me>0 ? '⚠ '+me+' mem OOM' : ' ';

    const scored = (d.ai||0)+(d.human||0)+(d.failed||0);
    const aiPct   = scored>0 ? (d.ai/scored*100)   : 0;
    const humPct  = scored>0 ? (d.human/scored*100) : 0;
    document.getElementById('t-ai-pct').textContent    = aiPct.toFixed(1)+'% of scored';
    document.getElementById('t-human-pct').textContent = humPct.toFixed(1)+'% of scored';

    const totalMem = d.total_mem_gb||64;
    const memFree  = d.mem_free_gb||0;
    const mpUsed   = d.mps_gb||0;
    document.getElementById('g-mem').style.width    = (memFree/totalMem*100).toFixed(1)+'%';
    document.getElementById('g-mem-val').textContent = memFree.toFixed(1)+' GB free';
    document.getElementById('g-mps').style.width    = (mpUsed/totalMem*100).toFixed(1)+'%';
    document.getElementById('g-mps-val').textContent = mpUsed.toFixed(1)+' GB';

    document.getElementById('s-device').textContent  = (d.device||'—').toUpperCase();
    document.getElementById('s-speed').textContent   = d.speed_rpm!=null ? fmt(d.speed_rpm,1)+' rec/min' : '—';
    document.getElementById('s-elapsed').textContent = fmtDur(d.elapsed_s);
    document.getElementById('s-eta').textContent     = fmtDur(d.eta_s);
    document.getElementById('s-mode').textContent    = d.mode||'—';

    document.getElementById('pulse').hidden = d.status !== 'running';

    const tbody = document.getElementById('tbody');
    const rows = d.recent||[];
    if(rows.length===0){
      tbody.innerHTML = '<tr><td colspan="6" class="empty">Waiting for data…</td></tr>';
    } else {
      tbody.innerHTML = rows.map(row=>`
        <tr>
          <td>${row.record_id||'—'}</td>
          <td>${chip(row.label)}</td>
          <td>${row.raw_score!=null ? Number(row.raw_score).toFixed(4) : '—'}</td>
          <td>${fmt(row.tokens)}</td>
          <td>${fmt(row.elapsed_ms)}</td>
          <td>${row.time||'—'}</td>
        </tr>`).join('');
    }
  } catch(e) { /* server not yet ready */ }
}

poll();
setInterval(poll, 2000);
</script>
"""


# ── HTTP server ───────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = DASHBOARD_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            body = json.dumps(state_snapshot()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, *_):
        pass  # suppress access log noise


def start_dashboard(port: int, run_id: str) -> None:
    state_update(status="loading", run_id=run_id)
    server = HTTPServer(("", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"\n  Dashboard → http://localhost:{port}\n")


# ── memory helpers ────────────────────────────────────────────────────────────

def _total_mem_gb() -> float:
    try:
        out = subprocess.check_output(["sysctl", "hw.memsize"], stderr=subprocess.DEVNULL).decode()
        return int(out.split()[1]) / 1024 ** 3
    except Exception:
        return 64.0


def available_memory_gb() -> float:
    try:
        vm = subprocess.check_output(["vm_stat"], stderr=subprocess.DEVNULL).decode()
        page_size = 16384
        lines = {}
        for line in vm.splitlines()[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                lines[k.strip()] = int(v.strip().rstrip("."))
        free = lines.get("Pages free", 0)
        inactive = lines.get("Pages inactive", 0)
        return (free + inactive) * page_size / 1024 ** 3
    except Exception:
        return float("inf")


def mps_allocated_gb() -> float:
    try:
        return torch.mps.driver_allocated_memory() / 1024 ** 3
    except Exception:
        return 0.0


def check_memory(label: str = "") -> None:
    avail = available_memory_gb()
    mps = mps_allocated_gb()
    tag = f" [{label}]" if label else ""
    state_update(mem_free_gb=round(avail, 2), mps_gb=round(mps, 2))
    if avail < MIN_FREE_GB:
        state_update(status="aborted")
        print(
            f"\nMEMORY ABORT{tag}: only {avail:.2f} GB available "
            f"(threshold {MIN_FREE_GB} GB).  MPS holds {mps:.2f} GB.\n"
            "Results written so far are intact — re-run to resume."
        )
        sys.exit(2)


# ── data helpers ──────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).digest().hex()


def sha256_dict(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def validate_data():
    corpus_path = DATA_ROOT / "corpus.csv"
    smoke_path = DATA_ROOT / "smoke_ids.csv"
    manifest_path = DATA_ROOT / "manifest.json"

    for p in (corpus_path, smoke_path, manifest_path):
        if not p.exists():
            sys.exit(f"Missing file: {p}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    actual_corpus_sha = sha256_file(corpus_path)
    if actual_corpus_sha != manifest["corpus_sha256"]:
        sys.exit(f"corpus.csv SHA256 mismatch\n  expected: {manifest['corpus_sha256']}\n  actual:   {actual_corpus_sha}")

    actual_smoke_sha = sha256_file(smoke_path)
    if actual_smoke_sha != manifest["smoke_sha256"]:
        sys.exit(f"smoke_ids.csv SHA256 mismatch\n  expected: {manifest['smoke_sha256']}\n  actual:   {actual_smoke_sha}")

    with open(corpus_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != manifest["expected_rows"]:
        sys.exit(f"Expected {manifest['expected_rows']} rows, got {len(rows)}")

    bad = [row["record_id"] for row in rows if sha256_text(row["text"]) != row["text_sha256"]]
    if bad:
        sys.exit(f"text_sha256 mismatch for {len(bad)} records: {bad[:5]}")

    rows.sort(key=lambda r: int(r["scoring_order"]))

    with open(smoke_path, newline="", encoding="utf-8") as f:
        smoke_ids = {r["record_id"] for r in csv.DictReader(f)}

    manifest_sha = sha256_file(manifest_path)
    print(f"Data validated  corpus={len(rows)} rows  smoke={len(smoke_ids)} ids")
    print(f"Manifest SHA256: {manifest_sha}")
    return manifest, rows, smoke_ids, manifest_sha


def load_completed(attempts_path: Path):
    done: set[str] = set()
    tally = {"AI": 0, "human": 0, "failed": 0, "mem_errors": 0}
    if not attempts_path.exists():
        return done, tally
    seen: dict = {}   # rid → {"label": str|None, "mem": bool}
    with open(attempts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec["record_id"]
            if rec.get("success"):
                done.add(rid)
                seen[rid] = {"label": rec.get("predicted_label"), "mem": False}
            else:
                mem = rec.get("memory_error", False)
                if rid not in seen:
                    seen[rid] = {"label": None, "mem": mem}
                elif mem:
                    seen[rid]["mem"] = True
    for info in seen.values():
        label = info["label"]
        if label in ("AI", "human"):
            tally[label] += 1
        else:
            tally["failed"] += 1
            if info["mem"]:
                tally["mem_errors"] += 1
    return done, tally


# ── scoring loop ──────────────────────────────────────────────────────────────

def score_records(bino, records, manifest_sha, attempts_path, results_path):
    detector_cfg = {
        "observer": OBSERVER_MODEL, "performer": PERFORMER_MODEL,
        "mode": MODE, "threshold": THRESHOLD,
        "max_length": MAX_LENGTH, "precision": PRECISION,
    }
    detector_cfg_sha = sha256_dict(detector_cfg)

    completed, tally = load_completed(attempts_path)
    pending = [r for r in records if r["record_id"] not in completed]
    total = len(records)
    skipped = total - len(pending)

    print(f"Records: {total} total  {skipped} already done  {len(pending)} to score")
    if skipped:
        print(f"  Resuming tally — AI={tally['AI']}  human={tally['human']}  "
              f"failed={tally['failed']}  mem_errors={tally['mem_errors']}")

    avail = available_memory_gb()
    mps = mps_allocated_gb()
    total_mem = _total_mem_gb()
    print(f"Unified memory available: {avail:.1f} GB  (abort threshold: {MIN_FREE_GB} GB)")
    check_memory("pre-scoring")

    state_update(
        status="running",
        total=total, done=skipped,
        ai=tally["AI"], human=tally["human"], failed=tally["failed"],
        mem_free_gb=round(avail, 2), mps_gb=round(mps, 2), total_mem_gb=round(total_mem, 1),
    )

    run_start = time.monotonic()
    recent: list[dict] = []
    attempt_log = open(attempts_path, "a", buffering=1)

    try:
        for i, row in enumerate(pending, 1):
            record_id = row["record_id"]
            text = row["text"]
            scored_label = None
            scored_raw = None
            scored_tokens = None
            scored_ms = None
            scored_mem_err = False

            for attempt_num in range(1, MAX_ATTEMPTS + 1):
                started = datetime.now(timezone.utc)
                t0 = time.monotonic()
                result = {
                    "record_id": record_id,
                    "detector_name": DETECTOR_NAME,
                    "detector_version": DETECTOR_VERSION,
                    "observer_model": OBSERVER_MODEL,
                    "observer_revision": "cached",
                    "performer_model": PERFORMER_MODEL,
                    "performer_revision": "cached",
                    "precision": PRECISION,
                    "max_length": MAX_LENGTH,
                    "mode": MODE,
                    "threshold": THRESHOLD,
                    "threshold_source": THRESHOLD_SOURCE,
                    "input_sha256": row["text_sha256"],
                    "data_manifest_sha256": manifest_sha,
                    "detector_config_sha256": detector_cfg_sha,
                    "attempt_number": attempt_num,
                    "started_at_utc": started.isoformat(),
                }
                try:
                    tokens = bino.tokenizer(text, return_tensors="pt")
                    input_tokens = tokens["input_ids"].shape[-1]
                    used_tokens = min(input_tokens, MAX_LENGTH)
                    truncated = input_tokens > MAX_LENGTH

                    if input_tokens < 2:
                        raise ValueError(f"Too few tokens: {input_tokens}")

                    raw_score = bino.compute_score(text)
                    ai_score = -raw_score
                    predicted_label = "AI" if raw_score < THRESHOLD else "human"
                    scored_label = predicted_label
                    scored_raw = raw_score
                    scored_tokens = input_tokens
                    scored_ms = int((time.monotonic() - t0) * 1000)

                    result.update({
                        "raw_score": raw_score, "ai_score": ai_score,
                        "predicted_label": predicted_label,
                        "input_tokens": input_tokens, "used_input_tokens": used_tokens,
                        "truncated": truncated, "success": True,
                        "error_type": None, "error_fingerprint": None,
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_ms": scored_ms,
                    })
                    attempt_log.write(json.dumps(result) + "\n")
                    break

                except Exception as exc:
                    elapsed_ms = int((time.monotonic() - t0) * 1000)
                    fp = hashlib.sha256(traceback.format_exc().encode()).hexdigest()[:16]
                    mem_err = is_memory_error(exc)
                    if mem_err:
                        scored_mem_err = True
                    result.update({
                        "raw_score": None, "ai_score": None, "predicted_label": None,
                        "input_tokens": None, "used_input_tokens": None, "truncated": None,
                        "success": False, "error_type": type(exc).__name__,
                        "error_fingerprint": fp,
                        "memory_error": mem_err,
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_ms": elapsed_ms,
                    })
                    attempt_log.write(json.dumps(result) + "\n")
                    if attempt_num == MAX_ATTEMPTS:
                        tag = " [MEM OOM]" if mem_err else ""
                        print(f"  FAILED{tag} {record_id}: {exc}")

            # Free MPS cache after every record
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            # Update tally
            if scored_label in ("AI", "human"):
                tally[scored_label] += 1
            else:
                tally["failed"] += 1
                if scored_mem_err:
                    tally["mem_errors"] += 1

            # Update recent log (most recent first, cap at 50)
            recent.insert(0, {
                "record_id": record_id,
                "label": scored_label if scored_label else ("mem_oom" if scored_mem_err else "failed"),
                "raw_score": round(scored_raw, 4) if scored_raw is not None else None,
                "tokens": scored_tokens,
                "elapsed_ms": scored_ms,
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            })
            if len(recent) > 50:
                recent.pop()

            done_count = skipped + i
            scored_count = tally["AI"] + tally["human"] + tally["failed"]
            ai_pct = 100 * tally["AI"] / scored_count if scored_count else 0
            elapsed = time.monotonic() - run_start
            speed_rpm = (i / elapsed * 60) if elapsed > 0 else 0
            remaining = total - done_count
            eta_s = (remaining / (i / elapsed)) if i > 0 and elapsed > 0 else None
            avail = available_memory_gb()
            mps_gb = mps_allocated_gb()

            state_update(
                done=done_count,
                ai=tally["AI"], human=tally["human"],
                failed=tally["failed"], mem_errors=tally["mem_errors"],
                mem_free_gb=round(avail, 2), mps_gb=round(mps_gb, 2),
                elapsed_s=round(elapsed, 1),
                speed_rpm=round(speed_rpm, 1),
                eta_s=round(eta_s, 0) if eta_s else None,
                recent=list(recent),
            )

            print(
                f"  [{done_count}/{total}]  "
                f"AI={tally['AI']} ({ai_pct:.1f}%)  human={tally['human']}  failed={tally['failed']}  "
                f"mem={avail:.1f}GB free  mps={mps_gb:.1f}GB"
            )
            check_memory(f"after record {done_count}")
            if done_count % 100 == 0:
                _mirror_outputs(attempts_path, results_path)

    finally:
        attempt_log.close()

    state_update(status="complete")

    # Build final results.csv
    best: dict = {}
    with open(attempts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec["record_id"]
            prev = best.get(rid)
            if prev is None or (rec["success"] and not prev["success"]):
                best[rid] = rec

    ordered = [best[r["record_id"]] for r in records if r["record_id"] in best]
    if ordered:
        fieldnames = list(ordered[0].keys())
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered)
        print(f"Wrote {len(ordered)} rows → {results_path}")
        _mirror_outputs(attempts_path, results_path)


# ── secondary mirror ──────────────────────────────────────────────────────────

def _mirror_outputs(*paths: Path) -> None:
    if not SECONDARY_ROOT:
        return
    dest = Path(SECONDARY_ROOT)
    dest.mkdir(parents=True, exist_ok=True)
    for src in paths:
        if src.exists():
            import shutil
            shutil.copy2(src, dest / src.name)
    print(f"Mirrored results → {dest}")


# ── verify ────────────────────────────────────────────────────────────────────

def verify(records, results_path):
    if not results_path.exists():
        print("No results file found.")
        return
    with open(results_path, newline="") as f:
        rows = list(csv.DictReader(f))
    scored_ids = {r["record_id"] for r in rows}
    expected_ids = {r["record_id"] for r in records}
    missing = expected_ids - scored_ids
    success = sum(1 for r in rows if r["success"] == "True")
    failed = sum(1 for r in rows if r["success"] != "True")
    print(f"Results: {len(rows)} rows  success={success}  failed={failed}  missing={len(missing)}")
    if missing:
        print(f"Missing record_ids (first 10): {sorted(missing)[:10]}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("smoke", "full", "verify"):
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    manifest, all_records, smoke_ids, manifest_sha = validate_data()

    if cmd == "verify":
        verify(all_records, OUTPUT_ROOT / "results.csv")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {'Apple GPU (MPS)' if device == 'mps' else 'CPU (slow)'}")

    print("Loading Binoculars models…")
    sys.path.insert(0, str(REPO_ROOT))
    from binoculars import Binoculars
    bino = Binoculars(mode="low-fpr", max_token_observed=MAX_LENGTH)
    print(f"Models loaded on {bino.device_1}")

    total_mem = _total_mem_gb()
    state_update(device=bino.device_1, total_mem_gb=round(total_mem, 1))
    check_memory("post-model-load")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None

    if cmd == "smoke":
        records = [r for r in all_records if r["record_id"] in smoke_ids]
        attempts_path = OUTPUT_ROOT / "smoke_attempts.jsonl"
        results_path  = OUTPUT_ROOT / "smoke_results.csv"
        mode_label = "smoke"
    else:
        records = all_records
        attempts_path = OUTPUT_ROOT / "attempts.jsonl"
        results_path  = OUTPUT_ROOT / "results.csv"
        mode_label = "full"

    if limit is not None:
        records = records[:limit]
        print(f"Limiting to {len(records)} records")

    run_id = manifest.get("run_id", "b1")
    state_update(mode=mode_label, total=len(records), started_at=datetime.now(timezone.utc).isoformat())
    start_dashboard(DASHBOARD_PORT, run_id)

    score_records(bino, records, manifest_sha, attempts_path, results_path)


if __name__ == "__main__":
    main()
