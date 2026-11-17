import json
from pathlib import Path


def generate_demo_page(sweep_json_path: str = "results/extA_sweep.json", output_html_path: str = "figures/extA_demo.html"):
    with open(sweep_json_path, "r") as f:
        data = json.load(f)

    json_str = json.dumps(data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BUSI Mixed-Precision Sensitivity Knob Demo (Extension A)</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #f8fafc;
    color: #0f172a;
    margin: 0;
    padding: 32px;
  }}
  .container {{
    max-width: 960px;
    margin: 0 auto;
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    padding: 32px;
  }}
  h1 {{ margin-top: 0; color: #1e293b; font-size: 24px; }}
  .subtitle {{ color: #64748b; margin-bottom: 24px; }}
  .control-panel {{
    background: #f1f5f9;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 24px;
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    align-items: center;
  }}
  .control-group {{
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  label {{ font-weight: 600; font-size: 14px; color: #334155; }}
  input[type=range] {{ width: 240px; }}
  input[type=number] {{ width: 100px; padding: 6px; border: 1px solid #cbd5e1; border-radius: 4px; }}
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }}
  .card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
  }}
  .card-label {{ font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }}
  .card-value {{ font-size: 28px; font-weight: 700; color: #0f172a; margin: 8px 0; }}
  .card-sub {{ font-size: 12px; color: #64748b; }}
  .nodes-box {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px;
  }}
  .node-tag {{
    display: inline-block;
    background: #e0f2fe;
    color: #0369a1;
    padding: 4px 10px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 13px;
    margin: 4px 6px 4px 0;
  }}
  .notice {{
    background: #fef3c7;
    border-left: 4px solid #f59e0b;
    padding: 12px;
    font-size: 13px;
    color: #92400e;
    margin-bottom: 20px;
    border-radius: 0 4px 4px 0;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>BUSI Hybrid Mixed-Precision Sensitivity Knob (Extension A)</h1>
  <div class="subtitle">Interactive slider exploring accuracy, size, and latency across sensitivity-ranked node exclusion.</div>

  <div class="notice">
    <strong>Systems Finding:</strong> Boundary QuantizeLinear / DequantizeLinear (QDQ) operations add compute overhead. Consequently, keeping more nodes in FP32 does not always result in monotonic latency changes.
  </div>

  <div class="control-panel">
    <div class="control-group">
      <label for="kSlider">Nodes Kept in FP32 (k): <span id="kVal">0</span></label>
      <input type="range" id="kSlider" min="0" max="9" step="1" value="0">
    </div>
    <div class="control-group">
      <label for="budgetInput">Or Set Max Allowed Dice Drop (%):</label>
      <input type="number" id="budgetInput" min="0.0" max="5.0" step="0.1" placeholder="e.g. 0.5">
    </div>
  </div>

  <div class="metrics-grid">
    <div class="card">
      <div class="card-label">Validation Dice</div>
      <div class="card-value" id="diceVal">--</div>
      <div class="card-sub" id="diceDrop">Drop: --</div>
    </div>
    <div class="card">
      <div class="card-label">Validation HD95</div>
      <div class="card-value" id="hd95Val">--</div>
      <div class="card-sub">Boundary error in pixels</div>
    </div>
    <div class="card">
      <div class="card-label">Model Disk Size</div>
      <div class="card-value" id="sizeVal">--</div>
      <div class="card-sub" id="sizeComp">--</div>
    </div>
    <div class="card">
      <div class="card-label">p50 CPU Latency</div>
      <div class="card-value" id="latVal">--</div>
      <div class="card-sub" id="latComp">--</div>
    </div>
  </div>

  <div class="nodes-box">
    <label>Nodes Retained in FP32 at Current Setting:</label>
    <div id="nodesList" style="margin-top: 10px;">None (Full INT8)</div>
  </div>
</div>

<script>
  const data = {json_str};
  const sweep = data.k_sweep;
  const fp32Val = data.fp32_val;

  const slider = document.getElementById('kSlider');
  const kVal = document.getElementById('kVal');
  const budgetInput = document.getElementById('budgetInput');
  const diceVal = document.getElementById('diceVal');
  const diceDrop = document.getElementById('diceDrop');
  const hd95Val = document.getElementById('hd95Val');
  const sizeVal = document.getElementById('sizeVal');
  const sizeComp = document.getElementById('sizeComp');
  const latVal = document.getElementById('latVal');
  const latComp = document.getElementById('latComp');
  const nodesList = document.getElementById('nodesList');

  slider.max = sweep.length - 1;

  function updateView(idx) {{
    const entry = sweep[idx];
    kVal.innerText = entry.k;
    diceVal.innerText = entry.val_dsc.toFixed(4);
    const drop = fp32Val.dsc - entry.val_dsc;
    diceDrop.innerText = 'Drop vs FP32: ' + (drop >= 0 ? '-' : '+') + Math.abs(drop).toFixed(4);
    hd95Val.innerText = entry.val_hd95.toFixed(2) + ' px';
    sizeVal.innerText = entry.size_mb.toFixed(2) + ' MB';
    sizeComp.innerText = 'FP32: ' + fp32Val.size_mb.toFixed(2) + ' MB (' + (entry.size_mb / fp32Val.size_mb * 100).toFixed(0) + '%)';
    latVal.innerText = entry.p50_latency_ms.toFixed(2) + ' ms';
    latComp.innerText = 'FP32: ' + fp32Val.latency_ms.toFixed(2) + ' ms';

    if (entry.excluded_nodes.length === 0) {{
      nodesList.innerHTML = '<em>None (Full INT8 QDQ graph)</em>';
    }} else {{
      nodesList.innerHTML = entry.excluded_nodes.map(n => '<span class="node-tag">' + n + '</span>').join('');
    }}
  }}

  slider.addEventListener('input', (e) => {{
    updateView(parseInt(e.target.value));
  }});

  budgetInput.addEventListener('input', (e) => {{
    const budgetPct = parseFloat(e.target.value);
    if (!isNaN(budgetPct)) {{
      const maxDrop = budgetPct / 100.0;
      let chosenIdx = 0;
      for (let i = 0; i < sweep.length; i++) {{
        if ((fp32Val.dsc - sweep[i].val_dsc) <= maxDrop) {{
          chosenIdx = i;
          break;
        }}
      }}
      slider.value = chosenIdx;
      updateView(chosenIdx);
    }}
  }});

  updateView(0);
</script>
</body>
</html>
"""
    out_file = Path(output_html_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated Extension A interactive HTML demo at: {out_file.resolve()}")


if __name__ == "__main__":
    sweep_path = Path("results/extA_sweep.json")
    if sweep_path.exists():
        generate_demo_page()
    else:
        print(f"Sweep file {sweep_path} not found yet. It will be generated during Phase 10.")
