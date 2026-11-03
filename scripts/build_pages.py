"""Generate index.html: the full dashboard running in the browser via stlite (Streamlit on Pyodide/WebAssembly).

GitHub Pages serves the repository root. The page fetches the app source, models and data
from the same site, so no Python server is needed. Re-run after changing the app or models:
    python scripts/build_pages.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STLITE = "1.12.0"

patterns = [
    "streamlit_app.py",
    "prognosai/*.py",
    "models/*.txt",
    "models/*.json",
    "models/*_test_predictions.csv",
    "reports/figures/*.png",
    "data/ai4i/ai4i2020.csv",
    "data/cmapss/test_FD001.txt",
]
files = sorted({p.relative_to(ROOT).as_posix() for pat in patterns for p in ROOT.glob(pat)})
mount = {f: {"url": f"./{f}"} for f in files}

html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PrognosAI · Predictive Maintenance</title>
  <meta name="description" content="Explainable, cost-aware predictive maintenance. Neural Storm 2026, FT-03. Runs fully in your browser." />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE}/build/stlite.css" />
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #fff; }}
    #boot {{ max-width: 640px; margin: 18vh auto 0; padding: 0 16px; color: #1e293b; text-align: center; }}
    #boot h1 {{ font-size: 2rem; margin-bottom: .25rem; }}
    #boot p {{ color: #64748b; line-height: 1.5; }}
    .bar {{ height: 4px; background: #e2e8f0; border-radius: 2px; overflow: hidden; margin: 24px auto; max-width: 320px; }}
    .bar::after {{ content: ""; display: block; height: 100%; width: 40%; background: #2563eb; animation: s 1.2s infinite ease-in-out; }}
    @keyframes s {{ 0% {{ transform: translateX(-100%); }} 100% {{ transform: translateX(250%); }} }}
  </style>
</head>
<body>
  <div id="root">
    <div id="boot">
      <h1>PrognosAI</h1>
      <p>Loading the full Python ML stack in your browser (Streamlit + LightGBM on WebAssembly).<br/>
         First load takes about 30–60 s, and nothing is sent to a server.</p>
      <div class="bar"></div>
      <p><a href="https://github.com/hexa3/prognosai">Source code &amp; README</a></p>
    </div>
  </div>
  <script type="module">
    import {{ mount }} from "https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE}/build/stlite.js";
    mount({{
      requirements: ["lightgbm", "plotly"],
      entrypoint: "streamlit_app.py",
      streamlitConfig: {{ "theme.base": "light", "theme.primaryColor": "#2563eb", "client.toolbarMode": "viewer" }},
      files: {json.dumps(mount, indent=8)},
    }}, document.getElementById("root"));
  </script>
</body>
</html>
"""
(ROOT / "index.html").write_text(html)
(ROOT / ".nojekyll").write_text("")
print(f"index.html written with {len(files)} mounted files")
