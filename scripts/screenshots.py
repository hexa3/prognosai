"""Capture dashboard screenshots for the README (requires a running app on --url)."""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
PAGES = {"overview": "", "machine": "machine", "fleet": "fleet", "planner": "planner", "batch": "batch", "model-card": "model-card"}

errors = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1500, "height": 1900}, device_scale_factor=1)
    for name, path in PAGES.items():
        page.goto(f"{URL}/{path}")
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=60000)
        page.wait_for_timeout(2500)
        page.wait_for_function("!document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120000)
        page.wait_for_timeout(2500)
        exc = page.query_selector_all("[data-testid='stException']")
        if exc:
            errors.append((name, exc[0].inner_text()[:1500]))
        page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
        print("captured", name)
    browser.close()
for name, e in errors:
    print(f"\n!!! EXCEPTION on {name}:\n{e}")

# Trim empty space below the content so README images stay compact.
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

for f in OUT.glob("*.png"):
    im = Image.open(f).convert("RGB")
    a = np.asarray(im)[:, 320:, :]
    rows = np.where((a < 245).any(axis=(1, 2)))[0]
    im.crop((0, 0, im.width, min(im.height, int(rows.max()) + 40))).save(f, optimize=True)

sys.exit(1 if errors else 0)
