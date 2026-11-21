// Overlap audit of the interface: opens every toolbar panel in turn at four
// screen sizes and both themes, and fails if more than one panel is open, a
// panel covers the map toolbar, a panel leaves the viewport or the page throws.
//
//   SYNTRA_URL=http://localhost:5291 SYNTRA_EMAIL=… SYNTRA_PASSWORD=… node scripts/ui_audit.mjs
//
// Needs Playwright with Chromium (npm i -D playwright && npx playwright install chromium)
// and a running engine + interface. Screenshots go to ui-audit/ (not in git).
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const URL = process.env.SYNTRA_URL || "http://localhost:5291";
const PANELS = ".analysis-flyout, .train-flyout, .perf-flyout, .lib-flyout, .layers-panel, .rs-flyout";
const TOOLS = ["layers", "rs", "train", "samples", "analysis", "performance", "layers", "settings", "rs", "basemap"];
const SIZES = [[1440, 900], [1280, 720], [768, 1024], [390, 844]];
const AOI = { type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[[55.7, 24.2], [55.73, 24.2], [55.73, 24.225], [55.7, 24.225], [55.7, 24.2]]] } };

mkdirSync("ui-audit", { recursive: true });
const browser = await chromium.launch();
const problems = [];
for (const [w, h] of SIZES) for (const theme of ["dark", "light"]) {
  const page = await browser.newPage({ viewport: { width: w, height: h } });
  const tag = `${w}x${h} ${theme}`;
  page.on("pageerror", (e) => problems.push(`${tag}: page error ${e.message}`));
  await page.goto(URL);
  if (process.env.SYNTRA_EMAIL) {
    await page.evaluate(([email, password]) => fetch("/api/auth/signin", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }) }),
      [process.env.SYNTRA_EMAIL, process.env.SYNTRA_PASSWORD || ""]);
  }
  await page.evaluate(([t, aoi]) => { localStorage.setItem("syntra_theme", t); localStorage.setItem("syntra_project", JSON.stringify({ aoi })); }, [theme, AOI]);
  await page.goto(URL);
  await page.waitForTimeout(1500);
  if (w < 700) { const m = await page.$(".mobile-switch button:last-child"); if (m) { await m.click(); await page.waitForTimeout(300); } }
  for (const t of TOOLS) {
    const btn = await page.$(`[data-tool="${t}"]`);
    if (!btn || !(await btn.isVisible())) { problems.push(`${tag}: no ${t} button`); continue; }
    await btn.click();
    await page.waitForTimeout(450);
    const r = await page.evaluate((sel) => {
      const vis = [...document.querySelectorAll(sel + ", .bm-pop")].filter((e) => e.offsetParent || getComputedStyle(e).position === "fixed");
      const tb = document.querySelector(".map-toolbar")?.getBoundingClientRect();
      const out = [];
      for (const e of vis) {
        const b = e.getBoundingClientRect(), name = e.className.split(" ")[0];
        if (b.right > innerWidth + 1 || b.bottom > innerHeight + 1 || b.left < -1 || b.top < -1) out.push(`${name} leaves the viewport`);
        if (tb && innerWidth >= 700 && b.right > tb.left + 1 && b.left < tb.right && b.bottom > tb.top && b.top < tb.bottom) out.push(`${name} covers the toolbar`);
      }
      return { n: vis.length + (document.querySelector(".set-window") ? 1 : 0), out };
    }, PANELS);
    if (r.n > 1) problems.push(`${tag} after ${t}: ${r.n} panels open`);
    r.out.forEach((o) => problems.push(`${tag} after ${t}: ${o}`));
    if (t === "settings" || t === "basemap") { await page.keyboard.press("Escape"); await page.waitForTimeout(200); }
  }
  await page.screenshot({ path: `ui-audit/${w}x${h}-${theme}.png` });
  await page.close();
}
await browser.close();
console.log(problems.length ? problems.join("\n") : `ok: ${SIZES.length * 2 * TOOLS.length} panel switches, no overlap`);
process.exit(problems.length ? 1 : 0);
