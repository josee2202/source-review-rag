// Second pass over the real UI. Usage: PUPPETEER_PATH=<puppeteer-core> node ui_capture.js <input.json> <outdir> <profiledir> <fragment to leave open>
// Leaves one fragment open, reads the page, downloads the Excel file and saves a screenshot of the whole page.
const puppeteer = require(process.env.PUPPETEER_PATH);
const fs = require('fs');
const path = require('path');

const [input, out, profile, openFragment] = process.argv.slice(2);
const { reference, reviewed } = JSON.parse(fs.readFileSync(input, 'utf8'));
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const wait = ms => new Promise(r => setTimeout(r, ms));

async function setArea(page, index, value) {
  const areas = await page.$$('textarea');
  await areas[index].click();
  await page.evaluate((el, v) => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, areas[index], value);
  await page.keyboard.down('Control');
  await page.keyboard.press('Enter');
  await page.keyboard.up('Control');
  await wait(1500);
}

async function clickText(page, selector, text) {
  const ok = await page.evaluate((selector, text) => {
    const el = [...document.querySelectorAll(selector)].find(e => e.innerText && e.innerText.includes(text));
    if (!el) return false;
    el.click();
    return true;
  }, selector, text);
  if (!ok) throw new Error(`not found: ${selector} ${text}`);
  await wait(800);
}

// Opens or closes each fragment according to its real state, and only then moves on
async function leaveOpen(page, wanted) {
  for (let attempt = 0; attempt < 6; attempt++) {
    const changed = await page.evaluate(wanted => {
      const details = [...document.querySelectorAll('[data-testid="stExpander"] details')]
        .filter(d => d.querySelector('summary').innerText.startsWith('Fragmento'));
      for (const d of details) {
        const shouldOpen = d.querySelector('summary').innerText.startsWith(`Fragmento ${wanted} `);
        if (d.open !== shouldOpen) { d.querySelector('summary').click(); return true; }
      }
      return false;
    }, wanted);
    if (!changed) return;
    await wait(1000);
  }
  throw new Error('could not set the state of the fragments');
}

(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', userDataDir: profile, args: ['--no-first-run', '--no-default-browser-check'] });
  const report = { errors: [] };
  try {
    const page = await browser.newPage();
    page.on('pageerror', e => report.errors.push(String(e)));
    await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
    await page.setViewport({ width: 1240, height: 1400, deviceScaleFactor: 1.5 });
    const cdp = await page.createCDPSession();
    await cdp.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: path.resolve(out) });
    await page.goto('http://localhost:8511', { waitUntil: 'networkidle2', timeout: 60000 });
    await page.waitForSelector('textarea', { timeout: 60000 });
    await setArea(page, 0, reference);
    await setArea(page, 1, reviewed);
    await clickText(page, 'label', 'Añadir análisis del modelo');
    await wait(1500);
    const started = Date.now();
    await clickText(page, 'button', 'Revisar');
    await page.waitForFunction(() => document.body.innerText.includes('Descargar resultados en Excel'), { timeout: 15 * 60 * 1000, polling: 2000 });
    report.ui_wait_s = (Date.now() - started) / 1000;
    await wait(2000);
    await clickText(page, 'button', 'Descargar resultados en Excel');
    await wait(4000);
    await leaveOpen(page, openFragment);
    await wait(1000);
    report.page_text = await page.evaluate(() => document.querySelector('[data-testid="stMain"]').innerText);
    report.titles = await page.evaluate(() => [...document.querySelectorAll('[data-testid="stExpander"] summary')].map(s => s.innerText));
    report.alerts = await page.evaluate(() => [...document.querySelectorAll('[data-testid="stAlert"]')].map(a => a.innerText));
    const height = await page.evaluate(() => document.querySelector('[data-testid="stMainBlockContainer"]').scrollHeight);
    await page.setViewport({ width: 1240, height: height + 20, deviceScaleFactor: 1.5 });
    await wait(1500);
    await page.screenshot({ path: path.join(out, 'page.png') });
  } finally {
    fs.writeFileSync(path.join(out, 'capture_report.json'), JSON.stringify(report, null, 2));
    await browser.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
