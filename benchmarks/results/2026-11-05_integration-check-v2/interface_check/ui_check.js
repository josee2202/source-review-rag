// Drives the real Streamlit UI in headless Chrome with a separate profile.
// Usage: PUPPETEER_PATH=<puppeteer-core> node ui_check.js <input.json> <outdir> <profiledir>
const puppeteer = require(process.env.PUPPETEER_PATH);
const fs = require('fs');
const path = require('path');

const [input, out, profile] = process.argv.slice(2);
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

const checkboxState = page => page.evaluate(() =>
  [...document.querySelectorAll('label')].find(l => l.innerText.includes('Añadir análisis del modelo')).querySelector('input').checked);

(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', userDataDir: profile, args: ['--no-first-run', '--no-default-browser-check'] });
  const report = { errors: [] };
  try {
    const page = await browser.newPage();
    page.on('pageerror', e => report.errors.push(String(e)));
    await page.setViewport({ width: 1360, height: 1700, deviceScaleFactor: 1.5 });
    const cdp = await page.createCDPSession();
    await cdp.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: path.resolve(out) });
    await page.goto('http://localhost:8511', { waitUntil: 'networkidle2', timeout: 60000 });
    await page.waitForSelector('textarea', { timeout: 60000 });
    report.analysis_checkbox_default = await checkboxState(page);
    await setArea(page, 0, reference);
    await setArea(page, 1, reviewed);
    await clickText(page, 'label', 'Añadir análisis del modelo');
    await wait(1500);
    report.analysis_checkbox_after_click = await checkboxState(page);
    const started = Date.now();
    await clickText(page, 'button', 'Revisar');
    await page.waitForFunction(() => document.body.innerText.includes('Descargar resultados en Excel'), { timeout: 15 * 60 * 1000, polling: 2000 });
    report.ui_wait_s = (Date.now() - started) / 1000;
    await wait(2000);
    const titles = await page.evaluate(() => [...document.querySelectorAll('[data-testid="stExpander"] summary')].map(s => s.innerText).filter(t => t.startsWith('Fragmento')));
    report.expander_titles = titles;
    // Opens every fragment, to read what a person sees
    for (const title of titles.slice(1)) await clickText(page, '[data-testid="stExpander"] summary', title);
    await wait(1500);
    report.page_text = await page.evaluate(() => document.querySelector('[data-testid="stMain"]').innerText);
    report.fragments_text = await page.evaluate(() => [...document.querySelectorAll('[data-testid="stExpander"]')]
      .filter(e => e.querySelector('summary').innerText.startsWith('Fragmento')).map(e => e.innerText));
    await clickText(page, 'button', 'Descargar resultados en Excel');
    await wait(4000);
    // Clean capture: only the fragment with the contradiction, open
    for (const title of titles) if (!title.startsWith('Fragmento 2')) await clickText(page, '[data-testid="stExpander"] summary', title);
    await wait(1200);
    const main = await page.$('[data-testid="stMainBlockContainer"]');
    await main.screenshot({ path: path.join(out, 'review-result.png') });
  } finally {
    fs.writeFileSync(path.join(out, 'ui_report.json'), JSON.stringify(report, null, 2));
    await browser.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
