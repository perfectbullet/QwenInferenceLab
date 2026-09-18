import { chromium } from 'playwright';
import assert from 'node:assert/strict';

// Backend fixture lives outside the page, so it survives real browser reloads.
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', error => errors.push(error.message));
let active = null;
const runs = [];
let starts = 0;
const questions = ['MATH-225', 'MATH-226'].map(id => ({ id, question: '$1+1$', reference_answer: '$2$' }));
const model = { id: 'fixture', baseUrl: 'http://fixture/v1', modelName: 'fixture' };
await page.route('**/api/**', async route => {
  const request = route.request();
  const path = new URL(request.url()).pathname;
  let body;
  if (path === '/api/questions') body = questions;
  else if (path === '/api/models') body = { models: [model], selectedId: model.id };
  else if (path === '/api/active-run') body = active;
  else if (path === '/api/runs') body = runs;
  else if (path === '/api/run') {
    starts++;
    active = { id: `run-${starts}`, questionId: request.postDataJSON().questionId, startedAt: new Date().toISOString(), model: 'fixture', params: {}, answer: '', reasoning: '持续思考', status: 'running', evaluation: 'unreviewed', notes: '', thinkingMs: null, totalMs: 0 };
    // An ended subscriber stream must fall back to progress queries, not restart.
    return route.fulfill({ contentType: 'text/event-stream', body: `data: ${JSON.stringify({ type: 'started', run: active })}\n\n` });
  } else if (path.endsWith('/cancel')) {
    assert.ok(active);
    runs.unshift({ ...active, status: 'cancelled' }); active = null; body = { ok: true };
  } else throw new Error(`Unexpected API: ${path}`);
  await route.fulfill({ json: body });
});
try {
  await page.goto('http://127.0.0.1:5173');
  await page.locator('.question-item').filter({ hasText: 'MATH-226' }).click();
  await page.getByRole('button', { name: '开始测试 →' }).click();
  await page.getByRole('status').waitFor();
  const id = active.id;
  await page.reload();
  await page.getByRole('status').waitFor();
  assert.match(await page.locator('.question-item.selected').innerText(), /MATH-226/);
  assert.equal(active.id, id);
  assert.equal(starts, 1);
  assert.ok(await page.getByRole('button', { name: '停止生成', exact: true }).isEnabled());
  assert.equal(await page.locator('.reasoning').getAttribute('open'), null);
  await page.locator('.question-item').filter({ hasText: 'MATH-225' }).click();
  assert.ok(await page.getByRole('button', { name: '开始测试 →' }).isDisabled());
  assert.ok(await page.getByRole('button', { name: '停止生成', exact: true }).isDisabled());
  await page.locator('.question-item').filter({ hasText: 'MATH-226' }).click();
  active.answer = '答案为 $2$';
  await page.locator('.answer').getByText('答案为', { exact: false }).waitFor();
  await page.reload();
  await page.locator('.answer').getByText('答案为', { exact: false }).waitFor();
  runs.unshift({ ...active, status: 'completed', totalMs: 1000, thinkingMs: 500 }); active = null;
  await page.waitForFunction(() => !document.querySelector('.primary').disabled);
  await page.locator('.answer .badge').getByText('已完成', { exact: true }).waitFor();
  await page.getByRole('button', { name: '开始测试 →' }).click();
  await page.getByRole('status').waitFor();
  await page.reload();
  await page.getByRole('status').waitFor();
  await page.getByRole('button', { name: '停止生成', exact: true }).click();
  await page.waitForFunction(() => !document.querySelector('.primary').disabled);
  assert.equal(runs[0].status, 'cancelled');
  assert.equal(starts, 2);
  assert.deepEqual(errors, []);
  console.log('PASS: reload during thinking/answer, same run, selection, completion, scoped controls, explicit cancellation');
} finally { await browser.close(); }
