import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1080 } });
const errors = [];
page.on('pageerror', e => errors.push(e.message));
try {
  await page.goto('http://127.0.0.1:5173');
  await page.getByText('250 题', { exact: true }).waitFor();
  assert.ok(await page.locator('.markdown .katex').count() > 0, 'question formulas render');
  await page.locator('.reference > summary').click();
  assert.ok(await page.locator('.reference .katex').count() > 0, 'reference formulas render');
  const health = await (await page.request.get('http://127.0.0.1:3100/api/health')).json();
  assert.equal(health.storage, 'mongodb');
  for (const section of await page.locator('.source-image').all()) {
    await section.locator('summary').click();
    await section.locator('img').scrollIntoViewIfNeeded();
    await page.waitForFunction(img => img.complete && img.naturalWidth > 0, await section.locator('img').elementHandle());
  }
  assert.equal(await page.locator('.source-image img').count(), 2);
  await mkdir('test-results', { recursive: true });
  await page.screenshot({ path: 'test-results/initial.png', fullPage: true });
  await page.screenshot({ path: 'test-results/mongodb-images.png', fullPage: true });
  console.log('Page loaded: 250 questions; question and reference formulas render.');
  if (process.env.LIVE_SMOKE === '1') {
    await page.getByLabel('搜索题目').fill('MATH-');
    const qs = await (await page.request.get('http://127.0.0.1:3100/api/questions')).json();
    const q = qs.sort((a,b) => a.question.length - b.question.length)[0];
    await page.getByLabel('搜索题目').fill(q.id);
    await page.locator('.question-item').filter({ hasText: q.id }).first().click();
    console.log(`Live CoT test: ${q.id}`);
    await page.getByRole('button', { name: '开始测试 →' }).click();
    await page.getByRole('status').filter({ hasText: '思考中' }).waitFor({ timeout: 60000 });
    console.log('Thinking indicator observed; awaiting formal answer.');
    await page.getByRole('button', { name: '开始测试 →' }).waitFor({ timeout: 900000 });
    const records = await (await page.request.get('http://127.0.0.1:3100/api/runs')).json();
    const r = records.find(r => r.questionId === q.id);
    console.log(JSON.stringify({ id:r.id, questionId:r.questionId, status:r.status, thinkingMs:r.thinkingMs, totalMs:r.totalMs, usage:r.usage }));
    assert.equal(r.status, 'completed'); assert.ok(r.thinkingMs !== null);
    assert.ok(r.modelBaseUrl?.startsWith('http'));
    assert.equal(r.modelName, r.request.model);
    assert.ok(r.modelConfigId);
    assert.ok(r.reasoning?.length > 0, 'reasoning is persisted separately');
    assert.equal(await page.locator('.reasoning').getAttribute('open'), null);
    await page.locator('.reasoning summary').click();
    assert.equal(await page.locator('.reasoning pre').textContent(), r.reasoning);
    assert.ok(await page.locator('.answer .katex').count() > 0);
    await page.locator('.review select').selectOption('review');
    await page.getByLabel('评价备注').fill('开发冒烟测试：CoT、公式与历史已验证；答案尚未人工判分。');
    await page.getByRole('button', { name:'保存评价' }).click();
    await page.getByRole('button', { name:'已保存 ✓' }).waitFor();
    await page.screenshot({ path:'test-results/live-answer.png', fullPage:true });
    await page.reload();
    await page.getByLabel('搜索题目').fill(q.id);
    await page.locator('.question-item').filter({ hasText:q.id }).first().click();
    await page.locator('.history button').first().click();
    assert.equal(await page.locator('.review select').inputValue(), 'review');
    assert.ok((await page.getByLabel('评价备注').inputValue()).includes('开发冒烟测试'));
    await page.getByRole('button', { name:'开始测试 →' }).click();
    await page.getByRole('button', { name:'停止生成', exact:true }).waitFor();
    await page.getByRole('button', { name:'停止生成', exact:true }).click();
    await page.getByRole('button', { name:'开始测试 →' }).waitFor({ timeout:30000 });
    assert.equal(await page.locator('.answer .badge').textContent(), '已取消');
    console.log('Review persisted across reload; cancellation verified.');
  }
  assert.deepEqual(errors, []);
  console.log('Browser smoke passed, no JavaScript errors.');
} finally { await browser.close(); }
