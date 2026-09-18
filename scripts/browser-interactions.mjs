import { chromium } from 'playwright';
import assert from 'node:assert/strict';
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ permissions: ['clipboard-read','clipboard-write'] });
const page = await context.newPage();
const errors = []; page.on('pageerror', e => errors.push(e.message));
// Deterministic streaming fixture: no model calls or database writes.
await page.addInitScript(() => {
  const original = window.fetch.bind(window);
  const questions = [
    { id:'TEST-A', question:'题目 A：$x+1=2$', reference_answer:'答案 A：$x=1$' },
    { id:'TEST-B', question:'题目 B：$y+2=4$', reference_answer:'答案 B：$y=2$' },
  ];
  const history = []; let active = null; let controller; let starts = 0;
  const models = [{id:'first',baseUrl:'http://model-a/v1',modelName:'model-a'},{id:'second',baseUrl:'http://model-b/v1',modelName:'model-b'}];
  let selectedId = 'first';
  const emit = e => controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(e)}\n\n`));
  window.testStream = {
    finish(status = 'completed') {
      active = { ...active, status, totalMs:1000, thinkingMs:500, finishReason:status === 'completed' ? 'stop' : null };
      emit({type:'done',run:active}); history.unshift(active); active = null; controller.close();
    },
    answer(text) { active.answer += text; emit({type:'answer_delta',text}); },
    starts: () => starts,
  };
  window.fetch = async (url, init) => {
    const p = new URL(String(url), location.origin).pathname;
    const json = v => new Response(JSON.stringify(v),{headers:{'Content-Type':'application/json'}});
    if (p === '/api/questions') return json(questions);
    if (p === '/api/models' && (!init || init.method === 'GET')) return json({models,selectedId});
    if (p === '/api/models' && init.method === 'POST') { const saved = {id:'added',...JSON.parse(init.body)}; models.push(saved); return json(saved); }
    if (p.startsWith('/api/models/') && init.method === 'PATCH') { const id = p.split('/').pop(); const model = models.find(m=>m.id===id); Object.assign(model,JSON.parse(init.body)); return json(model); }
    if (p === '/api/models/selection') { selectedId = JSON.parse(init.body).id; return json({ok:true}); }
    if (p === '/api/health') return json({model:'fixture',activeRunId:active?.id});
    if (p === '/api/active-run') return json(active);
    if (p === '/api/runs') return json(history);
    if (p.endsWith('/cancel')) { window.testStream.finish('cancelled'); return json({ok:true}); }
    if (p === '/api/run') {
      starts++; const {questionId,modelConfigId}=JSON.parse(init.body);
      if (modelConfigId !== selectedId) throw new Error('wrong selected model');
      active = {id:`run-${starts}`,questionId,question:questions.find(q=>q.id===questionId).question,startedAt:new Date().toISOString(),model:'fixture',params:{},answer:'',reasoning:'推理片段 A',status:'running',evaluation:'unreviewed',notes:'',thinkingMs:null,firstResponseMs:1,totalMs:0};
      return new Response(new ReadableStream({start(c) {
        controller=c; emit({type:'started',run:{...active,reasoning:''}});
        emit({type:'status',phase:'thinking'}); emit({type:'reasoning_delta',text:active.reasoning});
      }}),{headers:{'Content-Type':'text/event-stream'}});
    }
    return original(url,init);
  };
});
try {
  await page.goto('http://127.0.0.1:5173');
  await page.getByText('2 题',{exact:true}).waitFor();
  await page.getByLabel('模型 URL',{exact:true}).selectOption('http://model-b/v1');
  await page.waitForFunction(()=>document.querySelector('select[aria-label="模型名称"]').value==='second');
  await page.getByRole('button',{name:'复制题目 Markdown'}).click();
  assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),'题目 A：$x+1=2$');
  await page.locator('.reference > summary').click();
  await page.getByRole('button',{name:'复制参考答案 Markdown'}).click();
  assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),'答案 A：$x=1$');
  await page.getByRole('button',{name:'开始测试 →'}).click();
  await page.getByRole('status').waitFor();
  assert.equal(await page.locator('.reasoning').getAttribute('open'),null);
  assert.equal(await page.locator('.reasoning pre').count(),0);
  await page.locator('.reasoning summary').click();
  await page.getByText('推理片段 A',{exact:true}).waitFor();
  await page.locator('.question-item').filter({hasText:'TEST-B'}).click();
  assert.ok(await page.getByRole('button',{name:'开始测试 →'}).isDisabled());
  assert.ok(await page.getByRole('button',{name:'停止生成',exact:true}).isDisabled());
  assert.equal(await page.locator('.answer .thinking').count(),0);
  assert.equal(await page.locator('.answer').getByText('推理片段 A',{exact:true}).count(),0);
  await page.evaluate(()=>window.testStream.answer('**A 的答案**：$x=1$'));
  assert.equal(await page.locator('.answer').getByText('A 的答案',{exact:true}).count(),0);
  await page.locator('.question-item').filter({hasText:'TEST-A'}).click();
  assert.ok(await page.getByRole('button',{name:'停止生成',exact:true}).isEnabled());
  await page.getByRole('button',{name:'复制模型解答 Markdown'}).click();
  assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),'**A 的答案**：$x=1$');
  await page.locator('.question-item').filter({hasText:'TEST-B'}).click();
  await page.evaluate(()=>window.testStream.finish());
  await page.waitForFunction(()=>!document.querySelector('.primary').disabled);
  assert.equal(await page.locator('.answer').getByText('A 的答案',{exact:true}).count(),0);
  await page.locator('.question-item').filter({hasText:'TEST-A'}).click();
  await page.locator('.answer').getByText('A 的答案',{exact:true}).waitFor();
  await page.getByRole('button',{name:'开始测试 →'}).click();
  await page.getByRole('button',{name:'停止生成',exact:true}).click();
  await page.waitForFunction(()=>!document.querySelector('.primary').disabled);
  assert.equal(await page.locator('.answer .badge').textContent(),'已取消');
  assert.equal(await page.evaluate(()=>window.testStream.starts()),2);
  await page.getByText('管理模型配置',{exact:true}).click();
  await page.getByLabel('配置模型 URL',{exact:true}).fill('http://model-c/v1');
  await page.getByLabel('配置模型名称',{exact:true}).fill('model-c');
  await page.getByRole('button',{name:'保存并选用',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('select[aria-label="模型名称"]').value==='added');
  assert.equal(await page.getByLabel('模型 URL',{exact:true}).inputValue(),'http://model-c/v1');
  await page.getByRole('button',{name:'编辑当前',exact:true}).click();
  await page.getByLabel('配置模型名称',{exact:true}).fill('model-c-edited');
  await page.getByRole('button',{name:'保存修改并选用',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('select[aria-label="模型名称"]').selectedOptions[0]?.text==='model-c-edited');
  assert.deepEqual(errors,[]);
  console.log('Passed: model URL/name selection, add/edit; switch A/B during stream; single active task; scoped stop; all three raw Markdown copies; collapsed/expanded reasoning; completion on another question; cancellation.');
} finally {await browser.close();}
