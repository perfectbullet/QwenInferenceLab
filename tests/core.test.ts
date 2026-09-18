import test from 'node:test';
import assert from 'node:assert/strict';
import { readSSE } from '../shared/sse.js';
import { normalizeMath, splitPendingMath } from '../web/src/math.js';
import { infer } from '../server/src/inference.js';
import { defaults, type Run } from '../shared/types.js';
function stream(text: string) {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({ start(c) { for (const b of bytes) c.enqueue(Uint8Array.of(b)); c.close(); } });
}
function run(): Run { return { id: crypto.randomUUID(), questionId:'test', question:'1+1', startedAt:new Date().toISOString(), model:'mock', systemPrompt:'', params:defaults, request:{}, answer:'', status:'running', finishReason:null, firstResponseMs:null, thinkingMs:null, totalMs:0, thinkingNote:'', usage:null, evaluation:'unreviewed', notes:'' }; }
test('SSE handles byte-split Chinese, CRLF, comments, multi-line and trailing events', async () => {
  const result = [];
  for await (const event of readSSE(stream(': ping\r\ndata: 中文\r\ndata: 公式\r\n\r\ndata: [DONE]\n\n'))) result.push(event);
  assert.deepEqual(result, ['中文\n公式', '[DONE]']);
});
test('math normalization preserves code and escaped delimiters', () => {
  assert.equal(normalizeMath(String.raw`这是 \(x^2\) 和 \[\boxed{2}\]`), '这是 $x^2$ 和 \n\n$$\n\\boxed{2}\n$$\n\n');
  const source = '`\\(code\\)`\n```tex\n\\[x\\]\n```\n\\\\(literal\\\\)';
  assert.equal(normalizeMath(source), source);
  assert.equal(normalizeMath(String.raw`$x$ $$y$$ \(未闭合`), String.raw`$x$ $$y$$ \(未闭合`);
});
test('unfinished streamed math stays literal until closing delimiter arrives', () => {
  assert.deepEqual(splitPendingMath('正文\n$$\n\\frac{1}'), ['正文\n', '$$\n\\frac{1}']);
  assert.deepEqual(splitPendingMath('正文 $x'), ['正文 ', '$x']);
  assert.deepEqual(splitPendingMath('`$code` 和 $x$'), ['`$code` 和 $x$', '']);
  assert.deepEqual(splitPendingMath(normalizeMath('\\(x')), ['', '\\(x']);
  assert.deepEqual(splitPendingMath(normalizeMath('\\(x\\)')), ['$x$', '']);
});
test('inference separates reasoning from answer, accepts same-chunk answer and trailing usage', async () => {
  const original = globalThis.fetch;
  const chunks = [
    {choices:[{delta:{reasoning:'SECRET_THOUGHT'}}]},
    {choices:[{delta:{reasoning:'MORE_SECRET',content:'$2$'},finish_reason:'stop'}]},
    {choices:[],usage:{completion_tokens:9}},
  ];
  globalThis.fetch = async () => new Response(stream(chunks.map(c => `data: ${JSON.stringify(c)}\n\n`).join('') + 'data: [DONE]\n\n'));
  try {
    const r = run(); const emitted: unknown[] = [];
    await infer(r, 'http://test', new AbortController().signal, e => emitted.push(e), async () => {});
    assert.equal(r.status, 'completed'); assert.equal(r.answer, '$2$');
    assert.equal(r.usage?.completion_tokens, 9); assert.notEqual(r.thinkingMs, null);
    assert.equal(r.reasoning, 'SECRET_THOUGHTMORE_SECRET');
    assert.ok(!r.answer.includes('SECRET'));
    assert.equal(emitted.filter(e => (e as {type:string}).type === 'reasoning_delta').length, 2);
  } finally { globalThis.fetch = original; }
});
test('truncation and interrupted stream are distinct from success', async () => {
  const original = globalThis.fetch;
  try {
    for (const [ending, expected] of [['data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n', 'truncated'], ['', 'failed']] as const) {
      globalThis.fetch = async () => new Response(stream('data: {"choices":[{"delta":{"reasoning":"hidden"}}]}\n\n' + ending));
      const r = run(); await infer(r, 'http://test', new AbortController().signal, () => {}, async () => {});
      assert.equal(r.status, expected); assert.equal(r.thinkingMs, null);
    }
  } finally { globalThis.fetch = original; }
});
test('cancel and timeout have different outcomes', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => { throw new Error('aborted'); };
  try {
    for (const reason of ['cancelled', 'timeout']) {
      const controller = new AbortController(); controller.abort(reason);
      const r = run(); await infer(r, 'http://test', controller.signal, () => {}, async () => {});
      assert.equal(r.status, reason === 'cancelled' ? 'cancelled' : 'failed');
    }
  } finally { globalThis.fetch = original; }
});
