import 'dotenv/config';
import Fastify from 'fastify';
import fastifyStatic from '@fastify/static';
import { access } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defaults, type Params, type Run } from '../../shared/types.js';
import { infer, systemPrompt } from './inference.js';
import { QuestionStore, RunStore } from './store.js';
import { connectDatabase } from './database.js';
import { QuestionImages } from './images.js';
import { ModelStore, validateModel } from './models.js';
const root = fileURLToPath(new URL('../../', import.meta.url));
const { client, db } = await connectDatabase();
const store = new RunStore(db);
const questions = new QuestionStore(db);
const models = new ModelStore(db);
const initialBase = process.env.MODEL_BASE_URL || 'http://192.168.8.231:8200/v1';
const initialModel = process.env.MODEL_NAME || 'qwen38-27b';
const images = new QuestionImages(path.resolve(root, process.env.IMAGES_DIR || 'images'));
try {
  await store.initialize(); await questions.initialize();
  await models.initialize({ baseUrl: initialBase, modelName: initialModel });
  if (!await questions.count()) throw new Error('MongoDB 题库为空，请先运行 npm run db:migrate');
} catch {
  await client.close(); throw new Error('数据库初始化失败，请检查权限并先运行 npm run db:migrate');
}
const app = Fastify({ logger: true });
let active: { run: Run; controller: AbortController } | null = null;
app.setErrorHandler((error, request, reply) => {
  const info = error as { code?: string; statusCode?: number };
  request.log.error({ code: info.code }, '请求处理失败');
  reply.code(info.statusCode && info.statusCode < 500 ? info.statusCode : 503).send({ error: '请求失败，请检查请求格式或数据库连接' });
});
app.get('/api/health', async () => ({ ok: true, storage: 'mongodb', count: await questions.count(), activeRunId: active?.run.id || null }));
app.get('/api/models', async () => models.list());
app.post('/api/models', async (req, reply) => {
  let input;
  try { input = validateModel(req.body); } catch (e) { return reply.code(400).send({ error: (e as Error).message }); }
  try { return await models.save(input); } catch (e) {
    if ((e as {code?:number}).code === 11000) return reply.code(409).send({ error: '该 URL 和模型名称已经存在' });
    throw e;
  }
});
app.patch('/api/models/:id', async (req, reply) => {
  const { id } = req.params as { id: string };
  if (!await models.get(id)) return reply.code(404).send({ error: '配置不存在' });
  let input;
  try { input = validateModel(req.body); } catch (e) { return reply.code(400).send({ error: (e as Error).message }); }
  try { return await models.save(input, id); } catch (e) {
    if ((e as {code?:number}).code === 11000) return reply.code(409).send({ error: '该 URL 和模型名称已经存在' });
    throw e;
  }
});
app.post('/api/models/selection', async (req, reply) => {
  const id = (req.body as { id?: unknown })?.id;
  if (typeof id !== 'string' || !await models.get(id)) return reply.code(400).send({ error: '请选择有效模型配置' });
  await models.select(id); return { ok: true };
});
app.get('/api/questions', async () => Promise.all((await questions.list()).map(q => images.decorate(q))));
app.get('/api/questions/:id/images/:kind', async (req, reply) => {
  const { id, kind } = req.params as { id: string; kind: string };
  if (!['question', 'answer'].includes(kind)) return reply.code(404).send({ error: '图片不存在' });
  const q = await questions.get(id);
  const file = await images.resolve(kind === 'question' ? q?.image_path : q?.reference_answer_image_path);
  if (!file) return reply.code(404).send({ error: '原图未找到，请检查图片目录' });
  const mime: Record<string, string> = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp', '.gif': 'image/gif' };
  return reply.type(mime[path.extname(file).toLowerCase()]).header('X-Content-Type-Options', 'nosniff').header('Cache-Control', 'private, max-age=300').send(createReadStream(file));
});
app.get('/api/runs', async () => store.list());
app.get('/api/active-run', async () => active?.run || null);
app.post('/api/runs/:id/cancel', async (req, reply) => {
  const { id } = req.params as { id: string };
  if (active?.run.id !== id) return reply.code(404).send({ error: '运行已结束或不存在' });
  active.controller.abort('cancelled'); return { ok: true };
});
app.patch('/api/runs/:id', async (req, reply) => {
  const { id } = req.params as { id: string };
  if (active?.run.id === id) return reply.code(409).send({ error: '请在运行结束后评价' });
  const run = await store.get(id);
  if (!run) return reply.code(404).send({ error: '记录不存在' });
  const body = req.body as { evaluation?: Run['evaluation']; notes?: string };
  if (!body || !['unreviewed', 'correct', 'incorrect', 'review'].includes(body.evaluation || '') || typeof body.notes !== 'string' || body.notes.length > 10000) return reply.code(400).send({ error: '评价或备注格式错误' });
  const updated = await store.review(id, body.evaluation!, body.notes);
  return updated || reply.code(409).send({ error: '运行进行中，暂不能评价' });
});
app.post('/api/run', async (req, reply) => {
  if (active) return reply.code(409).send({ error: '已有运行进行中，请等待或停止' });
  const body = req.body as { questionId?: string; params?: Partial<Params>; modelConfigId?: string };
  const q = typeof body?.questionId === 'string' ? await questions.get(body.questionId) : null;
  if (!q) return reply.code(400).send({ error: '请选择有效题目' });
  const modelConfig = typeof body.modelConfigId === 'string' ? await models.get(body.modelConfigId) : null;
  if (!modelConfig) return reply.code(400).send({ error: '请选择有效模型配置' });
  const model = modelConfig.modelName;
  const base = modelConfig.baseUrl;
  if (active) return reply.code(409).send({ error: '已有运行进行中，请等待或停止' });
  const params: Params = { temperature: body.params?.temperature ?? defaults.temperature, top_p: body.params?.top_p ?? defaults.top_p, top_k: body.params?.top_k ?? defaults.top_k, max_tokens: body.params?.max_tokens ?? defaults.max_tokens };
  if (!Object.values(params).every(v => typeof v === 'number' && Number.isFinite(v)) || params.temperature < 0 || params.temperature > 2 || params.top_p <= 0 || params.top_p > 1 || !Number.isInteger(params.top_k) || params.top_k < 1 || params.top_k > 100 || !Number.isInteger(params.max_tokens) || params.max_tokens < 64 || params.max_tokens > 30000) return reply.code(400).send({ error: '参数范围错误：temperature 0–2，top_p (0,1]，top_k 1–100，max_tokens 64–30000' });
  const run: Run = {
    id: crypto.randomUUID(), questionId: q.id, question: q.question, startedAt: new Date().toISOString(),
    modelBaseUrl: base, modelName: model, modelConfigId: modelConfig.id,
    model, systemPrompt, params, request: { model, messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: q.question }], ...params, stream: true, stream_options: { include_usage: true }, chat_template_kwargs: { enable_thinking: true } },
    answer: '', status: 'running', finishReason: null, firstResponseMs: null, thinkingMs: null, totalMs: 0, thinkingNote: '运行尚未结束', usage: null, evaluation: 'unreviewed', notes: '',
  };
  const controller = new AbortController(); active = { run, controller };
  try { await store.save(run); } catch (e) { active = null; throw e; }
  reply.hijack(); reply.raw.writeHead(200, { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no' });
  const emit = (event: Record<string, unknown>) => { if (!reply.raw.destroyed) reply.raw.write(`data: ${JSON.stringify(event)}\n\n`); };
  emit({ type: 'started', run });
  const close = () => { if (!reply.raw.writableEnded) controller.abort('cancelled'); };
  reply.raw.on('close', close);
  const heartbeat = setInterval(() => { if (!reply.raw.destroyed) reply.raw.write(': heartbeat\n\n'); }, 10000);
  const timeout = setTimeout(() => controller.abort('timeout'), Number(process.env.MODEL_TIMEOUT_MS || 900000));
  try {
    await infer(run, base, controller.signal, emit, async () => {
      try { await store.save(run); } catch { throw new Error('数据库检查点保存失败'); }
    }, base === initialBase.replace(/\/+$/, '') ? process.env.MODEL_API_KEY || '' : '');
    await store.save(run);
    emit({ type: 'done', run });
  } catch { emit({ type: 'error', error: '运行记录保存失败，请检查数据库连接' }); }
  finally { clearInterval(heartbeat); clearTimeout(timeout); active = null; reply.raw.off('close', close); reply.raw.end(); }
});
const dist = path.join(root, 'dist');
try {
  await access(path.join(dist, 'index.html'));
  await app.register(fastifyStatic, { root: dist });
} catch { /* Vite serves the development UI. */ }
let ready = false;
app.addHook('onRequest', async (_req, reply) => { if (!ready) return reply.code(503).send({ error: '服务正在初始化' }); });
app.addHook('onClose', async () => { await client.close(); });
try {
  await app.listen({ host: process.env.HOST || '127.0.0.1', port: Number(process.env.PORT || 3100) });
  await store.recover(); ready = true;
} catch { await app.close(); throw new Error('启动失败，请检查端口占用或数据库连接'); }
for (const signal of ['SIGINT', 'SIGTERM'] as const) process.once(signal, async () => {
  active?.controller.abort('cancelled'); await app.close(); process.exit(0);
});
