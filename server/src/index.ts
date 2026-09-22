import 'dotenv/config';
import Fastify from 'fastify';
import fastifyStatic from '@fastify/static';
import swagger from '@fastify/swagger';
import swaggerUi from '@fastify/swagger-ui';
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
import { runStream } from './run-stream.js';
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
await app.register(swagger, {
  openapi: {
    info: { title: '数学测试台 API', version: '1.0.0', description: '本机数学模型测试、模型配置与运行历史接口。API Key 不会通过任何读取接口返回。' },
    servers: [{ url: 'http://127.0.0.1:3100', description: '本机服务' }],
    tags: [{ name: '系统' }, { name: '模型配置' }, { name: '题库' }, { name: '运行记录' }, { name: '测试' }],
  },
});
await app.register(swaggerUi, { routePrefix: '/docs', uiConfig: { docExpansion: 'list', deepLinking: false } });
const configuredConcurrency = Number(process.env.MAX_CONCURRENT_RUNS || 4);
if (!Number.isInteger(configuredConcurrency) || configuredConcurrency < 1 || configuredConcurrency > 64) throw new Error('MAX_CONCURRENT_RUNS 必须是 1–64 的整数');
const active = new Map<string, { run: Run; controller: AbortController }>();
app.setErrorHandler((error, request, reply) => {
  const info = error as { code?: string; statusCode?: number };
  request.log.error({ code: info.code }, '请求处理失败');
  reply.code(info.statusCode && info.statusCode < 500 ? info.statusCode : 503).send({ error: '请求失败，请检查请求格式或数据库连接' });
});
app.get('/api/health', { schema: { tags: ['系统'], summary: '获取服务与题库状态' } }, async () => ({ ok: true, storage: 'mongodb', count: await questions.count(), activeRunId: active.values().next().value?.run.id || null, activeRunCount: active.size, maxConcurrentRuns: configuredConcurrency }));
app.get('/api/models', { schema: { tags: ['模型配置'], summary: '列出模型配置及当前选中项', description: '结果只含 hasApiKey 标识，不返回 API Key 明文。' } }, async () => models.list());
const modelBodySchema = { type: 'object', required: ['baseUrl', 'modelName'], properties: { baseUrl: { type: 'string', format: 'uri', description: 'OpenAI 兼容 API 基础地址，例如 https://api.siliconflow.cn/v1' }, modelName: { type: 'string', description: '服务端模型名称' }, apiKey: { type: 'string', writeOnly: true, description: '可选；仅由后端保存，永不返回' } } } as const;
app.post('/api/models', { schema: { tags: ['模型配置'], summary: '新增模型配置', body: modelBodySchema } }, async (req, reply) => {
  let input;
  try { input = validateModel(req.body); } catch (e) { return reply.code(400).send({ error: (e as Error).message }); }
  try { const saved = await models.save(input); const { apiKey: _apiKey, ...publicConfig } = saved; return { ...publicConfig, hasApiKey: Boolean(saved.apiKey) }; } catch (e) {
    if ((e as {code?:number}).code === 11000) return reply.code(409).send({ error: '该 URL 和模型名称已经存在' });
    throw e;
  }
});
app.patch('/api/models/:id', { schema: { tags: ['模型配置'], summary: '编辑模型配置', params: { type: 'object', required: ['id'], properties: { id: { type: 'string' } } }, body: modelBodySchema } }, async (req, reply) => {
  const { id } = req.params as { id: string };
  if (!await models.get(id)) return reply.code(404).send({ error: '配置不存在' });
  let input;
  try { input = validateModel(req.body); } catch (e) { return reply.code(400).send({ error: (e as Error).message }); }
  try { const saved = await models.save(input, id); const { apiKey: _apiKey, ...publicConfig } = saved; return { ...publicConfig, hasApiKey: Boolean(saved.apiKey) }; } catch (e) {
    if ((e as {code?:number}).code === 11000) return reply.code(409).send({ error: '该 URL 和模型名称已经存在' });
    throw e;
  }
});
app.post('/api/models/selection', { schema: { tags: ['模型配置'], summary: '选择当前模型', body: { type: 'object', required: ['id'], properties: { id: { type: 'string' } } } } }, async (req, reply) => {
  const id = (req.body as { id?: unknown })?.id;
  if (typeof id !== 'string' || !await models.get(id)) return reply.code(400).send({ error: '请选择有效模型配置' });
  await models.select(id); return { ok: true };
});
app.get('/api/questions', { schema: { tags: ['题库'], summary: '列出全部题目' } }, async () => Promise.all((await questions.list()).map(q => images.decorate(q))));
app.get('/api/questions/:id/images/:kind', { schema: { tags: ['题库'], summary: '获取题目或参考答案图片', params: { type: 'object', required: ['id', 'kind'], properties: { id: { type: 'string' }, kind: { type: 'string', enum: ['question', 'answer'] } } } } }, async (req, reply) => {
  const { id, kind } = req.params as { id: string; kind: string };
  if (!['question', 'answer'].includes(kind)) return reply.code(404).send({ error: '图片不存在' });
  const q = await questions.get(id);
  const file = await images.resolve(kind === 'question' ? q?.image_path : q?.reference_answer_image_path);
  if (!file) return reply.code(404).send({ error: '原图未找到，请检查图片目录' });
  const mime: Record<string, string> = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp', '.gif': 'image/gif' };
  return reply.type(mime[path.extname(file).toLowerCase()]).header('X-Content-Type-Options', 'nosniff').header('Cache-Control', 'private, max-age=300').send(createReadStream(file));
});
app.get('/api/runs', { schema: { tags: ['运行记录'], summary: '列出全部测试历史' } }, async () => store.list());
app.get('/api/active-run', { schema: { tags: ['运行记录'], summary: '获取一个当前运行任务；没有则返回 null' } }, async () => active.values().next().value?.run || null);
app.post('/api/runs/:id/cancel', { schema: { tags: ['运行记录'], summary: '取消当前运行任务', params: { type: 'object', required: ['id'], properties: { id: { type: 'string' } } } } }, async (req, reply) => {
  const { id } = req.params as { id: string };
  const task = active.get(id);
  if (!task) return reply.code(404).send({ error: '运行已结束或不存在' });
  task.controller.abort('cancelled'); return { ok: true };
});
app.patch('/api/runs/:id', { schema: { tags: ['运行记录'], summary: '保存人工评价与备注', params: { type: 'object', required: ['id'], properties: { id: { type: 'string' } } }, body: { type: 'object', required: ['evaluation', 'notes'], properties: { evaluation: { type: 'string', enum: ['unreviewed', 'correct', 'incorrect', 'review'] }, notes: { type: 'string', maxLength: 10000 } } } } }, async (req, reply) => {
  const { id } = req.params as { id: string };
  if (active.has(id)) return reply.code(409).send({ error: '请在运行结束后评价' });
  const run = await store.get(id);
  if (!run) return reply.code(404).send({ error: '记录不存在' });
  const body = req.body as { evaluation?: Run['evaluation']; notes?: string };
  if (!body || !['unreviewed', 'correct', 'incorrect', 'review'].includes(body.evaluation || '') || typeof body.notes !== 'string' || body.notes.length > 10000) return reply.code(400).send({ error: '评价或备注格式错误' });
  const updated = await store.review(id, body.evaluation!, body.notes);
  return updated || reply.code(409).send({ error: '运行进行中，暂不能评价' });
});
app.post('/api/run', { schema: { tags: ['测试'], summary: '启动单题流式测试', description: '响应为 Server-Sent Events（SSE），依次产生 started、status、reasoning_delta、answer_delta、done 或 error 事件。', body: { type: 'object', required: ['questionId', 'modelConfigId'], properties: { questionId: { type: 'string' }, modelConfigId: { type: 'string' }, params: { type: 'object', properties: { temperature: { type: 'number', minimum: 0, maximum: 2 }, top_p: { type: 'number', exclusiveMinimum: 0, maximum: 1 }, top_k: { type: 'integer', minimum: 1, maximum: 100 }, min_p: { type: 'number', minimum: 0, maximum: 1 }, presence_penalty: { type: 'number', minimum: -2, maximum: 2 }, repetition_penalty: { type: 'number', exclusiveMinimum: 0, maximum: 2 }, max_tokens: { type: 'integer', minimum: 64, maximum: 30000 } } } } } } }, async (req, reply) => {
  if (active.size >= configuredConcurrency) return reply.code(409).send({ error: '运行并发已满，请等待或停止' });
  const body = req.body as { questionId?: string; params?: Partial<Params>; modelConfigId?: string };
  const q = typeof body?.questionId === 'string' ? await questions.get(body.questionId) : null;
  if (!q) return reply.code(400).send({ error: '请选择有效题目' });
  const modelConfig = typeof body.modelConfigId === 'string' ? await models.get(body.modelConfigId) : null;
  if (!modelConfig) return reply.code(400).send({ error: '请选择有效模型配置' });
  const model = modelConfig.modelName;
  const base = modelConfig.baseUrl;
  if (active.size >= configuredConcurrency) return reply.code(409).send({ error: '运行并发已满，请等待或停止' });
  const params: Params = { temperature: body.params?.temperature ?? defaults.temperature, top_p: body.params?.top_p ?? defaults.top_p, top_k: body.params?.top_k ?? defaults.top_k, min_p: body.params?.min_p ?? defaults.min_p, presence_penalty: body.params?.presence_penalty ?? defaults.presence_penalty, repetition_penalty: body.params?.repetition_penalty ?? defaults.repetition_penalty, max_tokens: body.params?.max_tokens ?? defaults.max_tokens };
  if (!Object.values(params).every(v => typeof v === 'number' && Number.isFinite(v)) || params.temperature < 0 || params.temperature > 2 || params.top_p <= 0 || params.top_p > 1 || !Number.isInteger(params.top_k) || params.top_k < 1 || params.top_k > 100 || params.min_p < 0 || params.min_p > 1 || params.presence_penalty < -2 || params.presence_penalty > 2 || params.repetition_penalty <= 0 || params.repetition_penalty > 2 || !Number.isInteger(params.max_tokens) || params.max_tokens < 64 || params.max_tokens > 30000) return reply.code(400).send({ error: '参数范围错误：temperature 0–2，top_p (0,1]，top_k 1–100，min_p 0–1，presence_penalty -2–2，repetition_penalty (0,2]，max_tokens 64–30000' });
  const run: Run = {
    id: crypto.randomUUID(), questionId: q.id, question: q.question, startedAt: new Date().toISOString(),
    modelBaseUrl: base, modelName: model, modelConfigId: modelConfig.id,
    model, systemPrompt, params, request: { model, messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: q.question }], ...params, stream: true, stream_options: { include_usage: true }, chat_template_kwargs: { enable_thinking: true } },
    answer: '', status: 'running', finishReason: null, firstResponseMs: null, thinkingMs: null, totalMs: 0, thinkingNote: '运行尚未结束', usage: null, evaluation: 'unreviewed', notes: '',
  };
  const controller = new AbortController(); active.set(run.id, { run, controller });
  try { await store.save(run); } catch (e) { active.delete(run.id); throw e; }
  reply.hijack(); reply.raw.writeHead(200, { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no' });
  const stream = runStream(reply.raw);
  const emit = stream.emit;
  emit({ type: 'started', run });
  const timeout = setTimeout(() => controller.abort('timeout'), Number(process.env.MODEL_TIMEOUT_MS || 900000));
  try {
    await infer(run, base, controller.signal, emit, async () => {
      try { await store.save(run); } catch { throw new Error('数据库检查点保存失败'); }
    }, modelConfig.apiKey || (base === initialBase.replace(/\/+$/, '') ? process.env.MODEL_API_KEY || '' : ''));
    await store.save(run);
    emit({ type: 'done', run });
  } catch { emit({ type: 'error', error: '运行记录保存失败，请检查数据库连接' }); }
  finally { clearTimeout(timeout); active.delete(run.id); stream.end(); }
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
  for (const task of active.values()) task.controller.abort('cancelled'); await app.close(); process.exit(0);
});
