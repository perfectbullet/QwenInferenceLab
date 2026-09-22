import { mkdir, readFile, writeFile, rename, open, unlink } from 'node:fs/promises';
import path from 'node:path';
import { defaults, type Run, type ModelSettings, type Question } from '../shared/types.js';
import { readSSE } from '../shared/sse.js';

const dir = path.resolve(process.argv[2] || 'test-results/batch-current');
const endpoint = process.env.BATCH_API_URL || 'http://127.0.0.1:3100';
const concurrency = Number(process.env.BATCH_CONCURRENCY || 4);
if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > 64) throw new Error('BATCH_CONCURRENCY 必须是 1–64 的整数');
type BatchModel = { baseUrl: string; modelName: string };
type Pending = { questionId: string; since: string; runId?: string };
type State = { model: BatchModel; params: typeof defaults; questionIds: string[]; modelConfigId: string;
  startedAt: string; status: string; pending?: Pending[] | Pending;
  results: { questionId: string; runId: string; status: string; totalMs: number }[]; error?: string };
const pause = () => new Promise(resolve => setTimeout(resolve, 3000));
async function api<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(endpoint + url, { signal: AbortSignal.timeout(30000), ...(body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }) });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}
await mkdir(dir, { recursive: true });
const lockPath = path.join(dir, 'worker.lock');
const lock = await open(lockPath, 'wx');
await lock.writeFile(String(process.pid)); await lock.close();
let state: State | undefined;
let saveChain = Promise.resolve();
let recoveryPending = new Set<Pending>();
let fatal: Error | undefined;
async function save() {
  // Concurrent workers may finish together; serialize atomic state replacements.
  const snapshot = JSON.stringify(state, null, 2);
  saveChain = saveChain.catch(() => {}).then(async () => {
    await writeFile(path.join(dir, 'state.tmp'), snapshot);
    await rename(path.join(dir, 'state.tmp'), path.join(dir, 'state.json'));
  });
  return saveChain;
}
const log = (message: string) => console.log(`${new Date().toISOString()} ${message}`);
const pendingList = () => {
  if (!state) throw new Error('批次状态未初始化');
  if (!Array.isArray(state.pending)) state.pending = state.pending ? [state.pending] : [];
  return state.pending;
};

async function submitAndWait(pending: Pending) {
  const questionId = pending.questionId;
  if (!pending.runId && recoveryPending.has(pending)) {
    const records = await api<Run[]>('/api/runs');
    const matches = records.filter(run => run.questionId === questionId && run.startedAt >= pending.since && run.modelBaseUrl === state!.model.baseUrl && run.modelName === state!.model.modelName);
    if (matches.length > 1) throw new Error(`${questionId} 有多条可能的恢复记录；暂停以避免重复生成`);
    if (matches.length === 1) { pending.runId = matches[0].id; await save(); log(`RECOVER ${questionId} ${pending.runId}`); }
  }
  while (!pending.runId) {
    if (fatal) throw fatal;
    const response = await fetch(endpoint + '/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ questionId, params: state!.params, modelConfigId: state!.modelConfigId }) });
    if (response.status === 409) { await pause(); if (fatal) throw fatal; continue; } // Server pool is full; retain this claimed question and retry.
    if (!response.ok || !response.body) throw new Error(`${questionId} 提交失败 HTTP ${response.status}；请核对后再恢复`);
    try {
      for await (const data of readSSE(response.body)) {
        const event = JSON.parse(data);
        if (event.type === 'started') {
          pending.runId = event.run.id; await save(); log(`START ${questionId} ${event.run.id}`);
        }
        if (event.type === 'done') break;
        if (event.type === 'error') throw new Error(event.error);
      }
    } catch (error) {
      // A subscriber may disconnect after the server accepted the task. Polling resolves it by run id.
      if (pending.runId) log(`Subscriber disconnected for ${questionId}; checking saved task`);
      else throw error;
    } finally { await response.body.cancel().catch(() => {}); }
  }
  while (true) {
    if (fatal) throw fatal;
    const records = await api<Run[]>('/api/runs');
    const finished = records.find(r => r.id === pending.runId);
    if (!finished) throw new Error(`${questionId} 的 runId 未在历史记录中找到；暂停以避免重复生成`);
    if (finished.status === 'running') { await pause(); if (fatal) throw fatal; continue; }
    return finished;
  }
}

try {
  try { state = JSON.parse(await readFile(path.join(dir, 'state.json'), 'utf8')); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
  if (!state) {
    const settings = await api<ModelSettings>('/api/models');
    const selected = settings.models.find(m => m.id === settings.selectedId);
    if (!selected) throw new Error('未找到当前选中的模型配置；请先在页面选择模型');
    const questions = await api<Question[]>('/api/questions');
    if (!questions.length) throw new Error('题库为空');
    state = { model: { baseUrl: selected.baseUrl, modelName: selected.modelName }, params: { ...defaults }, modelConfigId: selected.id, questionIds: questions.map(q => q.id), startedAt: new Date().toISOString(), status: 'running', pending: [], results: [] };
  }
  pendingList(); // Also migrates old serial state files with a single `pending` object.
  recoveryPending = new Set(pendingList());
  state.status = 'running'; delete state.error; await save();
  log(`Batch started: ${state.questionIds.length} questions; concurrency=${concurrency}; params=${JSON.stringify(state.params)}`);
  fatal = undefined;
  const assigned = new Set<Pending>();
  const claim = () => {
    if (fatal) return undefined;
    const recovered = pendingList().find(item => !assigned.has(item));
    if (recovered) { assigned.add(recovered); return recovered; }
    const completed = new Set(state!.results.map(result => result.questionId));
    const pendingIds = new Set(pendingList().map(item => item.questionId));
    const questionId = state!.questionIds.find(id => !completed.has(id) && !pendingIds.has(id));
    if (!questionId) return undefined;
    const pending = { questionId, since: new Date().toISOString() };
    pendingList().push(pending);
    assigned.add(pending);
    return pending;
  };
  const worker = async () => {
    while (!fatal) {
      const pending = claim();
      if (!pending) return;
      await save();
      try {
        const finished = await submitAndWait(pending);
        state!.results.push({ questionId: pending.questionId, runId: finished.id, status: finished.status, totalMs: finished.totalMs });
        state!.pending = pendingList().filter(item => item !== pending);
        await save();
        log(`END ${pending.questionId} ${finished.status} ${finished.totalMs}ms (${state!.results.length}/${state!.questionIds.length})`);
        if (finished.status === 'cancelled' || finished.status === 'interrupted') throw new Error(`${pending.questionId} 被手动停止或后端重启，暂停批次`);
      } catch (error) {
        fatal ||= error instanceof Error ? error : new Error(String(error));
        return;
      }
    }
  };
  await Promise.all(Array.from({ length: concurrency }, worker));
  if (fatal) throw fatal;
  if (pendingList().length) throw new Error('仍有未完成任务，暂停以避免重复生成');
  state.status = 'completed'; await save(); log('Batch completed');
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  if (state) { state.status = 'paused'; state.error = message; await save(); }
  log(`PAUSED ${message}`); process.exitCode = 1;
} finally { await saveChain; await unlink(lockPath); }
