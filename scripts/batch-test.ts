import { mkdir, readFile, writeFile, rename, open, unlink } from 'node:fs/promises';
import path from 'node:path';
import { defaults, type Run, type ModelSettings, type Question } from '../shared/types.js';
import { readSSE } from '../shared/sse.js';

const dir = path.resolve(process.argv[2] || 'test-results/batch-current');
const endpoint = process.env.BATCH_API_URL || 'http://127.0.0.1:3100';
type BatchModel = { baseUrl: string; modelName: string };
const pause = () => new Promise(resolve => setTimeout(resolve, 3000));
async function api<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(endpoint + url, { signal: AbortSignal.timeout(30000), ...(body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }) });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}
type State = { model: BatchModel; params: typeof defaults; questionIds: string[]; modelConfigId: string;
  startedAt: string; status: string; pending?: { questionId: string; since: string; runId?: string };
  results: { questionId: string; runId: string; status: string; totalMs: number }[]; error?: string };
await mkdir(dir, { recursive: true });
const lockPath = path.join(dir, 'worker.lock');
const lock = await open(lockPath, 'wx');
await lock.writeFile(String(process.pid)); await lock.close();
let state: State | undefined;
async function save() {
  await writeFile(path.join(dir, 'state.tmp'), JSON.stringify(state, null, 2));
  await rename(path.join(dir, 'state.tmp'), path.join(dir, 'state.json'));
}
const log = (message: string) => console.log(`${new Date().toISOString()} ${message}`);
try {
  try { state = JSON.parse(await readFile(path.join(dir, 'state.json'), 'utf8')); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
  if (!state) {
    const settings = await api<ModelSettings>('/api/models');
    const selected = settings.models.find(m => m.id === settings.selectedId);
    if (!selected) throw new Error('未找到当前选中的模型配置；请先在页面选择模型');
    const model: BatchModel = { baseUrl: selected.baseUrl, modelName: selected.modelName };
    const questions = await api<Question[]>('/api/questions');
    if (!questions.length) throw new Error('题库为空');
    state = { model, params: { ...defaults }, modelConfigId: selected.id, questionIds: questions.map(q => q.id), startedAt: new Date().toISOString(), status: 'running', results: [] };
  }
  state.status = 'running'; delete state.error; await save();
  log(`Batch started: ${state.questionIds.length} questions; params=${JSON.stringify(state.params)}`);
  for (const questionId of state.questionIds) {
    if (state.results.some(r => r.questionId === questionId)) continue;
    // Recover a disconnected/previous subscriber without submitting the question twice.
    if (!state.pending) {
      while (await api<Run | null>('/api/active-run')) await pause();
      state.pending = { questionId, since: new Date().toISOString() }; await save();
      const response = await fetch(endpoint + '/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ questionId, params: state.params, modelConfigId: state.modelConfigId }) });
      if (response.status === 409) { delete state.pending; await save(); throw new Error('另一个任务抢先启动；批次暂停，可重启脚本继续'); }
      if (!response.ok || !response.body) throw new Error(`提交失败 HTTP ${response.status}；请核对后再恢复`);
      try {
        for await (const data of readSSE(response.body)) {
          const event = JSON.parse(data);
          if (event.type === 'started') {
            state.pending.runId = event.run.id; await save(); log(`START ${questionId} ${event.run.id}`);
          }
          if (event.type === 'done') break;
          if (event.type === 'error') throw new Error(event.error);
        }
      } catch { log(`Subscriber disconnected for ${questionId}; checking saved task`); }
      finally { await response.body.cancel().catch(() => {}); }
    }
    if (state.pending.questionId !== questionId) throw new Error('批次进度不一致');
    let finished: Run;
    while (true) {
      const records = await api<Run[]>('/api/runs');
      const matches = records.filter(r => state!.pending!.runId ? r.id === state!.pending!.runId :
        r.questionId === questionId && r.startedAt >= state!.pending!.since && r.modelBaseUrl === state!.model.baseUrl && r.modelName === state!.model.modelName);
      if (matches.length !== 1) throw new Error('无法唯一确认已提交任务；暂停以避免重复生成，请检查数据库');
      finished = matches[0];
      if (finished.status !== 'running') break;
      await pause();
    }
    state.results.push({ questionId, runId: finished.id, status: finished.status, totalMs: finished.totalMs });
    delete state.pending; await save();
    log(`END ${questionId} ${finished.status} ${finished.totalMs}ms (${state.results.length}/${state.questionIds.length})`);
    if (finished.status === 'cancelled' || finished.status === 'interrupted') throw new Error('任务被手动停止或后端重启，暂停批次');
  }
  state.status = 'completed'; await save(); log('Batch completed');
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  if (state) { state.status = 'paused'; state.error = message; await save(); }
  log(`PAUSED ${message}`); process.exitCode = 1;
} finally { await unlink(lockPath); }
