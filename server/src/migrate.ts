import 'dotenv/config';
import { readFile, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import type { Question, Run } from '../../shared/types.js';
import { connectDatabase } from './database.js';
import { QuestionStore, RunStore } from './store.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
async function migrate() {
  // Validate all input before any database writes. Originals remain untouched.
  const raw = await readFile(path.resolve(root, process.env.DATASET_PATH || 'math_qa_275_20260617.mineru.filled_reference_answer-v2.jsonl'), 'utf8');
  const questions: Question[] = raw.split(/\r?\n/).filter(s => s.trim()).map((line, i) => {
    const q = JSON.parse(line);
    if (typeof q.id !== 'string' || !q.id || typeof q.question !== 'string' || !q.question.trim() || typeof q.reference_answer !== 'string' || !q.reference_answer.trim()) throw new Error(`题库第 ${i + 1} 行不完整`);
    const { _id, ...question } = q;
    return question;
  });
  if (new Set(questions.map(q => q.id)).size !== questions.length) throw new Error('题库 ID 重复');
  const dir = path.resolve(root, process.env.RUNS_DIR || 'data/runs');
  const names = await readdir(dir).catch(e => { if (e.code === 'ENOENT') return []; throw e; });
  const runs: Run[] = [];
  for (const name of names.filter(n => /^[a-f0-9-]+\.json$/.test(n))) {
    const run = JSON.parse(await readFile(path.join(dir, name), 'utf8'));
    if (run.id !== name.slice(0, -5) || typeof run.questionId !== 'string' || typeof run.answer !== 'string' || typeof run.startedAt !== 'string' || !['running','completed','cancelled','truncated','failed','interrupted'].includes(run.status)) throw new Error(`历史记录格式错误：${name}`);
    const { _id, ...record } = run;
    runs.push(record);
  }
  const { client, db } = await connectDatabase();
  try {
    const questionStore = new QuestionStore(db); const runStore = new RunStore(db);
    await questionStore.initialize(); await runStore.initialize();
    const importedQuestions = await questionStore.importMissing(questions);
    let importedRuns = 0;
    for (const run of runs) importedRuns += await runStore.importMissing(run);
    console.log(JSON.stringify({ sourceQuestions: questions.length, importedQuestions, sourceRuns: runs.length, importedRuns, existingRecords: '保留数据库已有内容，未覆盖', originals: '原始文件保留' }));
  } finally { await client.close(); }
}
migrate().catch(() => { console.error('迁移失败：请核对 MongoDB 配置、权限与源文件格式；原始文件未删除。'); process.exitCode = 1; });
