import 'dotenv/config';
import crypto from 'node:crypto';
import { connectDatabase } from '../server/src/database.js';
import { readSSE } from '../shared/sse.js';
import type { Question } from '../shared/types.js';

const args = new Map(process.argv.slice(2).map((value, index, all) => value.startsWith('--') ? [value.slice(2), all[index + 1]] : ['', '']));
const variant = args.get('variant');
if (!variant || !['A', 'B', 'C', 'D'].includes(variant)) throw new Error('请传入 --variant A|B|C|D');
const runId = args.get('run-id') || `qwen36-ab-${new Date().toISOString().replace(/[:.]/g, '-')}`;
const collectionName = args.get('collection') || 'qwen36_ab_benchmark_20260921';
const baseUrl = (process.env.QWEN36_API_URL || 'http://192.168.8.231:8200/v1').replace(/\/$/, '');
const model = 'nvidia/Qwen3.6-35B-A3B-NVFP4';
const questionIds = ['MATH-040', 'MATH-187'];
const promptSuffix = '请逐步推理，并将最终答案放在 \\boxed{} 中。';
const maxTokens = Number(args.get('max-tokens') || 32768);
if (!Number.isInteger(maxTokens) || maxTokens < 1) throw new Error('--max-tokens 必须是正整数');

/*
 * Qwen3.6 官方模型卡的 thinking-mode 建议（2026-09 核对）：
 *   temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
 *   presence_penalty=1.5, repetition_penalty=1.0。
 *
 * 该模型没有 reasoning_effort 分档；它只支持 thinking 开/关。
 * 因此不要发送 reasoning_effort=low。Qwen3.6 默认 thinking，此处显式
 * enable_thinking:true，避免不同 chat template 或运行时默认值影响对比。
 *
 * 官方建议多数请求预留 32,768 output tokens，故脚本默认该上限；纯速度
 * 压测可用 --max-tokens 4096 限定预算，但必须让所有 A/B/C/D 组保持相同值。
 * 参数支持范围依 vLLM 版本而异；本项目的 vLLM v0.28.0 已接受这些字段。
 */
const params = {
  temperature: 1.0,
  top_p: 0.95,
  top_k: 20,
  min_p: 0.0,
  presence_penalty: 1.5,
  repetition_penalty: 1.0,
  max_tokens: maxTokens,
};
const chatTemplateKwargs = { enable_thinking: true };
const variantConfig = {
  A: { speculativeConfig: false, enforceEager: false },
  B: { speculativeConfig: false, enforceEager: true },
  C: { speculativeConfig: true, enforceEager: true },
  D: { speculativeConfig: true, enforceEager: false },
}[variant];

type InferenceResult = {
  answer: string; reasoning: string; finishReason: string | null;
  usage: Record<string, number> | null; firstTokenMs: number | null; totalMs: number;
  decodeTokensPerSecond: number | null;
};

async function infer(prompt: string, outputTokenBudget = params.max_tokens): Promise<InferenceResult> {
  const started = performance.now();
  let firstTokenMs: number | null = null;
  let answer = ''; let reasoning = ''; let finishReason: string | null = null;
  let usage: Record<string, number> | null = null;
  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST', signal: AbortSignal.timeout(10 * 60_000),
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content: prompt }],
      ...params, max_tokens: outputTokenBudget, stream: true,
      stream_options: { include_usage: true },
      chat_template_kwargs: chatTemplateKwargs,
    }),
  });
  if (!response.ok || !response.body) throw new Error(`模型接口 HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`);
  for await (const data of readSSE(response.body)) {
    if (data === '[DONE]') break;
    const chunk = JSON.parse(data);
    if (chunk.usage) usage = chunk.usage;
    const choice = chunk.choices?.[0];
    if (!choice) continue;
    const delta = choice.delta || {};
    const thought = delta.reasoning ?? delta.reasoning_content;
    if (typeof thought === 'string' && thought) { firstTokenMs ??= performance.now() - started; reasoning += thought; }
    if (typeof delta.content === 'string' && delta.content) { firstTokenMs ??= performance.now() - started; answer += delta.content; }
    if (choice.finish_reason) finishReason = choice.finish_reason;
  }
  const totalMs = performance.now() - started;
  const completionTokens = usage?.completion_tokens;
  const decodeSeconds = firstTokenMs === null ? 0 : (totalMs - firstTokenMs) / 1000;
  return {
    answer, reasoning, finishReason, usage, firstTokenMs: firstTokenMs === null ? null : Math.round(firstTokenMs),
    totalMs: Math.round(totalMs),
    decodeTokensPerSecond: completionTokens && decodeSeconds > 0 ? Number((completionTokens / decodeSeconds).toFixed(2)) : null,
  };
}

function lastBoxed(text: string): string | null {
  const marker = '\\boxed{'; let start = text.lastIndexOf(marker);
  if (start < 0) return null;
  start += marker.length; let depth = 1;
  for (let index = start; index < text.length; index++) {
    if (text[index] === '{') depth++;
    if (text[index] === '}' && --depth === 0) return text.slice(start, index);
  }
  return null;
}

const normalize = (value: string) => value.replace(/\s+/g, '').replace(/\\left|\\right|\$|[。,.，]/g, '').toLowerCase();
function expectedAnswer(reference: string): string {
  return lastBoxed(reference) || reference.replace(/【(?:答案|分析|解答)】[\s\S]*$/u, '').replace(/^【答案】/u, '').trim();
}

const { client, db } = await connectDatabase();
try {
  const questions = await db.collection<Question>('questions').find({ id: { $in: questionIds } }, { projection: { _id: 0 } }).toArray();
  questions.sort((a, b) => questionIds.indexOf(a.id) - questionIds.indexOf(b.id));
  if (questions.length !== questionIds.length) throw new Error(`题目不完整：预期 ${questionIds.length}，实际 ${questions.length}`);

  const startedAt = new Date().toISOString();
  const warmups = [];
  for (let index = 0; index < 2; index++) warmups.push(await infer(`计算 1+1。${promptSuffix}`, 256));

  const results = [];
  for (const question of questions) {
    const prompt = `${question.question}\n\n${promptSuffix}`;
    const result = await infer(prompt);
    const boxedAnswer = lastBoxed(result.answer);
    const expected = expectedAnswer(question.reference_answer);
    const normalizedBoxed = boxedAnswer ? normalize(boxedAnswer) : '';
    const normalizedExpected = normalize(expected);
    results.push({
      questionId: question.id, question: question.question, referenceAnswer: question.reference_answer,
      difficulty: question.difficulty, mathType: question.math_type, prompt,
      ...result, boxedAnswer, boxedCompliant: boxedAnswer !== null,
      answerMatchHeuristic: Boolean(normalizedBoxed && normalizedExpected && (normalizedBoxed === normalizedExpected || normalizedBoxed.includes(normalizedExpected) || normalizedExpected.includes(normalizedBoxed))),
    });
    console.log(`${variant} ${question.id}: ${result.finishReason} ${result.totalMs}ms ${result.decodeTokensPerSecond ?? '-'} tok/s boxed=${boxedAnswer !== null}`);
  }

  const completed = results.filter(item => item.finishReason === 'stop').length;
  const tps = results.map(item => item.decodeTokensPerSecond).filter((value): value is number => value !== null);
  const totals = results.map(item => item.totalMs);
  const summary = {
    questionCount: results.length, completed,
    boxedCompliant: results.filter(item => item.boxedCompliant).length,
    heuristicMatches: results.filter(item => item.answerMatchHeuristic).length,
    averageTotalMs: Math.round(totals.reduce((sum, value) => sum + value, 0) / totals.length),
    averageDecodeTokensPerSecond: tps.length ? Number((tps.reduce((sum, value) => sum + value, 0) / tps.length).toFixed(2)) : null,
  };
  const document = {
    id: crypto.randomUUID(), runId, variant, variantConfig, startedAt, finishedAt: new Date().toISOString(),
    collectionName, server: '192.168.8.231', apiBaseUrl: baseUrl, model,
    serverParameters: {
      tensorParallelSize: 1, quantization: 'modelopt_fp4', kvCacheDtype: 'fp8', blockSize: 128,
      moeBackend: 'marlin', attentionBackend: 'flashinfer', trtllmAttention: true,
      gpuMemoryUtilization: 0.85, maxModelLen: 20480, maxNumSeqs: 2,
      maxNumBatchedTokens: 8192, chunkedPrefill: true, prefixCaching: true,
      asyncScheduling: true, languageModelOnly: true, reasoningParser: 'qwen3',
      speculativeConfig: variantConfig.speculativeConfig ? { method: 'mtp', numSpeculativeTokens: 4, moeBackend: 'flashinfer_cutlass' } : null,
      enforceEager: variantConfig.enforceEager,
    },
    clientParameters: { ...params, chatTemplateKwargs }, promptSuffix,
    warmup: { count: warmups.length, prompt: `计算 1+1。${promptSuffix}`, results: warmups },
    questionIds, questions: questions.map(({ id, question, reference_answer, difficulty, math_type }) => ({ id, question, referenceAnswer: reference_answer, difficulty, mathType: math_type })),
    results, summary,
  };
  await db.collection(collectionName).createIndex({ runId: 1, variant: 1 }, { unique: true });
  await db.collection(collectionName).replaceOne({ runId, variant }, document, { upsert: true });
  console.log(JSON.stringify({ runId, variant, collectionName, summary }, null, 2));
} finally {
  await client.close();
}
