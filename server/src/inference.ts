import { readSSE } from '../../shared/sse.js';
import type { Run } from '../../shared/types.js';
export const systemPrompt = '请使用中文解答数学问题。最终解答应包含清晰的解题步骤与结论。公式使用 LaTeX，行内公式用 $...$，独立公式用 $$...$$；适合时将最终结果放在 \\boxed{} 中。';
export type Emit = (event: Record<string, unknown>) => void;
export async function infer(run: Run, base: string, signal: AbortSignal, emit: Emit, checkpoint: () => Promise<void>, apiKey = process.env.MODEL_API_KEY || '') {
  const start = performance.now(); let thinkingStart: number | null = null;
  let answerStart: number | null = null; let lastSave = start; let ended = false;
  const now = () => Math.round(performance.now() - start);
  emit({ type: 'status', phase: 'waiting', elapsedMs: 0 });
  try {
    const response = await fetch(`${base.replace(/\/$/, '')}/chat/completions`, {
      method: 'POST', signal,
      headers: { 'Content-Type': 'application/json', ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}) },
      body: JSON.stringify(run.request),
    });
    if (!response.ok) throw new Error(`模型接口 HTTP ${response.status}: ${(await response.text()).slice(0, 300)}`);
    if (!response.body) throw new Error('模型接口未返回流');
    for await (const data of readSSE(response.body)) {
      if (data === '[DONE]') { ended = true; break; }
      const chunk = JSON.parse(data);
      if (chunk.error) throw new Error(chunk.error.message || '模型返回错误');
      if (chunk.usage) run.usage = chunk.usage;
      const choice = chunk.choices?.[0];
      if (choice) {
        const delta = choice.delta || {};
        const thought = delta.reasoning || delta.reasoning_content;
        if (typeof thought === 'string' && thought.length) {
          run.reasoning = (run.reasoning || '') + thought;
          emit({ type: 'reasoning_delta', text: thought });
          run.firstResponseMs ??= now();
          if (thinkingStart === null && answerStart === null) {
            thinkingStart = now(); emit({ type: 'status', phase: 'thinking', elapsedMs: now() });
          }
        }
        if (typeof delta.content === 'string' && delta.content.length) {
          run.firstResponseMs ??= now();
          if (answerStart === null) {
            answerStart = now();
            if (thinkingStart !== null) run.thinkingMs = answerStart - thinkingStart;
            emit({ type: 'status', phase: 'answering', elapsedMs: now(), thinkingMs: run.thinkingMs });
          }
          run.answer += delta.content;
          emit({ type: 'answer_delta', text: delta.content });
        }
        if (delta.tool_calls?.length) throw new Error('收到意外工具调用；本应用仅支持 CoT 解题。');
        if (choice.finish_reason) run.finishReason = choice.finish_reason;
      }
      run.totalMs = now();
      if (performance.now() - lastSave > 2000) { await checkpoint(); lastSave = performance.now(); }
    }
    if (!ended || !run.finishReason) throw new Error('模型连接提前结束，未收到完整结束事件');
    if (run.finishReason === 'length') run.status = 'truncated';
    else if (run.finishReason !== 'stop') throw new Error(`异常结束原因：${run.finishReason}`);
    else if (!run.answer.trim()) throw new Error('模型未生成正式解答');
    else run.status = 'completed';
  } catch (error) {
    if (signal.aborted && signal.reason === 'cancelled') run.status = 'cancelled';
    else {
      run.status = 'failed';
      run.error = signal.aborted ? '请求超时或连接已中断' : error instanceof Error ? error.message : String(error);
    }
  } finally {
    run.totalMs = now();
    run.thinkingNote = run.thinkingMs !== null ? '按流式片段到达时间估算' : thinkingStart !== null ? '思考阶段未完整结束，无法统计' : '未观察到独立思考片段';
  }
}
