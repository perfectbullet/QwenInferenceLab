export interface Question {
  id: string; question: string; reference_answer: string;
  tag?: string; difficulty?: string; math_type?: string;
  image_path?: string; reference_answer_image_path?: string;
  image_url?: string; reference_answer_image_url?: string;
}
export interface Params { temperature: number; top_p: number; top_k: number; max_tokens: number }
export const defaults: Params = { temperature: 1, top_p: 0.95, top_k: 20, max_tokens: 8192 };
export type RunStatus = 'running' | 'completed' | 'cancelled' | 'truncated' | 'failed' | 'interrupted';
/** API keys are deliberately omitted from this client-facing shape. */
export interface ModelConfig { id: string; baseUrl: string; modelName: string; hasApiKey?: boolean }
export interface ModelSettings { models: ModelConfig[]; selectedId: string }
export interface Run {
  id: string; questionId: string; question: string; startedAt: string;
  model: string; systemPrompt: string; params: Params;
  modelBaseUrl?: string; modelName?: string; modelConfigId?: string;
  request: Record<string, unknown>;
  answer: string; reasoning?: string; status: RunStatus; error?: string; finishReason: string | null;
  firstResponseMs: number | null; thinkingMs: number | null; totalMs: number;
  thinkingNote: string; usage: Record<string, unknown> | null;
  evaluation: 'unreviewed' | 'correct' | 'incorrect' | 'review'; notes: string;
}
