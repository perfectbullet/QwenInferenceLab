export interface Question {
  id: string; question: string; reference_answer: string;
  tag?: string; difficulty?: string; math_type?: string;
  image_path?: string; reference_answer_image_path?: string;
  image_url?: string; reference_answer_image_url?: string;
}
export interface Params {
  temperature: number; top_p: number; top_k: number; min_p: number;
  presence_penalty: number; repetition_penalty: number; max_tokens: number;
}
/** General-purpose reasoning sampling profile. */
export const defaults: Params = {
  temperature: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0,
  presence_penalty: 1.5, repetition_penalty: 1.0, max_tokens: 20480,
};
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


export type MathVerdict = "correct" | "incorrect" | "review" | "unresolved";
export interface MachineEvaluation {
  id: string; runId: string; questionId: string; pipelineVersion: string;
  goldAdapterVersion: string; mathVerifyVersion: string;
  runtime: { status: RunStatus; finishReason?: string | null; error?: string | null };
  math: { verdict: MathVerdict; level: number; method: string; confidence: number; reasonCode: string };
  extraction?: Record<string, unknown> | null;
  verifyConfig?: Record<string, unknown> | null;
  judge?: Record<string, unknown> | null;
  createdAt?: string; updatedAt?: string;
}
export interface EvaluationSummary {
  total: number;
  verdicts: Record<string, number>;
  levels: Record<string, number>;
  methods: Record<string, number>;
}
export interface EvaluationListResponse {
  items: MachineEvaluation[]; total: number; page: number; limit: number;
  summary: EvaluationSummary;
}
