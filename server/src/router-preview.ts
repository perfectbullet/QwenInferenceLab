import type { Collection, Db } from 'mongodb';
import type {
  RouterPreviewRequest,
  RouterPreviewResponse,
  RouterPredictionRecord,
} from '../../shared/types.js';

const profiles = new Set(['development', 'conservative']);

export function validateRouterPreviewRequest(input: unknown): RouterPreviewRequest {
  if (!input || typeof input !== 'object') throw new Error('请求格式错误');
  const value = input as Record<string, unknown>;
  const question = typeof value.question === 'string' ? value.question.trim() : '';
  const profile = value.profile === undefined ? 'development' : value.profile;
  const questionId = value.questionId;
  if (!question || question.length > 50_000) throw new Error('题目长度必须为 1–50000 字符');
  if (typeof profile !== 'string' || !profiles.has(profile)) throw new Error('Router profile 无效');
  if (questionId !== undefined && (typeof questionId !== 'string' || !questionId.trim())) {
    throw new Error('questionId 无效');
  }
  return {
    question,
    profile: profile as RouterPreviewRequest['profile'],
    ...(typeof questionId === 'string' ? { questionId: questionId.trim() } : {}),
  };
}

export class RouterRuntimeClient {
  readonly baseUrl: string;
  readonly timeoutMs: number;

  constructor(
    baseUrl = process.env.ROUTER_RUNTIME_URL || 'http://127.0.0.1:8100',
    timeoutMs = Number(process.env.ROUTER_RUNTIME_TIMEOUT_MS || 90_000),
  ) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    if (!Number.isFinite(timeoutMs) || timeoutMs < 1000 || timeoutMs > 300_000) {
      throw new Error('ROUTER_RUNTIME_TIMEOUT_MS 必须为 1000–300000');
    }
    this.timeoutMs = timeoutMs;
  }

  async preview(input: RouterPreviewRequest): Promise<RouterPreviewResponse> {
    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch {
      throw new Error('Router Runtime 不可用');
    }
    const payload = await response.json().catch(() => ({})) as Record<string, unknown>;
    if (!response.ok) {
      const detail = typeof payload.detail === 'string' ? payload.detail : 'Router Runtime 请求失败';
      throw new Error(detail);
    }
    return payload as unknown as RouterPreviewResponse;
  }
}

export class RouterPredictionStore {
  readonly collection: Collection<RouterPredictionRecord>;

  constructor(db: Db, name = 'router_predictions') {
    this.collection = db.collection<RouterPredictionRecord>(name);
  }

  async initialize() {
    await this.collection.createIndex({ predictionId: 1 }, { unique: true });
    await this.collection.createIndex({ createdAt: -1 });
    await this.collection.createIndex({ profile: 1, decision: 1, createdAt: -1 });
    await this.collection.createIndex({ textHash: 1, createdAt: -1 });
  }

  async save(record: RouterPredictionRecord) {
    await this.collection.insertOne({ ...record });
  }
}
