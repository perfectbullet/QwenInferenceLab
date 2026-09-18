import type { Db } from 'mongodb';
import type { ModelConfig, ModelSettings } from '../../shared/types.js';
type StoredModelConfig = ModelConfig & { apiKey?: string };
export function validateModel(input: unknown): Pick<StoredModelConfig, 'baseUrl' | 'modelName' | 'apiKey'> {
  const data = input as Partial<ModelConfig> | null;
  if (!data || typeof data.baseUrl !== 'string' || typeof data.modelName !== 'string') throw new Error('请填写模型 URL 和模型名称');
  let url: URL;
  try { url = new URL(data.baseUrl.trim()); } catch { throw new Error('模型 URL 格式不正确'); }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error('URL 仅支持 HTTP/HTTPS，不能包含凭据、查询参数或片段');
  const modelName = data.modelName.trim();
  if (!modelName || modelName.length > 200 || /[\x00-\x1f]/.test(modelName)) throw new Error('模型名称无效');
  const suppliedKey = (input as { apiKey?: unknown }).apiKey;
  if (suppliedKey !== undefined && typeof suppliedKey !== 'string') throw new Error('API Key 格式不正确');
  const apiKey = typeof suppliedKey === 'string' ? suppliedKey.trim() : undefined;
  if (apiKey !== undefined && (!apiKey || apiKey.length > 2000 || /[\x00-\x1f]/.test(apiKey))) throw new Error('API Key 无效');
  return { baseUrl: url.href.replace(/\/+$/, ''), modelName, ...(apiKey ? { apiKey } : {}) };
}
export class ModelStore {
  readonly configs;
  readonly settings;
  constructor(db: Db, prefix = '') {
    this.configs = db.collection<StoredModelConfig>(`${prefix}model_configs`);
    this.settings = db.collection<{ key: string; selectedId: string }>(`${prefix}model_settings`);
  }
  async initialize(initial: unknown) {
    await this.configs.createIndex({ id: 1 }, { unique: true });
    await this.configs.createIndex({ baseUrl: 1, modelName: 1 }, { unique: true });
    await this.settings.createIndex({ key: 1 }, { unique: true });
    if (!await this.configs.countDocuments()) {
      const config = validateModel(initial);
      await this.configs.updateOne({ id: 'default' }, { $setOnInsert: { id: 'default', ...config } }, { upsert: true });
    }
    const first = await this.configs.findOne({});
    await this.settings.updateOne({ key: 'selection' }, { $setOnInsert: { key: 'selection', selectedId: first!.id } }, { upsert: true });
  }
  async list(): Promise<ModelSettings> {
    const stored = await this.configs.find({}, { projection: { _id: 0 } }).toArray();
    return { models: stored.map(({ apiKey, ...model }) => ({ ...model, hasApiKey: Boolean(apiKey) })), selectedId: (await this.settings.findOne({ key: 'selection' }))?.selectedId || '' };
  }
  async get(id: string) { return this.configs.findOne({ id }, { projection: { _id: 0 } }); }
  async save(input: unknown, id = crypto.randomUUID() as string) {
    const inputConfig = validateModel(input);
    const existing = await this.configs.findOne({ id }, { projection: { apiKey: 1 } });
    // A blank field in the edit form means "leave the existing key unchanged".
    const config: StoredModelConfig = { id, ...inputConfig, ...(inputConfig.apiKey || existing?.apiKey ? { apiKey: inputConfig.apiKey || existing?.apiKey } : {}) };
    await this.configs.replaceOne({ id }, config, { upsert: true }); return config;
  }
  async select(id: string) {
    if (!await this.get(id)) throw new Error('模型配置不存在');
    await this.settings.updateOne({ key: 'selection' }, { $set: { selectedId: id } });
  }
}
