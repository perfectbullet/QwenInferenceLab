import { useEffect, useState } from 'react';
import type { ModelConfig, ModelSettings } from '../../shared/types';
import './models.css';
async function api<T>(url: string, method = 'GET', body?: unknown): Promise<T> {
  const r = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, ...(body ? { body: JSON.stringify(body) } : {}) });
  const data = await r.json(); if (!r.ok) throw new Error(data.error || '模型配置保存失败'); return data;
}
export function ModelSelector({ value, onChange, disabled }: { value: ModelConfig | null; onChange: (value: ModelConfig | null) => void; disabled: boolean }) {
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [working, setWorking] = useState(false); const [error, setError] = useState('');
  const [baseUrl, setBaseUrl] = useState(''); const [modelName, setModelName] = useState('');
  const [editId, setEditId] = useState<string | null>(null);
  useEffect(() => {
    let disposed = false;
    api<ModelSettings>('/api/models').then(s => {
      if (!disposed) { setModels(s.models); onChange(s.models.find(m => m.id === s.selectedId) || null); }
    }).catch(e => { if (!disposed) setError(String(e)); });
    return () => { disposed = true; };
  }, [onChange]);
  async function choose(config: ModelConfig) {
    const previous = value; setWorking(true); setError(''); onChange(null);
    try { await api('/api/models/selection', 'POST', { id: config.id }); onChange(config); }
    catch(e) { setError(String(e)); onChange(previous); }
    finally { setWorking(false); }
  }
  async function save() {
    setWorking(true); setError('');
    try {
      const saved = await api<ModelConfig>(editId ? `/api/models/${editId}` : '/api/models', editId ? 'PATCH' : 'POST', { baseUrl, modelName });
      const settings = await api<ModelSettings>('/api/models'); setModels(settings.models);
      await choose(saved); setEditId(null); setBaseUrl(''); setModelName('');
    } catch(e) { setError(String(e)); }
    finally { setWorking(false); }
  }
  const locked = disabled || working;
  return <div className="model-selector">
    <label>模型 URL<select aria-label="模型 URL" disabled={locked || !models.length} value={value?.baseUrl || ''} onChange={e => { const config = models.find(m => m.baseUrl === e.target.value); if (config) void choose(config); }}><option value="" disabled>选择地址</option>{[...new Set(models.map(m => m.baseUrl))].map(url => <option key={url}>{url}</option>)}</select></label>
    <label>模型名称<select aria-label="模型名称" disabled={locked || !value} value={value?.id || ''} onChange={e => { const config = models.find(m => m.id === e.target.value); if (config) void choose(config); }}><option value="" disabled>选择模型</option>{models.filter(m => m.baseUrl === value?.baseUrl).map(m => <option value={m.id} key={m.id}>{m.modelName}</option>)}</select></label>
    <details><summary>管理模型配置</summary>
      <div className="model-edit-actions"><button disabled={locked} onClick={() => { setEditId(null); setBaseUrl(''); setModelName(''); }}>新增</button><button disabled={locked || !value} onClick={() => { if (value) { setEditId(value.id); setBaseUrl(value.baseUrl); setModelName(value.modelName); } }}>编辑当前</button></div>
      <label>API 基础地址<input aria-label="配置模型 URL" disabled={locked} placeholder="http://服务器:8200/v1" value={baseUrl} onChange={e => setBaseUrl(e.target.value)}/></label>
      <label>服务模型名称<input aria-label="配置模型名称" disabled={locked} placeholder="qwen38-27b" value={modelName} onChange={e => setModelName(e.target.value)}/></label>
      <button disabled={locked || !baseUrl.trim() || !modelName.trim()} onClick={save}>{working ? '保存中…' : editId ? '保存修改并选用' : '保存并选用'}</button>
    </details>
    {error && <p role="alert" className="error">{error}</p>}
  </div>;
}
