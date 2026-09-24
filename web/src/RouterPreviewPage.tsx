import React, { useState } from 'react';
import type {
  RouterPreviewResponse,
  RouterProfile,
} from '../../shared/types';
import { MathText } from './MathText';
import './router-preview.css';

async function preview(question: string, profile: RouterProfile) {
  const response = await fetch('/api/router/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, profile }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload as RouterPreviewResponse;
}

const percent = (value: number | null) => (
  value === null ? '—' : `${(value * 100).toFixed(1)}%`
);

export function RouterPreviewPage({ onBack }: { onBack: () => void }) {
  const [question, setQuestion] = useState('');
  const [profile, setProfile] = useState<RouterProfile>('development');
  const [result, setResult] = useState<RouterPreviewResponse | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!question.trim() || pending) return;
    setPending(true);
    setError('');
    setResult(null);
    try {
      setResult(await preview(question.trim(), profile));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setPending(false);
    }
  }

  return <div className="router-page">
    <header className="router-header">
      <div>
        <button className="back-button" onClick={onBack}>← 返回测试台</button>
        <span className="eyebrow">SHADOW ROUTER / EXPERIMENTAL</span>
        <h1>Router 影子测试</h1>
        <p>只预测 Local / Cloud，不会启动模型推理。每次预测会保存到共享 MongoDB。</p>
      </div>
      <span className="router-version">Qwen3-Embedding-0.6B · Router V1</span>
    </header>

    <form className="router-input card" onSubmit={submit}>
      <label htmlFor="router-question">输入一道数学题</label>
      <textarea
        id="router-question"
        value={question}
        onChange={event => setQuestion(event.target.value)}
        placeholder="例如：求方程 x² - 5x + 6 = 0 的所有实数解。"
        rows={7}
        maxLength={50000}
      />
      <div className="router-controls">
        <label>策略档位
          <select
            value={profile}
            onChange={event => setProfile(event.target.value as RouterProfile)}
          >
            <option value="development">开发档 · Coverage 51.67%</option>
            <option value="conservative">保守档 · 更少 False Local</option>
          </select>
        </label>
        <button className="primary" disabled={pending || !question.trim()}>
          {pending ? '正在生成向量并检索…' : '分析路由 →'}
        </button>
      </div>
      <small>开发档用于跑通流程；结果不构成生产安全保证。</small>
    </form>

    {error && <div className="error" role="alert">{error}</div>}

    {result && <section className="router-result">
      <div className={`router-decision card ${result.decision}`}>
        <div>
          <span className="eyebrow">ROUTER DECISION</span>
          <h2>{result.decision === 'local' ? 'LOCAL' : 'CLOUD'}</h2>
          <p>{result.decision === 'local'
            ? '当前邻域满足本地模型开发档条件。'
            : '当前邻域证据不足，建议交给 Cloud。'}</p>
        </div>
        <div className="router-score">
          <small>Router Score</small>
          <strong>{percent(result.score)}</strong>
          <span>{result.durationMs.toFixed(0)} ms</span>
        </div>
      </div>

      <div className="router-metrics">
        <div><small>Top-1 相似度</small><strong>{percent(result.features.top1Similarity)}</strong></div>
        <div><small>邻居成功均值</small><strong>{percent(result.features.neighborSuccessMean)}</strong></div>
        <div><small>不完美邻居</small><strong>{result.features.nonPerfectNeighborCount}/{result.features.usableNeighborCount}</strong></div>
        <div><small>安全风险距离差</small><strong>{result.features.safeVsRiskMargin?.toFixed(3) ?? '—'}</strong></div>
      </div>

      <section className="card router-details">
        <div className="section-top">
          <h2>判断依据</h2>
          <span className="badge">{result.profile}</span>
        </div>
        <div className="reason-codes">
          {result.reasonCodes.map(code => <code key={code}>{code}</code>)}
        </div>
        <details>
          <summary>查看完整 Features 与 Policy</summary>
          <pre>{JSON.stringify({
            features: result.features,
            policyConfig: result.policyConfig,
            routerVersion: result.routerVersion,
            embeddingVersion: result.embeddingVersion,
            corpusSize: result.corpusSize,
          }, null, 2)}</pre>
        </details>
      </section>

      <section className="card router-neighbors">
        <div className="section-top">
          <h2>Top-{result.neighbors.length} 相似历史题</h2>
          <span className="muted">仅 labelUsable=true 参与能力聚合</span>
        </div>
        <div className="neighbor-list">
          {result.neighbors.map((neighbor, index) => <article key={neighbor.questionId}>
            <div className="neighbor-rank">{index + 1}</div>
            <div className="neighbor-question">
              <div>
                <strong>{neighbor.questionId}</strong>
                <span>{neighbor.mathType || '未分类'} · {neighbor.difficulty || '未知难度'}</span>
              </div>
              <MathText text={neighbor.question}/>
            </div>
            <div className="neighbor-stats">
              <span>相似度 <strong>{percent(neighbor.similarity)}</strong></span>
              <span>本地成功率 <strong>{percent(neighbor.localSuccessRate)}</strong></span>
              <span className={neighbor.safeLocal === false ? 'risk' : 'safe'}>
                {neighbor.labelUsable
                  ? (neighbor.safeLocal ? 'Safe' : 'Non-perfect')
                  : 'Unknown'}
              </span>
            </div>
          </article>)}
        </div>
      </section>
    </section>}
  </div>;
}
