import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { defaults, type Params, type Question, type Run, type ModelConfig } from '../../shared/types';
import { ModelSelector } from './ModelSelector';
import { readSSE } from '../../shared/sse';
import { MathText } from './MathText';
import { SourceImage } from './SourceImage';
import { CopyButton } from './CopyButton';
import { Reasoning } from './Reasoning';
import './interaction.css';
import 'katex/dist/katex.min.css';
import './style.css';
const labels: Record<string, string> = { running: '进行中', completed: '已完成', cancelled: '已取消', truncated: '输出被截断', failed: '失败', interrupted: '运行中断' };
const seconds = (n: number | null) => n === null ? '未知' : `${(n / 1000).toFixed(1)} 秒`;
async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init); const body = await r.json();
  if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`); return body;
}
function App() {
  const [questions, setQuestions] = useState<Question[]>([]); const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState(''); const [search, setSearch] = useState('');
  const [tag, setTag] = useState(''); const [difficulty, setDifficulty] = useState('');
  const [params, setParams] = useState<Params>(defaults); const [busy, setBusy] = useState(false);
  const [liveRun, setLiveRun] = useState<Run | null>(null); const [phase, setPhase] = useState('waiting');
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [viewed, setViewed] = useState<Record<string, string>>({});
  const submitting = useRef(false);
  const [phaseAt, setPhaseAt] = useState(Date.now()); const [tick, setTick] = useState(Date.now());
  const [error, setError] = useState(''); const [selectedModel, setSelectedModel] = useState<ModelConfig | null>(null);
  const [remoteRun, setRemoteRun] = useState<Run | null>(null);
  const activeRun = busy ? liveRun : remoteRun;
  const activeQuestion = busy ? pendingQuestion : remoteRun?.questionId;
  const hasActive = busy || !!remoteRun;
  const isViewingActive = !!activeQuestion && selected === activeQuestion;
  const current = isViewingActive ? activeRun : runs.find(r => r.questionId === selected && r.id === viewed[selected]) || runs.find(r => r.questionId === selected) || null;
  const currentRunning = current?.status === 'running';
  const [evaluation, setEvaluation] = useState<Run['evaluation']>('unreviewed'); const [notes, setNotes] = useState('');
  const [saved, setSaved] = useState(false);
  async function refresh() {
    const records = await api<Run[]>('/api/runs'); setRuns(records);
  }
  useEffect(() => {
    Promise.all([api<Question[]>('/api/questions'), api<Run[]>('/api/runs')])
      .then(([qs, rs]) => { setQuestions(qs); setSelected(qs[0]?.id || ''); setRuns(rs); })
      .catch(e => setError(String(e)));
  }, []);
  useEffect(() => { if (!hasActive) return; const id = setInterval(() => setTick(Date.now()), 250); return () => clearInterval(id); }, [hasActive]);
  useEffect(() => {
    if (busy) return;
    let disposed = false;
    const poll = async () => {
      try {
        const active = await api<Run | null>('/api/active-run');
        const records = await api<Run[]>('/api/runs');
        if (!disposed) { setRemoteRun(active); setRuns(records); }
      } catch(e) { if (!disposed) setError(String(e)); }
    };
    void poll(); const id = setInterval(poll, 2000);
    return () => { disposed = true; clearInterval(id); };
  }, [busy]);
  useEffect(() => { setEvaluation(current?.evaluation || 'unreviewed'); setNotes(current?.notes || ''); setSaved(false); }, [current?.id]);
  const q = questions.find(q => q.id === selected);
  const filtered = questions.filter(q => `${q.id} ${q.question}`.toLowerCase().includes(search.toLowerCase()) && (!tag || q.tag === tag) && (!difficulty || q.difficulty === difficulty));
  const history = runs.filter(r => r.questionId === selected);
  async function run() {
    if (!q || !selectedModel || hasActive || submitting.current) return;
    submitting.current = true;
    setBusy(true); setPendingQuestion(q.id); setError(''); setLiveRun(null); setPhase('waiting'); setPhaseAt(Date.now()); setTick(Date.now());
    let done = false;
    try {
      const response = await fetch('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ questionId: q.id, params, modelConfigId: selectedModel.id }) });
      if (!response.ok) throw new Error((await response.json()).error);
      if (!response.body) throw new Error('未收到流式响应');
      for await (const data of readSSE(response.body)) {
        const event = JSON.parse(data);
        if (event.type === 'started') { setLiveRun(event.run); setViewed(v => ({ ...v, [event.run.questionId]: event.run.id })); }
        if (event.type === 'status') { setPhase(event.phase); setPhaseAt(Date.now()); setTick(Date.now()); if (event.thinkingMs !== undefined) setLiveRun(r => r ? { ...r, thinkingMs: event.thinkingMs } : r); }
        if (event.type === 'answer_delta') setLiveRun(r => r ? { ...r, answer: r.answer + event.text } : r);
        if (event.type === 'reasoning_delta') setLiveRun(r => r ? { ...r, reasoning: (r.reasoning || '') + event.text } : r);
        if (event.type === 'done') { setLiveRun(event.run); setRuns(rs => [event.run, ...rs.filter(r => r.id !== event.run.id)]); done = true; }
        if (event.type === 'error') throw new Error(event.error);
      }
      if (!done) throw new Error('浏览器连接中断；请查看历史记录确认运行状态');
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally {
      await refresh().catch(e => setError(String(e)));
      setRemoteRun(await api<Run | null>('/api/active-run').catch(() => null));
      setBusy(false); setPendingQuestion(null); submitting.current = false;
    }
  }
  async function cancel(id: string) {
    try { await api(`/api/runs/${id}/cancel`, { method: 'POST' }); } catch(e) { setError(String(e)); }
  }
  async function saveReview() {
    if (!current) return;
    try { const updated = await api<Run>(`/api/runs/${current.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ evaluation, notes }) }); setRuns(rs => rs.map(r => r.id === updated.id ? updated : r)); setSaved(true); await refresh(); } catch(e) { setError(String(e)); }
  }
  return <div className="app">
    <aside><div className="brand"><span className="logo">∑</span><div><strong>数学测试台</strong><small>QWEN · CoT LAB</small></div></div>
      <ModelSelector value={selectedModel} onChange={setSelectedModel} disabled={hasActive}/>
      <div className="library-title">题库 <span>{questions.length} 题</span></div>
      <input aria-label="搜索题目" placeholder="搜索题号或题目…" value={search} onChange={e => setSearch(e.target.value)}/>
      <div className="filters"><select aria-label="分类" value={tag} onChange={e => setTag(e.target.value)}><option value="">全部分类</option>{[...new Set(questions.map(q => q.tag).filter(Boolean))].map(t => <option key={t}>{t}</option>)}</select><select aria-label="难度" value={difficulty} onChange={e => setDifficulty(e.target.value)}><option value="">全部难度</option>{[...new Set(questions.map(q => q.difficulty).filter(Boolean))].map(t => <option key={t}>{t}</option>)}</select></div>
      <div className="question-list">{filtered.map(item => <button key={item.id} className={`question-item ${selected === item.id ? 'selected' : ''}`} onClick={() => { setSelected(item.id); setError(''); }}><div><strong>{item.id}</strong><span>{activeQuestion === item.id ? '正在解答' : item.difficulty}</span></div><p>{item.question.replace(/\$|\\[a-z]+/g, '').slice(0, 65)}</p><small>{item.tag}</small></button>)}{!filtered.length && <p className="muted">没有匹配题目</p>}</div><footer>个人测试工作台 · 思考模式已开启</footer>
    </aside>
    <main><header><div><span className="eyebrow">数学推理 / 单题测试</span><h1>让解题过程清晰可见</h1><p className="muted">查看正式解答，对照参考答案，积累每次测试。</p></div><span className="model"><i/> {activeRun?.modelName || activeRun?.model || selectedModel?.modelName || '请选择模型'} · CoT</span></header>
    {error && <div role="alert" className="error">{error}</div>}
    {hasActive && !isViewingActive && <div className="notice">{activeQuestion} 正在解答，可继续浏览其他题目。<button onClick={() => activeQuestion && setSelected(activeQuestion)}>返回正在解答的题目</button></div>}
    {q && <><section className="card"><div className="section-top"><h2>{q.id} <span className="badge">{q.tag} · {q.difficulty}</span></h2><CopyButton key={q.id} label="题目" text={q.question}/></div><MathText text={q.question}/><SourceImage key={`${q.id}-question`} title="题目原图" reference={q.image_path} url={q.image_url}/>
    <details className="settings"><summary>生成参数 <span>思考模式固定开启</span></summary><div className="parameter-grid">{(Object.keys(defaults) as (keyof Params)[]).map(key => <label key={key}>{key}<input type="number" disabled={busy} step={key === 'temperature' || key === 'top_p' ? '0.05' : '1'} value={params[key]} onChange={e => setParams(p => ({ ...p, [key]: Number(e.target.value) }))}/></label>)}</div><small>输出预算包含思考与正文；输入 + 输出不能超过模型 32768 token 上限。</small></details>
    <div className="actions"><button className="primary" disabled={hasActive || !selectedModel} onClick={run}>{isViewingActive ? '正在解答…' : '开始测试 →'}</button><button disabled={!isViewingActive || !activeRun} onClick={() => activeRun && cancel(activeRun.id)}>停止生成</button><span className="muted">每次独立解题 · 不发送参考答案</span></div></section>
    <section className="card answer"><div className="section-top"><h2>模型解答</h2><CopyButton key={current?.id || q.id} label="模型解答" text={current?.answer || ''}/>{current && <span className={`badge ${current.status === 'completed' ? 'success' : ''}`}>{labels[current.status]}</span>}</div>
    {current && <p className="run-model">本次模型：{current.modelName || current.model} · URL：{current.modelBaseUrl || '旧记录未保存'}</p>}
    {isViewingActive && <div className="thinking" role="status"><span className="pulse"/>{(busy ? phase : current?.answer ? 'answering' : current?.reasoning ? 'thinking' : 'waiting') === 'waiting' ? '等待模型响应…' : (busy ? phase === 'thinking' : !current?.answer) ? '思考中…' : '正在生成解答…'} <strong>{seconds(busy ? Math.max(0, tick - phaseAt) : Math.max(0, tick - Date.parse(current?.startedAt || new Date().toISOString())))}</strong>{current?.thinkingMs != null && <span> · 已思考 {seconds(current.thinkingMs)}</span>}</div>}
    {(current || isViewingActive) && <Reasoning key={current?.id || q.id} text={current?.reasoning} running={isViewingActive}/>}
    {current?.answer ? <MathText text={current.answer}/> : !isViewingActive && <div className="empty"><span>∫</span><p>选择题目，开始一次解题测试</p><small>思考过程默认折叠，可展开查看。</small></div>}
    {current && !currentRunning && <><div className="metrics"><div><small>思考阶段</small><strong>{seconds(current.thinkingMs)}</strong></div><div><small>总耗时</small><strong>{seconds(current.totalMs)}</strong></div><div><small>首响应</small><strong>{seconds(current.firstResponseMs)}</strong></div><div><small>生成 tokens</small><strong>{String(current.usage?.completion_tokens ?? '未知')}</strong></div></div><p className="asset-note">{current.thinkingNote} · 结束原因：{current.finishReason || '无'} · 思考时间不是 GPU 精确计时。</p>{current.status === 'truncated' && <div className="notice">输出预算耗尽，答案可能不完整。可提高预算后重试。</div>}{current.error && <div className="error">{current.error}</div>}
    <details><summary>本次运行参数</summary><pre>{JSON.stringify({ modelName: current.modelName || current.model, modelBaseUrl: current.modelBaseUrl || null, params: current.params, enable_thinking: true, systemPrompt: current.systemPrompt, usage: current.usage }, null, 2)}</pre></details>
    <div className="review"><label>人工评价<select value={evaluation} onChange={e => { setEvaluation(e.target.value as Run['evaluation']); setSaved(false); }}><option value="unreviewed">未评价</option><option value="correct">正确</option><option value="incorrect">错误</option><option value="review">待复核</option></select></label><textarea placeholder="记录问题、结论或需要复核的步骤…" aria-label="评价备注" value={notes} onChange={e => { setNotes(e.target.value); setSaved(false); }}/><button onClick={saveReview}>{saved ? '已保存 ✓' : '保存评价'}</button></div></>}
    </section>
    <details className="card reference"><summary>参考答案 <span>点击展开对照 · OCR 内容需人工复核</span></summary><div className="reference-copy"><CopyButton key={q.id} label="参考答案" text={q.reference_answer}/></div><MathText text={q.reference_answer}/><SourceImage key={`${q.id}-answer`} title="参考答案原图" reference={q.reference_answer_image_path} url={q.reference_answer_image_url}/></details>
    <section className="card"><div className="section-top"><h2>测试历史</h2><span className="muted">当前题目 · {history.length} 次</span></div>{history.length ? <div className="history">{history.map(r => <button disabled={isViewingActive || r.status === 'running'} key={r.id} onClick={() => setViewed(v => ({ ...v, [selected]: r.id }))}><span>{new Date(r.startedAt).toLocaleString('zh-CN')}</span><span>{labels[r.status]} · {seconds(r.totalMs)}</span><span>{{ unreviewed: '未评价', correct: '正确', incorrect: '错误', review: '待复核' }[r.evaluation]} ↗</span></button>)}</div> : <p className="muted">还没有测试记录。每次运行都会自动保存。</p>}</section></>}
    </main></div>;
}
createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
