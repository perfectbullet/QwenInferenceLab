import { useEffect, useMemo, useState } from "react";
import type { EvaluationListResponse, MachineEvaluation } from "../../shared/types";
import "./evaluations.css";

const verdictLabel: Record<string, string> = { correct: "正确", incorrect: "错误", review: "待复核", unresolved: "未解决" };
const methodLabel: Record<string, string> = { math_verify: "Math-Verify", llm_multi_part: "多小问 Judge", llm_semantic: "语义 Judge", runtime_precheck: "运行预检" };

async function loadEvaluations(params: URLSearchParams, signal: AbortSignal) {
  const response = await fetch(`/api/evaluations?${params}`, { signal });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body as EvaluationListResponse;
}

export function EvaluationsPage({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<EvaluationListResponse | null>(null);
  const [page, setPage] = useState(1);
  const [verdict, setVerdict] = useState("");
  const [level, setLevel] = useState("");
  const [method, setMethod] = useState("");
  const [questionId, setQuestionId] = useState("");
  const [runId, setRunId] = useState("");
  const [error, setError] = useState("");
  const params = useMemo(() => {
    const value = new URLSearchParams({ page: String(page), limit: "25" });
    if (verdict) value.set("verdict", verdict);
    if (level) value.set("level", level);
    if (method) value.set("method", method);
    if (questionId.trim()) value.set("questionId", questionId.trim());
    if (runId.trim()) value.set("runId", runId.trim());
    return value;
  }, [page, verdict, level, method, questionId, runId]);
  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(() => {
      setError("");
      loadEvaluations(params, controller.signal).then(setData).catch(reason => {
        if (reason.name !== "AbortError") setError(String(reason));
      });
    }, 180);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [params]);
  const pages = Math.max(1, Math.ceil((data?.total || 0) / (data?.limit || 25)));
  const resetPage = (setter: (value: string) => void) => (value: string) => { setter(value); setPage(1); };
  return <div className="evaluations-shell">
    <header className="evaluations-header">
      <div><span className="eyebrow">Python Evaluator V1</span><h1>机器评测结果</h1><p className="muted">查看 MongoDB evaluations 中的 verdict、路由层级和完整审计证据。</p></div>
      <button onClick={onBack}>← 返回数学测试台</button>
    </header>
    {error && <div role="alert" className="error">{error}</div>}
    <section className="evaluation-summary">
      <div><small>当前筛选</small><strong>{data?.summary.total ?? "—"}</strong></div>
      {Object.entries(data?.summary.verdicts || {}).sort().map(([name, count]) => <div key={name}><small>{verdictLabel[name] || name}</small><strong>{count}</strong></div>)}
    </section>
    <section className="card evaluation-filters">
      <label>Verdict<select value={verdict} onChange={event => resetPage(setVerdict)(event.target.value)}><option value="">全部</option><option value="correct">正确</option><option value="incorrect">错误</option><option value="review">待复核</option><option value="unresolved">未解决</option></select></label>
      <label>Level<select value={level} onChange={event => resetPage(setLevel)(event.target.value)}><option value="">全部</option><option value="0">Level 0</option><option value="1">Level 1</option><option value="2">Level 2</option><option value="3">Level 3</option></select></label>
      <label>Method<select value={method} onChange={event => resetPage(setMethod)(event.target.value)}><option value="">全部</option><option value="math_verify">Math-Verify</option><option value="llm_multi_part">多小问 Judge</option><option value="llm_semantic">语义 Judge</option><option value="runtime_precheck">运行预检</option></select></label>
      <label>Question ID<input value={questionId} onChange={event => resetPage(setQuestionId)(event.target.value)} placeholder="MATH-001"/></label>
      <label>Run ID<input value={runId} onChange={event => resetPage(setRunId)(event.target.value)} placeholder="完整 runId"/></label>
    </section>
    <section className="card evaluation-table-card">
      <div className="section-top"><h2>Evaluations</h2><span className="muted">第 {data?.page || page} / {pages} 页 · 每页 {data?.limit || 25} 条</span></div>
      <div className="evaluation-table-wrap"><table className="evaluation-table"><thead><tr><th>题目 / Run</th><th>运行</th><th>评测方法</th><th>Verdict</th><th>Reason</th><th>更新时间</th></tr></thead><tbody>
        {(data?.items || []).map(item => <EvaluationRow key={item.id} item={item}/>)}</tbody></table></div>
      {!data?.items.length && <p className="muted evaluation-empty">没有匹配的 Evaluation。</p>}
      <div className="evaluation-pagination"><button disabled={page <= 1} onClick={() => setPage(value => value - 1)}>上一页</button><span>{page} / {pages}</span><button disabled={page >= pages} onClick={() => setPage(value => value + 1)}>下一页</button></div>
    </section>
  </div>;
}

function EvaluationRow({ item }: { item: MachineEvaluation }) {
  const evidence = { extraction: item.extraction, verifyConfig: item.verifyConfig, judge: item.judge, versions: { pipelineVersion: item.pipelineVersion, goldAdapterVersion: item.goldAdapterVersion, mathVerifyVersion: item.mathVerifyVersion } };
  return <><tr><td><strong>{item.questionId}</strong><small>{item.runId}</small></td><td><span className={`evaluation-status ${item.runtime.status}`}>{item.runtime.status}</span><small>{item.runtime.finishReason || "无 finishReason"}</small></td><td><strong>Level {item.math.level}</strong><small>{methodLabel[item.math.method] || item.math.method}</small></td><td><span className={`verdict ${item.math.verdict}`}>{verdictLabel[item.math.verdict] || item.math.verdict}</span><small>置信度 {Number(item.math.confidence || 0).toFixed(2)}</small></td><td><strong>{item.math.reasonCode}</strong><details><summary>查看证据</summary><pre>{JSON.stringify(evidence, null, 2)}</pre></details></td><td>{item.updatedAt ? new Date(item.updatedAt).toLocaleString("zh-CN") : "—"}</td></tr></>;
}
