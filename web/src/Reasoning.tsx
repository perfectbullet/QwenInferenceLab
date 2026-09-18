import { useState } from 'react';
export function Reasoning({ text, running }: { text?: string; running: boolean }) {
  const [open, setOpen] = useState(false);
  return <details className="reasoning" onToggle={e => setOpen(e.currentTarget.open)}>
    <summary>思考过程 <span>默认折叠 · 点击展开</span></summary>
    {open && <pre>{text || (running ? '等待思考内容…' : '此记录未保存思考过程')}</pre>}
  </details>;
}
