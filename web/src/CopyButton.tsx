import { useState } from 'react';
export function CopyButton({ text, label }: { text: string; label: string }) {
  const [message, setMessage] = useState('');
  async function copy() {
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(text);
      else {
        const input = document.createElement('textarea'); input.value = text;
        input.style.position = 'fixed'; input.style.opacity = '0'; document.body.append(input);
        try { input.select(); if (!document.execCommand('copy')) throw new Error('copy'); }
        finally { input.remove(); }
      }
      setMessage('已复制');
    } catch { setMessage('复制失败，请重试'); }
  }
  return <button className="copy-button" aria-label={`复制${label} Markdown`} disabled={!text} onClick={copy}>{message || '复制 Markdown'}</button>;
}
