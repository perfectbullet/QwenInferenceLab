import { useState } from 'react';
import './images.css';
export function SourceImage({ url, reference, title }: { url?: string; reference?: string; title: string }) {
  const [failed, setFailed] = useState(false);
  if (!reference) return null;
  if (!url || failed) return <p className="asset-note">{title}未找到：{reference}</p>;
  return <details className="source-image"><summary>{title} <span>点击查看，点击图片放大</span></summary>
    <a href={url} target="_blank" rel="noreferrer"><img src={url} alt={title} loading="lazy" onError={() => setFailed(true)}/></a>
  </details>;
}
