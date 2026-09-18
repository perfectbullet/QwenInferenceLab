import Markdown from 'react-markdown';
import { memo } from 'react';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { normalizeMath, splitPendingMath } from './math';
export const MathText = memo(function MathText({ text }: { text: string }) {
  const [complete, pending] = splitPendingMath(normalizeMath(text));
  return <div className="markdown"><Markdown remarkPlugins={[remarkMath]} rehypePlugins={[[rehypeKatex, { strict: false, trust: false, throwOnError: false }]]} skipHtml>{complete}</Markdown>{pending && <span style={{ whiteSpace: 'pre-wrap' }}>{pending}</span>}</div>;
});
