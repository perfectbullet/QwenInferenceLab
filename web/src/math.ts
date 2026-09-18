// Protect code and escaped backslashes before translating alternative math delimiters.
export function normalizeMath(source: string): string {
  return source.replace(/(`{3,}|~{3,})[^\n]*\n[\s\S]*?(?:\1|$)|(`+)[^`]*?\2|\\\\|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]/g, token => {
    if (token.startsWith('\\(')) return `$${token.slice(2, -2)}$`;
    if (token.startsWith('\\[')) return `\n\n$$\n${token.slice(2, -2).trim()}\n$$\n\n`;
    return token;
  });
}

// remark-math accepts unfinished display blocks; keep the unfinished tail literal
// instead so a streamed formula only enters KaTeX after its closing delimiter.
export function splitPendingMath(source: string): [string, string] {
  for (let i = 0; i < source.length;) {
    if (source[i] === '`' || (source[i] === '~' && source.slice(i, i + 3) === '~~~')) {
      const char = source[i]; let end = i;
      while (source[end] === char) end++;
      const marker = source.slice(i, end); const close = source.indexOf(marker, end);
      i = close < 0 ? source.length : close + marker.length; continue;
    }
    if (source.slice(i, i + 2) === '\\(' || source.slice(i, i + 2) === '\\[') return [source.slice(0, i), source.slice(i)];
    if (source[i] === '\\') { i += 2; continue; }
    if (source[i] !== '$') { i++; continue; }
    const marker = source[i + 1] === '$' ? '$$' : '$';
    let end = i + marker.length; let closed = false;
    while (end < source.length) {
      if (source[end] === '\\') { end += 2; continue; }
      if (source.slice(end, end + marker.length) === marker) { closed = true; break; }
      end++;
    }
    if (!closed) return [source.slice(0, i), source.slice(i)];
    i = end + marker.length;
  }
  return [source, ''];
}
