// Incremental UTF-8 and SSE decoder. Comments, CRLF and multi-line data are supported.
export async function* readSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = ''; let lines: string[] = [];
  function line(value: string): string | undefined {
    if (value === '') {
      if (!lines.length) return;
      const event = lines.join('\n'); lines = []; return event;
    }
    if (value.startsWith('data:')) lines.push(value.slice(5).replace(/^ /, ''));
  }
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let end: number;
      while ((end = buffer.indexOf('\n')) >= 0) {
        const event = line(buffer.slice(0, end).replace(/\r$/, ''));
        buffer = buffer.slice(end + 1);
        if (event !== undefined) yield event;
      }
      if (done) {
        if (buffer) line(buffer.replace(/\r$/, ''));
        const event = line(''); if (event !== undefined) yield event;
        break;
      }
    }
  } finally { reader.releaseLock(); }
}
