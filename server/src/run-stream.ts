import type { ServerResponse } from 'node:http';

// The browser is only an observer. Disconnecting must never cancel inference.
export function runStream(response: ServerResponse) {
  let connected = !response.destroyed && !response.writableEnded;
  const detach = () => { connected = false; clearInterval(heartbeat); };
  const write = (text: string) => {
    if (!connected || response.destroyed || response.writableEnded) return;
    try { response.write(text); } catch { detach(); }
  };
  const heartbeat = setInterval(() => write(': heartbeat\n\n'), 10000);
  response.once('close', detach);
  response.on('error', detach);
  return {
    emit: (event: Record<string, unknown>) => write(`data: ${JSON.stringify(event)}\n\n`),
    end: () => {
      detach();
      response.off('close', detach);
      if (!response.destroyed && !response.writableEnded) response.end();
    },
  };
}
