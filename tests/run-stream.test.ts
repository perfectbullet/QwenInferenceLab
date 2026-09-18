import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { runStream } from '../server/src/run-stream.js';

test('browser disconnect leaves task alive; completion still saves without an observer', async () => {
  const controller = new AbortController();
  let release!: () => void;
  const continueTask = new Promise<void>(resolve => { release = resolve; });
  let disconnected!: () => void;
  const closed = new Promise<void>(resolve => { disconnected = resolve; });
  let finish!: () => void;
  const finished = new Promise<void>(resolve => { finish = resolve; });
  let saved = false;
  const server = createServer(async (_req, response) => {
    response.writeHead(200, { 'Content-Type': 'text/event-stream' });
    const stream = runStream(response);
    response.once('close', disconnected);
    stream.emit({ type: 'started', run: { id: 'same-run' } });
    await continueTask;
    assert.equal(controller.signal.aborted, false);
    stream.emit({ type: 'reasoning_delta', text: 'still thinking' });
    stream.emit({ type: 'answer_delta', text: '42' });
    saved = true;
    stream.emit({ type: 'done' });
    stream.end();
    finish();
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  try {
    const address = server.address() as { port: number };
    const response = await fetch(`http://127.0.0.1:${address.port}`);
    const reader = response.body!.getReader();
    assert.match(new TextDecoder().decode((await reader.read()).value), /same-run/);
    await reader.cancel();
    await closed;
    release();
    await finished;
    assert.equal(saved, true);
    // Explicit cancellation remains independent of the subscriber transport.
    controller.abort('cancelled');
    assert.equal(controller.signal.reason, 'cancelled');
  } finally {
    release();
    server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
  }
});
