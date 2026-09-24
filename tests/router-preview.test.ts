import assert from 'node:assert/strict';
import test from 'node:test';
import { RouterRuntimeClient, validateRouterPreviewRequest } from '../server/src/router-preview.js';

test('validateRouterPreviewRequest trims and defaults profile', () => {
  assert.deepEqual(validateRouterPreviewRequest({ question: '  1 + 1  ' }), {
    question: '1 + 1',
    profile: 'development',
  });
  assert.throws(
    () => validateRouterPreviewRequest({ question: '', profile: 'development' }),
    /题目长度/,
  );
  assert.throws(
    () => validateRouterPreviewRequest({ question: 'x', profile: 'unknown' }),
    /profile/,
  );
});

test('RouterRuntimeClient forwards preview request', async () => {
  const originalFetch = globalThis.fetch;
  let captured = '';
  globalThis.fetch = async (_url, init) => {
    captured = String(init?.body);
    return new Response(JSON.stringify({
      profile: 'development',
      decision: 'local',
      score: 1,
      experimental: true,
      reasonCodes: ['LOCAL_THRESHOLDS_PASSED'],
      features: {},
      neighbors: [],
      policyConfig: {},
      routerVersion: 'router-v1',
      embeddingVersion: 'qwen3-embedding-0.6b-v1',
      embeddingModel: 'Qwen/Qwen3-Embedding-0.6B',
      corpusSize: 250,
      dimension: 1024,
      textHash: 'hash',
      durationMs: 1,
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };
  try {
    const result = await new RouterRuntimeClient('http://router.test', 1000).preview({
      question: 'x',
      profile: 'development',
    });
    assert.equal(result.decision, 'local');
    assert.deepEqual(JSON.parse(captured), { question: 'x', profile: 'development' });
  } finally {
    globalThis.fetch = originalFetch;
  }
});
