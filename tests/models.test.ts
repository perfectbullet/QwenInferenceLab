import test from 'node:test';
import assert from 'node:assert/strict';
import { validateModel } from '../server/src/models.js';
test('model configuration normalizes URLs and rejects invalid/credential-bearing URLs', () => {
  assert.deepEqual(validateModel({baseUrl:' http://localhost:8200/v1/ ',modelName:' model-a '}),{baseUrl:'http://localhost:8200/v1',modelName:'model-a'});
  for (const baseUrl of ['file:///tmp/a','http://user:pass@example.com/v1','http://host/v1?key=secret','not-a-url']) assert.throws(()=>validateModel({baseUrl,modelName:'a'}));
  assert.throws(()=>validateModel({baseUrl:'http://localhost/v1',modelName:' '}));
});
