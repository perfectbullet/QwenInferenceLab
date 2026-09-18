import 'dotenv/config';
import test from 'node:test';
import assert from 'node:assert/strict';
import { connectDatabase } from '../server/src/database.js';
import { QuestionStore, RunStore } from '../server/src/store.js';
import { defaults, type Run } from '../shared/types.js';
import { ModelStore } from '../server/src/models.js';

test('real MongoDB: idempotent import, persistence, review and interrupted recovery', { skip: process.env.MONGODB_INTEGRATION !== '1' }, async () => {
  const { client, db } = await connectDatabase();
  const prefix = `test_${crypto.randomUUID().replaceAll('-','')}`;
  const created: string[] = [];
  try {
    for (const suffix of ['runs', 'questions', 'model_configs', 'model_settings']) {
      const name = `${prefix}_${suffix}`;
      await db.createCollection(name); created.push(name);
    }
    const runs = new RunStore(db, `${prefix}_runs`);
    const questions = new QuestionStore(db, `${prefix}_questions`);
    await runs.initialize(); await questions.initialize();
    const q = { id:'test-q', question:'1+1', reference_answer:'2', image_path:'images/题目.jpg' };
    assert.equal(await questions.importMissing([q]), 1);
    assert.equal(await questions.importMissing([{ ...q, reference_answer:'do not overwrite' }]), 0);
    assert.equal((await questions.get(q.id))?.reference_answer, '2');
    assert.equal(await questions.count(), 1);
    const r: Run = { id:crypto.randomUUID(), questionId:q.id, question:q.question, startedAt:new Date().toISOString(), model:'test', systemPrompt:'', params:defaults, request:{}, answer:'$2$', status:'completed', finishReason:'stop', firstResponseMs:1, thinkingMs:2, totalMs:3, thinkingNote:'test', usage:null, evaluation:'unreviewed', notes:'' };
    const models = new ModelStore(db, `${prefix}_`);
    await models.initialize({baseUrl:'http://localhost:8200/v1',modelName:'first'});
    const added = await models.save({baseUrl:'http://localhost:8300/v1',modelName:'second'});
    await models.select(added.id);
    await models.initialize({baseUrl:'http://localhost:9999/v1',modelName:'do-not-override'});
    assert.equal((await models.list()).selectedId, added.id);
    assert.equal((await models.list()).models.length, 2);
    r.modelBaseUrl = added.baseUrl; r.modelName = added.modelName; r.modelConfigId = added.id;
    assert.equal(await runs.importMissing(r), 1);
    await models.save({baseUrl:'http://localhost:8400/v1',modelName:'edited'}, added.id);
    assert.equal((await runs.get(r.id))?.modelBaseUrl, 'http://localhost:8300/v1');
    assert.equal((await runs.get(r.id))?.modelName, 'second');
    await runs.review(r.id, 'correct', '持久化评价');
    assert.equal(await runs.importMissing(r), 0);
    const reloaded = await new RunStore(db, `${prefix}_runs`).get(r.id);
    assert.equal(reloaded?.evaluation, 'correct'); assert.equal(reloaded?.notes, '持久化评价');
    assert.ok(!('_id' in reloaded!));
    const unfinished = { ...r, id:crypto.randomUUID(), status:'running' as const };
    await runs.save(unfinished);
    assert.equal(await runs.review(unfinished.id, 'incorrect', 'should not save'), null);
    await runs.recover();
    assert.equal((await runs.get(unfinished.id))?.status, 'interrupted');
    assert.equal((await runs.list()).length, 2);
    assert.equal((await runs.get(r.id))?.status, 'completed');
  } finally {
    // Only remove the uniquely named collections created by this test.
    try { for (const name of created) await db.collection(name).drop(); }
    finally { await client.close(); }
  }
});
