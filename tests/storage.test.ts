import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, symlink } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { mongoConfig } from '../server/src/database.js';
import { QuestionImages } from '../server/src/images.js';

test('Mongo credentials use driver options, including special characters', () => {
  const config = mongoConfig({ MONGODB_DATABASE: 'math_test', MONGODB_USERNAME:'test@user', MONGODB_PASSWORD:'p@ss:/#word', MONGODB_HOST:'localhost' });
  assert.equal(config.uri, 'mongodb://localhost:27017');
  assert.equal(config.options.auth?.password, 'p@ss:/#word');
  assert.equal(config.options.authSource, 'math_test');
  assert.equal(mongoConfig({ MONGODB_DATABASE:'math_test', MONGODB_USERNAME:'test', MONGODB_PASSWORD:'pass', MONGODB_AUTH_SOURCE:'admin' }).options.authSource, 'admin');
  assert.throws(() => mongoConfig({ MONGODB_DATABASE:'test', MONGODB_USERNAME:'only-user' }));
  assert.throws(() => mongoConfig({}));
});

test('images resolve Chinese names and reject traversal, unsupported files and external symlinks', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'math-images-'));
  await mkdir(path.join(dir, 'images'));
  await writeFile(path.join(dir,'images','题目.jpg'), 'fixture');
  await writeFile(path.join(dir,'outside.jpg'), 'private');
  await symlink(path.join(dir,'outside.jpg'), path.join(dir,'images','escape.jpg'));
  const images = new QuestionImages(path.join(dir,'images'));
  assert.equal(await images.resolve('images/题目.jpg'), path.join(dir,'images','题目.jpg'));
  for (const name of ['images/../outside.jpg', '/outside.jpg', 'images/escape.jpg', 'images/not-found.jpg', 'images/unsafe.svg', 'images/..\\outside.jpg']) assert.equal(await images.resolve(name), null);
  const q = await images.decorate({ id:'测试-1', question:'题', reference_answer:'答', image_path:'images/题目.jpg' });
  assert.equal(q.image_url, '/api/questions/%E6%B5%8B%E8%AF%95-1/images/question');
  assert.equal(q.reference_answer_image_url, undefined);
});
