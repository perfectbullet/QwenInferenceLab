import type { Collection, Db } from 'mongodb';
import type { Question, Run } from '../../shared/types.js';
export class RunStore {
  readonly collection: Collection<Run>;
  constructor(db: Db, name = 'runs') { this.collection = db.collection<Run>(name); }
  async initialize() {
    await this.collection.createIndex({ id: 1 }, { unique: true });
    await this.collection.createIndex({ questionId: 1, startedAt: -1 });
    await this.collection.createIndex({ startedAt: -1 });
  }
  async save(run: Run) {
    await this.collection.replaceOne({ id: run.id }, run, { upsert: true });
  }
  async importMissing(run: Run) {
    return (await this.collection.updateOne({ id: run.id }, { $setOnInsert: run }, { upsert: true })).upsertedCount;
  }
  async list(): Promise<Run[]> {
    return this.collection.find({}, { projection: { _id: 0 } }).sort({ startedAt: -1 }).toArray();
  }
  async get(id: string): Promise<Run | null> {
    return this.collection.findOne({ id }, { projection: { _id: 0 } });
  }
  async review(id: string, evaluation: Run['evaluation'], notes: string) {
    return this.collection.findOneAndUpdate({ id, status: { $ne: 'running' } }, { $set: { evaluation, notes } }, { returnDocument: 'after', projection: { _id: 0 } });
  }
  async recover() {
    await this.collection.updateMany({ status: 'running' }, { $set: {
      status: 'interrupted', error: '服务重启，运行被中断；耗时为最后一次保存的观测值。',
    } });
  }
}

export class QuestionStore {
  readonly collection: Collection<Question & { sortOrder?: number }>;
  constructor(db: Db, name = 'questions') { this.collection = db.collection(name); }
  async initialize() { await this.collection.createIndex({ id: 1 }, { unique: true }); }
  async count() { return this.collection.countDocuments(); }
  async list(): Promise<Question[]> {
    return this.collection.find({}, { projection: { _id: 0, sortOrder: 0 } }).sort({ sortOrder: 1, id: 1 }).toArray();
  }
  async get(id: string): Promise<Question | null> {
    return this.collection.findOne({ id }, { projection: { _id: 0, sortOrder: 0 } });
  }
  async importMissing(questions: Question[]) {
    if (!questions.length) return 0;
    const result = await this.collection.bulkWrite(questions.map((q, sortOrder) => ({ updateOne: {
      filter: { id: q.id }, update: { $setOnInsert: { ...q, sortOrder } }, upsert: true,
    } })));
    return result.upsertedCount;
  }
}
