import type { Collection, Db, Document, Filter } from "mongodb";
import type { EvaluationListResponse, MachineEvaluation } from "../../shared/types.js";

export interface EvaluationQuery {
  page?: number | string; limit?: number | string; verdict?: string; level?: number | string;
  method?: string; questionId?: string; runId?: string;
}

function counts(rows: Document[]) {
  return Object.fromEntries(rows.map(row => [String(row._id), Number(row.count)]));
}

export class EvaluationStore {
  readonly collection: Collection<MachineEvaluation>;
  constructor(db: Db, name = "evaluations") { this.collection = db.collection<MachineEvaluation>(name); }

  async list(query: EvaluationQuery): Promise<EvaluationListResponse> {
    const page = Math.max(1, Number(query.page) || 1);
    const limit = Math.min(100, Math.max(1, Number(query.limit) || 25));
    const filter: Filter<MachineEvaluation> = {};
    if (query.verdict) filter["math.verdict"] = query.verdict;
    if (query.method) filter["math.method"] = query.method;
    if (query.level !== undefined && query.level !== "") filter["math.level"] = Number(query.level);
    if (query.questionId) filter.questionId = query.questionId.trim();
    if (query.runId) filter.runId = query.runId.trim();
    const [items, total, verdicts, levels, methods] = await Promise.all([
      this.collection.find(filter, { projection: { _id: 0, apiKey: 0, "judge.apiKey": 0 } }).sort({ updatedAt: -1, createdAt: -1 }).skip((page - 1) * limit).limit(limit).toArray(),
      this.collection.countDocuments(filter),
      this.collection.aggregate([{ $match: filter }, { $group: { _id: "$math.verdict", count: { $sum: 1 } } }]).toArray(),
      this.collection.aggregate([{ $match: filter }, { $group: { _id: "$math.level", count: { $sum: 1 } } }]).toArray(),
      this.collection.aggregate([{ $match: filter }, { $group: { _id: "$math.method", count: { $sum: 1 } } }]).toArray(),
    ]);
    return { items, total, page, limit, summary: { total, verdicts: counts(verdicts), levels: counts(levels), methods: counts(methods) } };
  }
}
