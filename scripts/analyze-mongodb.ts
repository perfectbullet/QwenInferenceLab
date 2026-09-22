/**
 * Read-only MongoDB profiling for the math question bank and answer runs.
 * Outputs are intentionally aggregate-only: neither questions nor answers are exported.
 */
import 'dotenv/config';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { connectDatabase } from '../server/src/database.js';

type Count = { value: string; count: number };
const outputDir = path.resolve('reports');
const generatedAt = () => new Date().toISOString();

function safeValue(value: unknown) {
  if (value === undefined || value === null || value === '') return '(missing_or_empty)';
  return String(value);
}

function summarize(values: number[]) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  const at = (ratio: number) => sorted.length ? sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * ratio) - 1)] : null;
  const total = sorted.reduce((sum, value) => sum + value, 0);
  return {
    count: sorted.length,
    min: sorted[0] ?? null,
    p50: at(0.5), p90: at(0.9), p95: at(0.95), max: sorted.at(-1) ?? null,
    mean: sorted.length ? Number((total / sorted.length).toFixed(2)) : null,
  };
}

async function distribution(collection: any, expression: unknown, match: Record<string, unknown> = {}, limit = 100): Promise<Count[]> {
  return collection.aggregate([
    { $match: match },
    { $group: { _id: expression, count: { $sum: 1 } } },
    { $project: { _id: 0, value: { $ifNull: ['$_id', '(missing_or_empty)'] }, count: 1 } },
    { $sort: { count: -1, value: 1 } }, { $limit: limit },
  ]).toArray();
}

async function fieldCompleteness(collection: any, fields: string[]) {
  const result: Record<string, unknown> = {};
  for (const field of fields) {
    const [present, typeDistribution] = await Promise.all([
      collection.countDocuments({ [field]: { $exists: true, $nin: [null, ''] } }),
      distribution(collection, { $type: `$${field}` }),
    ]);
    result[field] = { present_count: present, missing_or_empty_count: await collection.countDocuments({ $or: [{ [field]: { $exists: false } }, { [field]: null }, { [field]: '' }] }), type_distribution: typeDistribution };
  }
  return result;
}

async function main() {
  const { client, db } = await connectDatabase();
  try {
    const questions = db.collection('questions');
    const runs = db.collection('runs');
    const [questionCount, runCount, questionDates, runDates] = await Promise.all([
      questions.countDocuments(), runs.countDocuments(),
      questions.aggregate([{ $group: { _id: null, ids: { $push: '$id' } } }]).toArray(),
      runs.aggregate([{ $group: { _id: null, earliest: { $min: '$startedAt' }, latest: { $max: '$startedAt' } } }]).toArray(),
    ]);

    const [questionTextLengths, referenceLengths, questionImageCount, answerImageCount] = await Promise.all([
      questions.aggregate([{ $project: { length: { $strLenCP: { $ifNull: ['$question', ''] } } } }]).toArray(),
      questions.aggregate([{ $project: { length: { $strLenCP: { $ifNull: ['$reference_answer', ''] } } } }]).toArray(),
      questions.countDocuments({ image_path: { $exists: true, $nin: [null, ''] } }),
      questions.countDocuments({ reference_answer_image_path: { $exists: true, $nin: [null, ''] } }),
    ]);
    const questionReport = {
      report_type: 'mongodb_math_question_bank_statistics', generated_at: generatedAt(),
      database: db.databaseName, collection: 'questions', total_documents: questionCount,
      id_overview: { distinct_id_count: new Set((questionDates[0]?.ids || []).filter(Boolean)).size },
      field_completeness: await fieldCompleteness(questions, ['id', 'question', 'reference_answer', 'tag', 'difficulty', 'math_type', 'image_path', 'reference_answer_image_path']),
      category_distribution: {
        tag: await distribution(questions, { $ifNull: ['$tag', '(missing_or_empty)'] }),
        difficulty: await distribution(questions, { $ifNull: ['$difficulty', '(missing_or_empty)'] }),
        math_type: await distribution(questions, { $ifNull: ['$math_type', '(missing_or_empty)'] }),
      },
      content_length_characters: { question: summarize(questionTextLengths.map((x: any) => x.length)), reference_answer: summarize(referenceLengths.map((x: any) => x.length)) },
      image_coverage: { question_image_count: questionImageCount, reference_answer_image_count: answerImageCount, no_question_image_count: questionCount - questionImageCount, no_reference_answer_image_count: questionCount - answerImageCount },
      notes: ['Text lengths are Unicode character counts.', 'This report contains aggregate statistics only; it does not include question or reference-answer content.'],
    };

    const numericRunFields = ['totalMs', 'firstResponseMs', 'thinkingMs'];
    const numericStats: Record<string, unknown> = {};
    for (const field of numericRunFields) {
      const rows = await runs.find({ [field]: { $type: 'number' } }, { projection: { [field]: 1, _id: 0 } }).toArray();
      numericStats[field] = summarize(rows.map((row: any) => row[field]));
    }
    const usage = await runs.aggregate([{ $project: {
      prompt_tokens: { $convert: { input: '$usage.prompt_tokens', to: 'double', onError: null, onNull: null } },
      completion_tokens: { $convert: { input: '$usage.completion_tokens', to: 'double', onError: null, onNull: null } },
      total_tokens: { $convert: { input: '$usage.total_tokens', to: 'double', onError: null, onNull: null } },
    } }]).toArray();
    const runQuestionIds = await runs.distinct('questionId', { questionId: { $type: 'string' } });
    const categoryByRun = await runs.aggregate([
      { $lookup: { from: 'questions', localField: 'questionId', foreignField: 'id', as: 'question_metadata' } },
      { $unwind: { path: '$question_metadata', preserveNullAndEmptyArrays: true } },
      { $facet: {
        tag: [{ $group: { _id: { $ifNull: ['$question_metadata.tag', '(question_not_found_or_tag_missing)'] }, count: { $sum: 1 } } }, { $sort: { count: -1, _id: 1 } }],
        difficulty: [{ $group: { _id: { $ifNull: ['$question_metadata.difficulty', '(question_not_found_or_difficulty_missing)'] }, count: { $sum: 1 } } }, { $sort: { count: -1, _id: 1 } }],
        math_type: [{ $group: { _id: { $ifNull: ['$question_metadata.math_type', '(question_not_found_or_math_type_missing)'] }, count: { $sum: 1 } } }, { $sort: { count: -1, _id: 1 } }],
      } },
    ]).toArray();
    const normalizeFacet = (rows: any[]) => rows.map(({ _id, count }) => ({ value: safeValue(_id), count }));
    const runReport = {
      report_type: 'mongodb_math_run_statistics', generated_at: generatedAt(), database: db.databaseName, collection: 'runs', total_documents: runCount,
      time_range_started_at: { earliest: runDates[0]?.earliest ?? null, latest: runDates[0]?.latest ?? null },
      field_completeness: await fieldCompleteness(runs, ['id', 'questionId', 'startedAt', 'model', 'modelName', 'modelBaseUrl', 'status', 'finishReason', 'answer', 'reasoning', 'error', 'usage', 'evaluation', 'notes']),
      status_distribution: await distribution(runs, { $ifNull: ['$status', '(missing_or_empty)'] }),
      evaluation_distribution: await distribution(runs, { $ifNull: ['$evaluation', '(missing_or_empty)'] }),
      finish_reason_distribution: await distribution(runs, { $ifNull: ['$finishReason', '(missing_or_empty)'] }),
      model_distribution: { model: await distribution(runs, { $ifNull: ['$model', '(missing_or_empty)'] }), model_name: await distribution(runs, { $ifNull: ['$modelName', '(missing_or_empty)'] }), model_base_url: await distribution(runs, { $ifNull: ['$modelBaseUrl', '(missing_or_empty)'] }) },
      parameter_distribution: { temperature: await distribution(runs, { $ifNull: ['$params.temperature', '(missing_or_empty)'] }), top_p: await distribution(runs, { $ifNull: ['$params.top_p', '(missing_or_empty)'] }), top_k: await distribution(runs, { $ifNull: ['$params.top_k', '(missing_or_empty)'] }), min_p: await distribution(runs, { $ifNull: ['$params.min_p', '(missing_or_empty)'] }), presence_penalty: await distribution(runs, { $ifNull: ['$params.presence_penalty', '(missing_or_empty)'] }), repetition_penalty: await distribution(runs, { $ifNull: ['$params.repetition_penalty', '(missing_or_empty)'] }), max_tokens: await distribution(runs, { $ifNull: ['$params.max_tokens', '(missing_or_empty)'] }) },
      duration_milliseconds: numericStats,
      usage_tokens: { prompt_tokens: summarize(usage.map((x: any) => x.prompt_tokens)), completion_tokens: summarize(usage.map((x: any) => x.completion_tokens)), total_tokens: summarize(usage.map((x: any) => x.total_tokens)) },
      question_coverage: { distinct_question_id_count: runQuestionIds.length, question_bank_count: questionCount, question_bank_coverage_ratio: questionCount ? Number((runQuestionIds.length / questionCount).toFixed(6)) : null, runs_by_question_category: Object.fromEntries(Object.entries(categoryByRun[0] || {}).map(([key, rows]) => [key, normalizeFacet(rows as any[])])) },
      notes: ['Durations are milliseconds. Percentiles use nearest-rank calculation.', 'Run-to-question category statistics are computed by questionId lookup; unmatched records are explicitly labeled.', 'This report contains aggregate statistics only; answer, reasoning, request, notes and error contents are not exported.'],
    };
    await mkdir(outputDir, { recursive: true });
    await Promise.all([
      writeFile(path.join(outputDir, 'mongodb-question-statistics.json'), `${JSON.stringify(questionReport, null, 2)}\n`),
      writeFile(path.join(outputDir, 'mongodb-run-statistics.json'), `${JSON.stringify(runReport, null, 2)}\n`),
    ]);
    console.log(JSON.stringify({ ok: true, questions: questionCount, runs: runCount, output: outputDir }));
  } finally { await client.close(); }
}

main().catch((error) => { console.error(error instanceof Error ? error.message : error); process.exitCode = 1; });
