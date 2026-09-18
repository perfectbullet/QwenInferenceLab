import path from 'node:path';
import { realpath, stat } from 'node:fs/promises';
import type { Question } from '../../shared/types.js';

export class QuestionImages {
  constructor(readonly dir: string) {}
  async resolve(reference: string | undefined): Promise<string | null> {
    if (!reference || !reference.startsWith('images/')) return null;
    const relative = reference.slice('images/'.length);
    if (!relative || relative.includes('\\') || relative.split('/').some(p => p === '..' || p === '.') || path.isAbsolute(relative) || !/\.(jpe?g|png|webp|gif)$/i.test(relative)) return null;
    try {
      const root = await realpath(this.dir); const file = await realpath(path.join(root, relative));
      if (!file.startsWith(root + path.sep) || !(await stat(file)).isFile()) return null;
      return file;
    } catch (e) {
      if (['ENOENT','ENOTDIR','EACCES'].includes((e as NodeJS.ErrnoException).code || '')) return null;
      throw e;
    }
  }
  async decorate(question: Question): Promise<Question> {
    const [image, answer] = await Promise.all([this.resolve(question.image_path), this.resolve(question.reference_answer_image_path)]);
    const base = `/api/questions/${encodeURIComponent(question.id)}/images`;
    return { ...question, image_url: image ? `${base}/question` : undefined, reference_answer_image_url: answer ? `${base}/answer` : undefined };
  }
}
