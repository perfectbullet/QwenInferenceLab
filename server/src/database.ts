import { MongoClient, type MongoClientOptions } from 'mongodb';

export function mongoConfig(env: NodeJS.ProcessEnv = process.env) {
  const database = env.MONGODB_DATABASE;
  if (!database || /[\s/\\.\"$]/.test(database)) throw new Error('请设置有效的 MONGODB_DATABASE');
  const username = env.MONGODB_USERNAME;
  const password = env.MONGODB_PASSWORD;
  if (!env.MONGODB_URI && (!username !== !password)) throw new Error('MONGODB_USERNAME 与 MONGODB_PASSWORD 必须同时配置');
  const port = Number(env.MONGODB_PORT || 27017);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('MONGODB_PORT 无效');
  const host = env.MONGODB_HOST || '127.0.0.1';
  if (!env.MONGODB_URI && /[\s/@?#]/.test(host)) throw new Error('MONGODB_HOST 应为主机名或 IP');
  const uri = env.MONGODB_URI || `mongodb://${host}:${port}`;
  const options: MongoClientOptions = {
    appName: 'qwen-math-lab', maxPoolSize: 10, serverSelectionTimeoutMS: 5000,
    connectTimeoutMS: 5000, socketTimeoutMS: 15000,
    ...(!env.MONGODB_URI && username && password ? {
      auth: { username, password }, authSource: env.MONGODB_AUTH_SOURCE || database,
    } : {}),
  };
  return { uri, options, database };
}

export async function connectDatabase() {
  const config = mongoConfig();
  let client: MongoClient | undefined;
  try {
    client = new MongoClient(config.uri, config.options);
    await client.connect();
    const db = client.db(config.database); await db.command({ ping: 1 });
    return { client, db };
  } catch (error) {
    await client?.close();
    const code = (error as { code?: number }).code;
    throw new Error(code === 18 ? 'MongoDB 认证失败，请检查账号、密码和 MONGODB_AUTH_SOURCE' : 'MongoDB 连接失败，请检查地址、网络和数据库权限');
  }
}
