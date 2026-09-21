# Qwen 数学测试台

本机逐题测试，固定 CoT 思考模式，思考过程默认折叠且可展开，流式渲染数学解答并保存历史。解答期间可切题浏览，正文支持复制原始 Markdown。

```bash
npm ci
# 首次配置 .env 后，导入题库和旧历史（可重复执行，不覆盖已有数据）
npm run db:migrate
npm run dev
```

打开 http://127.0.0.1:5173 。默认连接 `192.168.8.231:8200` 的 `qwen38-27b`。

侧栏可管理并下拉选择模型 URL/名称，配置与当前选择保存到 MongoDB。`.env` 仅初始化首个配置；每次回答保存实际使用的 URL 和模型名称快照。

题库、运行历史、评价存入 MongoDB；配置参见 `.env.example`。原图保留在根目录 `images/`，页面可预览；迁移前的 JSON 文件仍保留，新运行不再写入文件。

- [第一版使用说明](docs/第一版使用说明.md)
- [架构与开发者导览（面向 Python 开发者）](docs/架构与开发者导览.md)
- [后续开发计划](docs/后续开发计划.md)
- [前后端开发路线图](docs/前后端开发路线图.md)

部署与性能文档已合并为以下四个入口；被合并的原始记录保存在 `docs/archive/2026-历史记录/`：

- [Qwen3.6-35B-A3B：RTX 5090 单卡部署说明](docs/Qwen3.6-35B-A3B-Docker+Python-Frontend+Text-Only+RTX5090部署说明.md)
- [Qwen3.8-27B 部署与运维手册](docs/Qwen3.8-27B部署与运维手册.md)
- [推测解码与性能实验总结](docs/推测解码与性能实验总结.md)
- [推理评测与下一阶段路线图](docs/推理评测与下一阶段路线图.md)

`npm test` 测试；`npm run build` 类型检查并构建；`npm start` 提供构建后的页面和 API（3100 端口）。
