# Merchant Growth Copilot (MVP)

面向商家经营诊断的 Copilot MVP。当前版本支持品牌维度分析、意图识别、无关问题拦截、受控动作建议，以及前端双模式演示（Mock / Real LLM）。

## 当前能力
- 品牌维度分析：输入 `brand_id`，只基于该品牌聚合数据进行分析与回答。
- 意图识别：`diagnosis / root_cause / action_recommendation / irrelevant`。
- 无关问题拦截：识别到非经营问题时直接返回引导文案，不进入业务分析模块。
- 受控分析输出：动作建议受框架约束，不自由发挥。
- 前端演示双模式（`Mock Demo` 纯前端本地演示，`Real LLM` 通过本地 API 调用后端真实模型链路）。

## 项目结构
- `merchant_growth_mvp.py`：主逻辑（数据、分析、意图识别、LLM 编排、品牌聚合）。
- `MVP.py`：命令行演示入口（品牌维度）。
- `MVP_test_cases.ipynb`：测试与交互 notebook（含 API key 检查、品牌输入、回归 case）。
- `copilot_frontend.html`：单文件前端页面（品牌输入 + Mock/Real 模式开关）。
- `copilot_server.py`：本地服务，提供静态页面和 `POST /api/analyze` 接口。
- `MVP.ipynb`：历史 notebook（保留演示和过程）。
- `.env.example`：环境变量示例。

## 环境配置
推荐将真实 key 放在仓库外，避免误提交。

1. 在 `~/.merchant-agent/.env` 放置：
```env
OPENAI_API_KEY=your_key
# 可选：网络受限时配置
# OPENAI_BASE_URL=https://your-proxy-or-gateway/v1
```
2. 可选设置（优先级最高）：
```bash
export MERCHANT_AGENT_ENV_PATH="$HOME/.merchant-agent/.env"
```

读取优先级为：
1. `MERCHANT_AGENT_ENV_PATH`
2. `~/.merchant-agent/.env`
3. 项目目录下 `.env`

## 运行方式

### 方式 A：Notebook 测试与回归
```bash
cd "/Users/pro/Desktop/merchant agent"
jupyter notebook
```
打开 `MVP_test_cases.ipynb`，先运行 `API Key Check`，再运行 `Interactive Run`。

### 方式 B：命令行快速演示
```bash
cd "/Users/pro/Desktop/merchant agent"
python MVP.py
```

### 方式 C：前端页面（推荐演示）
```bash
cd "/Users/pro/Desktop/merchant agent"
python copilot_server.py
```
浏览器打开：
`http://127.0.0.1:8765/copilot_frontend.html`

页面内可切换：
- `Mock Demo`：前端本地模拟，不调用后端 LLM。
- `Real LLM`：调用 `/api/analyze`，走真实模型链路。

## API 说明
`POST /api/analyze`

请求体示例：
```json
{
  "brand_id": "B001",
  "question": "给我一些提升订单的建议",
  "mock_mode": false
}
```

## 常见问题
- `500` 报错：先确认当前 Python 环境安装了 `openai`，再确认 `.env` 中 `OPENAI_API_KEY` 有效；若报 `APIConnectionError`，可配置 `OPENAI_BASE_URL`。
- `Real LLM` 无响应：确认页面是从 `http://127.0.0.1:8765/...` 打开而不是 `file://`，并确认 `copilot_server.py` 进程正在运行且终端未退出。
