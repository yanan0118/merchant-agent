# Merchant Growth Copilot (MVP)

## 30 秒看懂
这是一个商家增长 Copilot：Rule 负责证据组织与 fallback，LLM 负责分析决策与建议生成。  
系统支持 `Mock Demo` 和 `Real LLM` 双模式；真实链路返回稳定的结构化 schema（assessment / diagnoses / actions / talking_points），可直接用于前端渲染与测试。

面向商家经营分析的本地 Copilot Demo，当前版本采用：
- Rule 负责证据组织（metrics / peer comparison / fallback）
- LLM 负责分析与决策（planner + analyst）
- 前端支持 `Mock Demo` 和 `Real LLM` 双模式

## 当前架构
- `question -> planner (LLM优先, rule fallback) -> evidence -> analyst (LLM优先, fallback) -> 前端渲染`
- 品牌维度分析：输入 `brand_id`，按品牌聚合数据分析。
- 无关问题拦截：判定为 `analysis_type=other` 时直接返回引导文案。

## Planner Schema（开放内容，稳定接口）
后端 planner 输出：
```json
{
  "analysis_type": "diagnosis | root_cause | action_planning | risk_review | opportunity_scan | other",
  "analysis_goal": "...",
  "focus_areas": ["traffic | conversion | ops_readiness | user_experience | supply_quality | other"],
  "suggested_evidence_priority": ["orders | impressions | conversion_rate | image_coverage | open_hours | accept_time_mins | prep_time_mins | merchant_cancel_rate | active_spu_count | discount_rate"],
  "reason": "..."
}
```
约束：
- `analysis_type` 必须在白名单
- `focus_areas <= 3`
- `suggested_evidence_priority <= 5`

## Analyst Schema（开放诊断，稳定接口）
后端 `analysis_result` 核心结构：
```json
{
  "analysis_type": "diagnosis | root_cause | action_planning | risk_review | opportunity_scan | other",
  "overall_assessment": {
    "status": "healthy | warning | risk | mixed",
    "summary": "..."
  },
  "key_diagnoses": [
    {
      "title": "...",
      "description": "...",
      "importance": "high | medium | low",
      "evidence_refs": ["metric_name"]
    }
  ],
  "recommended_actions": [
    {
      "action": "...",
      "why_it_matters": "...",
      "priority": "high | medium | low",
      "related_diagnoses": ["title"]
    }
  ],
  "talking_points": ["..."],
  "confidence_note": "..."
}
```
约束：
- `recommended_actions <= 3`
- `evidence_refs` 必须引用 evidence 里的指标名
- 不限制 action 来自固定 map（允许 LLM 自由生成）

## Evidence 设计
`build_analysis_evidence(...)` 以原始证据为主：
- `merchant_profile`
- `performance_summary`
- `metrics`
- `peer_comparison`（数值对比）
- `rule_hints.note`（仅参考）

## 前端模式
- `Mock Demo`：前端本地模板渲染（用于稳定演示）。
- `Real LLM`：调用 `/api/analyze`，直接按后端 analyst schema 渲染：
  - `analysis_result.overall_assessment.summary`
  - `analysis_result.key_diagnoses`
  - `analysis_result.recommended_actions`
  - `analysis_result.talking_points`

## 项目文件
- `merchant_growth_mvp.py`：主逻辑（planner、evidence、analyst、fallback、pipeline）
- `copilot_server.py`：本地服务（静态页面 + `POST /api/analyze`）
- `copilot_frontend.html`：单页前端（brand id + mock/real 切换）
- `MVP_test_cases.ipynb`：测试 notebook
- `MVP.py`：命令行入口
- `.env.example`：环境变量示例

## 环境变量
推荐把 key 放仓库外目录：
```env
OPENAI_API_KEY=your_key
# 可选：网络受限场景
# OPENAI_BASE_URL=https://your-proxy-or-gateway/v1
```
建议路径：`~/.merchant-agent/.env`

可选显式指定：
```bash
export MERCHANT_AGENT_ENV_PATH="$HOME/.merchant-agent/.env"
```

读取优先级：
1. `MERCHANT_AGENT_ENV_PATH`
2. `~/.merchant-agent/.env`
3. 项目内 `.env`

## 运行
启动服务：
```bash
cd "/Users/pro/Desktop/merchant agent"
python copilot_server.py
```
打开页面：
`http://127.0.0.1:8765/copilot_frontend.html`

或运行 notebook：
```bash
cd "/Users/pro/Desktop/merchant agent"
jupyter notebook
```
打开 `MVP_test_cases.ipynb`。

## API
`POST /api/analyze`

请求体：
```json
{
  "brand_id": "B001",
  "question": "为什么最近订单下滑？",
  "mock_mode": false
}
```

## 快速验证
- `Real LLM` 模式下，查看返回中的 `analysis_result.overall_assessment.summary` 等字段是否被页面直接渲染。
- 故意让 LLM 不可用（如断网/无key）时，确认仍返回同一 analyst schema，`mode=fallback_rule`。

## 常见问题
- `500`：先确认当前 Python 环境已安装 `openai` 且 `OPENAI_API_KEY` 可读。
- `APIConnectionError`：配置 `OPENAI_BASE_URL`。
- 页面调用失败：确认服务进程在运行，并从 `http://127.0.0.1:8765/...` 打开页面（非 `file://`）。
