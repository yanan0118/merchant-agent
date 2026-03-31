import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import tiktoken
except ImportError:
    tiktoken = None


HEALTH_CONFIG = {
    "warning_threshold": 0.70,
    "risk_hist_threshold": 0.90,
    "risk_peer_threshold": 0.70,
}

GAP_CONFIG = {
    "gap_percentile_threshold": 0.30,
    "strong_gap_percentile_threshold": 0.15,
}

DISPLAY_ORDER = [
    "traffic",
    "ops_readiness",
    "user_experience",
    "supply_quality",
]

DRIVER_WEIGHTS = {
    "traffic": 0.35,
    "ops_readiness": 0.25,
    "user_experience": 0.20,
    "supply_quality": 0.20,
}

METRIC_TO_DRIVER = {
    "conversion_rate": "traffic",
    "image_coverage": "ops_readiness",
    "open_hours": "ops_readiness",
    "accept_time_mins": "ops_readiness",
    "prep_time_mins": "user_experience",
    "merchant_cancel_rate": "user_experience",
    "active_spu_count": "supply_quality",
    "discount_rate": "supply_quality",
}

ACTION_MAP = {
    "conversion_rate": "优化商品图片或设置更有吸引力的折扣",
    "image_coverage": "补齐菜单图片，优先补热销商品图",
    "open_hours": "延长晚间营业时间，覆盖高峰时段",
    "accept_time_mins": "优化接单流程，缩短接单时长",
    "prep_time_mins": "优化后厨流程，缩短出餐时间",
    "merchant_cancel_rate": "减少商责取消，优化备货与接单判断",
    "active_spu_count": "补充动销 SKU，优化菜单供给结构",
    "discount_rate": "提升核心商品折扣力度，增强价格竞争力",
}

METRIC_DIRECTION = {
    "conversion_rate": True,
    "image_coverage": True,
    "open_hours": True,
    "accept_time_mins": False,
    "prep_time_mins": False,
    "merchant_cancel_rate": False,
    "active_spu_count": True,
    "discount_rate": True,
}


@dataclass
class HealthResult:
    status: str
    hist_percentile: float
    peer_percentile: float
    order_change_rate: float
    summary: str


@dataclass
class GapResult:
    metric: str
    driver: str
    percentile: float
    value: float
    label: str


@dataclass
class ActionResult:
    driver: str
    metric: str
    action: str
    score: float
    priority: str


@dataclass
class AnalysisPlan:
    intent: str
    focus_metrics: List[str]
    analysis_modules: List[str]
    time_range: str


PLAN_LIBRARY = {
    "root_cause": AnalysisPlan(
        intent="root_cause",
        focus_metrics=[
            "orders_7d",
            "impressions",
            "conversion_rate",
            "prep_time_mins",
            "merchant_cancel_rate",
        ],
        analysis_modules=["health", "gap"],
        time_range="7d",
    ),
    "action_recommendation": AnalysisPlan(
        intent="action_recommendation",
        focus_metrics=[
            "conversion_rate",
            "image_coverage",
            "open_hours",
            "merchant_cancel_rate",
            "active_spu_count",
            "discount_rate",
        ],
        analysis_modules=["gap", "action"],
        time_range="7d",
    ),
    "diagnosis": AnalysisPlan(
        intent="diagnosis",
        focus_metrics=[
            "orders_7d",
            "orders_prev_7d",
            "conversion_rate",
            "image_coverage",
            "prep_time_mins",
        ],
        analysis_modules=["health", "gap", "action"],
        time_range="7d",
    ),
}


def create_mock_inputs(seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    merchant_df = pd.DataFrame(
        [
            {
                "merchant_id": "M001",
                "merchant_name": "Blue Bottle Demo Store",
                "category": "coffee",
                "district": "Pudong",
                "orders_7d": 120,
                "orders_prev_7d": 145,
                "orders_last_30d_pctl": 0.94,
                "orders_peer_pctl": 0.78,
                "impressions": 5200,
                "conversion_rate": 0.021,
                "image_coverage": 0.46,
                "open_hours": 8.5,
                "accept_time_mins": 3.2,
                "prep_time_mins": 18.0,
                "merchant_cancel_rate": 0.075,
                "active_spu_count": 22,
                "discount_rate": 0.06,
            },
            {
                "merchant_id": "M002",
                "merchant_name": "Sunrise Burger Demo Store",
                "category": "burger",
                "district": "Xuhui",
                "orders_7d": 280,
                "orders_prev_7d": 270,
                "orders_last_30d_pctl": 0.55,
                "orders_peer_pctl": 0.42,
                "impressions": 10200,
                "conversion_rate": 0.034,
                "image_coverage": 0.81,
                "open_hours": 12.0,
                "accept_time_mins": 2.1,
                "prep_time_mins": 12.5,
                "merchant_cancel_rate": 0.021,
                "active_spu_count": 41,
                "discount_rate": 0.11,
            },
        ]
    )

    rng = np.random.default_rng(seed)
    peer_benchmark = {
        "conversion_rate": np.clip(rng.normal(0.03, 0.005, 200), 0.005, 0.2),
        "image_coverage": np.clip(rng.normal(0.78, 0.12, 200), 0.05, 1.0),
        "open_hours": np.clip(rng.normal(11.0, 1.8, 200), 4.0, 20.0),
        "accept_time_mins": np.clip(rng.normal(2.5, 0.6, 200), 0.5, 10.0),
        "prep_time_mins": np.clip(rng.normal(14.0, 2.5, 200), 5.0, 40.0),
        "merchant_cancel_rate": np.clip(rng.normal(0.03, 0.012, 200), 0.0, 0.25),
        "active_spu_count": np.clip(rng.normal(35, 8, 200), 1, 200),
        "discount_rate": np.clip(rng.normal(0.10, 0.03, 200), 0.0, 0.6),
    }
    return merchant_df, peer_benchmark


def get_merchant_row(merchant_df: pd.DataFrame, merchant_id: str) -> pd.Series:
    return merchant_df.loc[merchant_df["merchant_id"] == merchant_id].iloc[0]


def percentile_rank(value: float, population: np.ndarray, higher_is_better: bool = True) -> float:
    population = np.asarray(population)
    raw = float((population < value).mean())
    return raw if higher_is_better else 1 - raw


def gap_severity(percentile: float) -> float:
    return max(0.0, 1.0 - percentile)


def priority_label(score: float) -> str:
    if score >= 0.60:
        return "High"
    if score >= 0.35:
        return "Medium"
    return "Low"


def calculate_growth_health(row: pd.Series, config: Dict[str, float]) -> HealthResult:
    hist_p = float(row["orders_last_30d_pctl"])
    peer_p = float(row["orders_peer_pctl"])

    if hist_p >= config["risk_hist_threshold"] and peer_p >= config["risk_peer_threshold"]:
        status = "Risk"
    elif hist_p >= config["warning_threshold"] or peer_p >= config["warning_threshold"]:
        status = "Warning"
    else:
        status = "Healthy"

    growth = (row["orders_7d"] - row["orders_prev_7d"]) / row["orders_prev_7d"]
    summary = (
        f"近7日订单变化 {growth:.1%}；"
        f"历史异常分位 {hist_p:.0%}；"
        f"同行异常分位 {peer_p:.0%}。"
    )

    return HealthResult(
        status=status,
        hist_percentile=hist_p,
        peer_percentile=peer_p,
        order_change_rate=float(growth),
        summary=summary,
    )


def identify_gaps(
    row: pd.Series,
    peer_data: Dict[str, np.ndarray],
    gap_config: Dict[str, float],
) -> List[GapResult]:
    gap_results: List[GapResult] = []

    for metric, driver in METRIC_TO_DRIVER.items():
        value = float(row[metric])
        higher_is_better = METRIC_DIRECTION[metric]
        percentile = percentile_rank(value, peer_data[metric], higher_is_better=higher_is_better)

        if percentile < gap_config["strong_gap_percentile_threshold"]:
            label = "Strong Gap"
        elif percentile < gap_config["gap_percentile_threshold"]:
            label = "Gap"
        elif percentile > 0.70:
            label = "Strong"
        else:
            label = "Normal"

        gap_results.append(
            GapResult(
                metric=metric,
                driver=driver,
                percentile=percentile,
                value=value,
                label=label,
            )
        )

    return gap_results


def generate_actions(gaps: List[GapResult]) -> List[ActionResult]:
    actions: List[ActionResult] = []

    for gap in gaps:
        if gap.label not in {"Gap", "Strong Gap"}:
            continue

        severity = gap_severity(gap.percentile)
        score = severity * DRIVER_WEIGHTS[gap.driver]
        actions.append(
            ActionResult(
                driver=gap.driver,
                metric=gap.metric,
                action=ACTION_MAP[gap.metric],
                score=score,
                priority=priority_label(score),
            )
        )

    return sorted(actions, key=lambda item: item.score, reverse=True)


def build_structured_output(
    merchant: pd.Series,
    health: HealthResult,
    gaps: List[GapResult],
    actions: List[ActionResult],
    question: str,
) -> Dict[str, Any]:
    grouped_gaps: Dict[str, List[Dict[str, Any]]] = {driver: [] for driver in DISPLAY_ORDER}
    for gap in gaps:
        grouped_gaps[gap.driver].append(
            {
                "metric": gap.metric,
                "value": gap.value,
                "percentile": round(gap.percentile, 3),
                "label": gap.label,
            }
        )

    return {
        "merchant_id": merchant["merchant_id"],
        "merchant_name": merchant["merchant_name"],
        "question": question,
        "health": asdict(health),
        "gaps_by_driver": grouped_gaps,
        "recommended_actions": [asdict(action) for action in actions],
    }


def render_report(output: Dict[str, Any]) -> None:
    print("=== Merchant Growth Copilot ===")
    print(f"Merchant: {output['merchant_name']} ({output['merchant_id']})")
    print(f"Question: {output['question']}")
    print()

    print("[1] Health Status")
    print(f"- Status: {output['health']['status']}")
    print(f"- Summary: {output['health']['summary']}")
    print()

    print("[2] Key Gaps by Driver")
    for driver in DISPLAY_ORDER:
        print(f"- {driver}")
        items = output["gaps_by_driver"][driver]
        if not items:
            print("  - No metrics")
            continue
        for item in items:
            print(
                f"  - {item['metric']}: {item['label']} "
                f"(value={item['value']}, percentile={item['percentile']:.1%})"
            )
    print()

    print("[3] Recommended Actions")
    if not output["recommended_actions"]:
        print("- No recommended actions")
    else:
        for action in output["recommended_actions"]:
            print(
                f"- [{action['priority']}] {action['action']} "
                f"(driver={action['driver']}, metric={action['metric']}, score={action['score']:.2f})"
            )


def plan_analysis(question: str) -> AnalysisPlan:
    if ("为什么" in question) or ("原因" in question) or ("下降" in question):
        return PLAN_LIBRARY["root_cause"]
    if ("建议" in question) or ("怎么做" in question) or ("提升" in question):
        return PLAN_LIBRARY["action_recommendation"]
    return PLAN_LIBRARY["diagnosis"]


def run_pipeline(
    merchant_row: pd.Series,
    question: str,
    peer_benchmark: Dict[str, np.ndarray],
    health_config: Dict[str, float],
    gap_config: Dict[str, float],
    plan_override: Optional[AnalysisPlan] = None,
) -> Dict[str, Any]:
    plan = plan_override or plan_analysis(question)
    result: Dict[str, Any] = {
        "question": question,
        "plan": asdict(plan),
        "merchant_id": merchant_row["merchant_id"],
        "merchant_name": merchant_row["merchant_name"],
    }

    gap_results: Optional[List[GapResult]] = None

    if "health" in plan.analysis_modules:
        health_result = calculate_growth_health(merchant_row, health_config)
        result["health"] = asdict(health_result)

    if "gap" in plan.analysis_modules:
        gap_results = identify_gaps(merchant_row, peer_benchmark, gap_config)
        result["gaps"] = [asdict(item) for item in gap_results]

    if "action" in plan.analysis_modules:
        if gap_results is None:
            gap_results = identify_gaps(merchant_row, peer_benchmark, gap_config)
        action_results = generate_actions(gap_results)
        result["actions"] = [asdict(item) for item in action_results]

    return result


def load_local_env(env_path: str = ".env") -> bool:
    env_file = Path(env_path)
    if not env_file.exists():
        return False

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return True


def estimate_text_tokens(text: str, model: str = "gpt-4.1") -> int:
    if tiktoken is None:
        return max(1, len(text) // 4)
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def normalize_plan_from_intent(intent: str) -> AnalysisPlan:
    if intent not in PLAN_LIBRARY:
        raise ValueError(f"Unsupported intent: {intent}")
    return PLAN_LIBRARY[intent]


def extract_json_object(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        return match.group(0)
    raise ValueError(f"No JSON object found in model output: {text}")


def parse_intent_payload(raw_text: str) -> Dict[str, Any]:
    json_text = extract_json_object(raw_text)
    payload = json.loads(json_text)
    intent = payload.get("intent")
    if intent not in PLAN_LIBRARY:
        raise ValueError(f"Invalid intent from model: {intent}")
    return {
        "intent": intent,
        "reason": payload.get("reason", ""),
        "raw_response": raw_text,
        "parsed_payload": payload,
    }


def build_intent_router_prompt(question: str) -> str:
    return f"""
你是商家经营分析系统的意图识别模块。

请只从以下三个 intent 中选择一个：
- diagnosis
- root_cause
- action_recommendation

输出必须是 JSON，格式如下：
{{"intent": "...", "reason": "..."}}

用户问题：{question}
""".strip()


def identify_intent(question: str, model: str = "gpt-4.1-mini", mock_mode: bool = True) -> Dict[str, Any]:
    if mock_mode:
        rule_plan = plan_analysis(question)
        return {
            "intent": rule_plan.intent,
            "reason": "mock mode: 先用现有 rule-based plan 代替 LLM 意图识别",
            "plan": rule_plan,
            "mode": "mock",
        }

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 没设置，请先在工作区根目录创建 .env")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "你只负责识别 intent，不做经营分析。"},
            {"role": "user", "content": build_intent_router_prompt(question)},
        ],
    )

    raw_text = response.output_text
    try:
        parsed = parse_intent_payload(raw_text)
        intent = parsed["intent"]
    except Exception as exc:
        fallback_plan = plan_analysis(question)
        return {
            "intent": fallback_plan.intent,
            "reason": f"fallback to rule-based planner because model output was invalid: {exc}",
            "plan": fallback_plan,
            "mode": "fallback",
            "raw_response": raw_text,
        }

    return {
        "intent": intent,
        "reason": parsed.get("reason", ""),
        "plan": normalize_plan_from_intent(intent),
        "mode": "real",
        "raw_response": raw_text,
        "parsed_payload": parsed["parsed_payload"],
    }


def build_analysis_framework(output: Dict[str, Any]) -> Dict[str, Any]:
    allowed_actions = [
        {
            "metric": item["metric"],
            "driver": item["driver"],
            "action": item["action"],
            "priority": item["priority"],
            "score": round(item["score"], 4),
        }
        for item in output.get("actions", [])
    ]
    return {
        "question": output["question"],
        "intent": output["plan"]["intent"],
        "focus_metrics": output["plan"]["focus_metrics"],
        "analysis_modules": output["plan"]["analysis_modules"],
        "health": output.get("health"),
        "gaps": output.get("gaps", []),
        "allowed_actions": allowed_actions,
    }


def build_controlled_analysis_prompt(output: Dict[str, Any]) -> str:
    framework = build_analysis_framework(output)
    return f"""
你是商家经营分析助手，但你必须严格遵守给定分析框架，不要自由发挥。

请基于以下 framework 输出中文分析，结构固定为：
1. Intent
2. Executive Summary
3. Key Problems
4. Recommended Actions
5. Suggested Talking Points

硬性要求：
- 只能引用 framework 里出现的数据与字段
- 如果没有 health 模块，就不要写 health 结论
- 如果没有 gaps，就明确说明缺少 gap 结果
- Recommended Actions 只能从 allowed_actions 中选择，最多 3 条
- action 文案必须复用 allowed_actions 里的 action 原文，不要改写，不要新增
- 不要输出 SQL，不要假设额外数据
- 用简洁中文

analysis framework:
{json.dumps(framework, ensure_ascii=False, indent=2)}
""".strip()


def mock_controlled_analysis(output: Dict[str, Any]) -> str:
    framework = build_analysis_framework(output)
    lines = [f"1. Intent\n- {framework['intent']}"]

    if framework["health"]:
        health = framework["health"]
        lines.append(
            "2. Executive Summary\n"
            f"- 当前健康度状态为 {health['status']}，近7日订单变化 {health['order_change_rate']:.1%}。"
        )
    else:
        lines.append("2. Executive Summary\n- 当前问题聚焦于专项诊断，不输出健康度总览。")

    if framework["gaps"]:
        gap_lines = [
            f"- {item['metric']} 属于 {item['driver']}，当前标签为 {item['label']}"
            for item in framework["gaps"][:3]
        ]
        lines.append("3. Key Problems\n" + "\n".join(gap_lines))
    else:
        lines.append("3. Key Problems\n- 当前没有 gap 结果可供分析。")

    if framework["allowed_actions"]:
        action_lines = [
            f"- {item['action']}（metric={item['metric']}, priority={item['priority']}）"
            for item in framework["allowed_actions"][:3]
        ]
        lines.append("4. Recommended Actions\n" + "\n".join(action_lines))
    else:
        lines.append("4. Recommended Actions\n- 当前 intent 未要求输出 action 建议。")

    lines.append(
        "5. Suggested Talking Points\n"
        "- 先同步诊断结论，再按优先级推进已有 action。\n"
        "- 本轮结论严格基于现有 framework，没有扩展新的建议。\n\n"
        "（这是 mock 输出，未调用真实 LLM）"
    )
    return "\n\n".join(lines)


def generate_controlled_analysis(
    output: Dict[str, Any],
    model: str = "gpt-4.1",
    mock_mode: bool = True,
) -> Dict[str, Any]:
    prompt = build_controlled_analysis_prompt(output)
    estimated_input_tokens = estimate_text_tokens(prompt, model=model)

    if mock_mode:
        return {
            "text": mock_controlled_analysis(output),
            "estimated_input_tokens": estimated_input_tokens,
            "actual_input_tokens": None,
            "actual_output_tokens": None,
            "actual_total_tokens": None,
            "mode": "mock",
        }

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 没设置，请先在工作区根目录创建 .env")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "你是一个严格遵循分析框架的商家经营分析助手。"},
            {"role": "user", "content": prompt},
        ],
    )

    usage = getattr(response, "usage", None)
    return {
        "text": response.output_text,
        "estimated_input_tokens": estimated_input_tokens,
        "actual_input_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "actual_output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        "actual_total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "mode": "real",
    }


def run_llm_orchestrated_pipeline(
    merchant_row: pd.Series,
    question: str,
    peer_benchmark: Dict[str, np.ndarray],
    health_config: Dict[str, float] = HEALTH_CONFIG,
    gap_config: Dict[str, float] = GAP_CONFIG,
    intent_model: str = "gpt-4.1-mini",
    analysis_model: str = "gpt-4.1",
    mock_mode: bool = True,
) -> Dict[str, Any]:
    load_local_env()
    intent_result = identify_intent(question=question, model=intent_model, mock_mode=mock_mode)
    structured_output = run_pipeline(
        merchant_row=merchant_row,
        question=question,
        peer_benchmark=peer_benchmark,
        health_config=health_config,
        gap_config=gap_config,
        plan_override=intent_result["plan"],
    )
    analysis_result = generate_controlled_analysis(
        output=structured_output,
        model=analysis_model,
        mock_mode=mock_mode,
    )
    return {
        "question": question,
        "intent_result": {
            "intent": intent_result["intent"],
            "reason": intent_result["reason"],
            "mode": intent_result["mode"],
        },
        "structured_output": structured_output,
        "analysis_result": analysis_result,
    }
