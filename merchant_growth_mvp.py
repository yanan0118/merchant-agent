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
    "irrelevant": AnalysisPlan(
        intent="irrelevant",
        focus_metrics=[],
        analysis_modules=[],
        time_range="",
    ),
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

BUSINESS_RELATED_KEYWORDS = {
    "商家", "门店", "店铺", "经营", "业务", "订单", "流量", "曝光", "转化", "销量",
    "gmv", "营业", "客单价", "履约", "出餐", "取消率", "sku", "商品", "菜单",
    "折扣", "活动", "诊断", "原因", "下降", "提升", "建议", "merchant",
    "conversion", "orders", "traffic", "cancel", "discount", "store",
}

IRRELEVANT_KEYWORDS = {
    "天气", "新闻", "股票", "八卦", "电影", "电视剧", "动漫", "游戏", "旅游", "翻译",
    "写代码", "python 教程", "算法题", "笑话", "星座", "菜谱", "健身", "减肥", "情感",
    "nba", "足球", "彩票", "出行", "机票", "酒店", "music", "movie", "recipe",
}

IRRELEVANT_RESPONSE_TEXT = (
    "这是无关问题。请继续提问商家经营诊断、订单波动原因、增长建议、转化/履约/供给等相关问题。"
)


def create_mock_inputs(seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    merchant_df = pd.DataFrame(
        [
            {
                "merchant_id": "M001",
                "brand_id": "B001",
                "brand_name": "Blue Bottle",
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
                "merchant_id": "M003",
                "brand_id": "B001",
                "brand_name": "Blue Bottle",
                "merchant_name": "Blue Bottle Downtown",
                "category": "coffee",
                "district": "Jingan",
                "orders_7d": 156,
                "orders_prev_7d": 168,
                "orders_last_30d_pctl": 0.86,
                "orders_peer_pctl": 0.67,
                "impressions": 6300,
                "conversion_rate": 0.026,
                "image_coverage": 0.62,
                "open_hours": 9.5,
                "accept_time_mins": 2.8,
                "prep_time_mins": 15.2,
                "merchant_cancel_rate": 0.041,
                "active_spu_count": 30,
                "discount_rate": 0.08,
            },
            {
                "merchant_id": "M002",
                "brand_id": "B002",
                "brand_name": "Sunrise Burger",
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
            {
                "merchant_id": "M004",
                "brand_id": "B002",
                "brand_name": "Sunrise Burger",
                "merchant_name": "Sunrise Burger Center",
                "category": "burger",
                "district": "Pudong",
                "orders_7d": 248,
                "orders_prev_7d": 255,
                "orders_last_30d_pctl": 0.62,
                "orders_peer_pctl": 0.49,
                "impressions": 9800,
                "conversion_rate": 0.031,
                "image_coverage": 0.76,
                "open_hours": 11.3,
                "accept_time_mins": 2.3,
                "prep_time_mins": 13.4,
                "merchant_cancel_rate": 0.028,
                "active_spu_count": 38,
                "discount_rate": 0.1,
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


def get_brand_row(merchant_df: pd.DataFrame, brand_id: str) -> pd.Series:
    brand_df = merchant_df.loc[merchant_df["brand_id"] == brand_id]
    if brand_df.empty:
        raise ValueError(f"brand_id not found: {brand_id}")

    brand_name = str(brand_df.iloc[0]["brand_name"])
    aggregated = brand_df.iloc[0].copy()
    numeric_cols = brand_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        aggregated[col] = float(brand_df[col].mean())

    aggregated["merchant_id"] = f"BRAND:{brand_id}"
    aggregated["merchant_name"] = f"{brand_name} (Brand Aggregate)"
    aggregated["brand_id"] = brand_id
    aggregated["brand_name"] = brand_name
    return aggregated


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


def fallback_rule_planner(question: str) -> Dict[str, Any]:
    """Rule-based planner used as the stable fallback path."""
    lower_q = question.lower()

    if any(keyword in question for keyword in IRRELEVANT_KEYWORDS) and not any(
        keyword in question or keyword in lower_q for keyword in BUSINESS_RELATED_KEYWORDS
    ):
        plan = PLAN_LIBRARY["irrelevant"]
        return {
            "intent": plan.intent,
            "modules": plan.analysis_modules,
            "focus_metrics": plan.focus_metrics,
            "reason": "规则识别为无关问题",
            "plan": plan,
            "mode": "fallback_rule",
        }

    if ("为什么" in question) or ("原因" in question) or ("下降" in question):
        plan = PLAN_LIBRARY["root_cause"]
        return {
            "intent": plan.intent,
            "modules": plan.analysis_modules,
            "focus_metrics": plan.focus_metrics,
            "reason": "规则识别为原因分析问题",
            "plan": plan,
            "mode": "fallback_rule",
        }
    if ("建议" in question) or ("怎么做" in question) or ("提升" in question):
        plan = PLAN_LIBRARY["action_recommendation"]
        return {
            "intent": plan.intent,
            "modules": plan.analysis_modules,
            "focus_metrics": plan.focus_metrics,
            "reason": "规则识别为动作建议问题",
            "plan": plan,
            "mode": "fallback_rule",
        }
    if not any(keyword in question or keyword in lower_q for keyword in BUSINESS_RELATED_KEYWORDS):
        plan = PLAN_LIBRARY["irrelevant"]
        return {
            "intent": plan.intent,
            "modules": plan.analysis_modules,
            "focus_metrics": plan.focus_metrics,
            "reason": "规则识别为无关问题",
            "plan": plan,
            "mode": "fallback_rule",
        }

    plan = PLAN_LIBRARY["diagnosis"]
    return {
        "intent": plan.intent,
        "modules": plan.analysis_modules,
        "focus_metrics": plan.focus_metrics,
        "reason": "规则识别为整体经营诊断问题",
        "plan": plan,
        "mode": "fallback_rule",
    }


def plan_analysis(question: str) -> AnalysisPlan:
    """Backward-compatible wrapper for previous callers."""
    return fallback_rule_planner(question)["plan"]


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
        "brand_id": merchant_row.get("brand_id", ""),
        "brand_name": merchant_row.get("brand_name", ""),
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


def resolve_env_path(env_path: Optional[str] = None) -> Path:
    if env_path:
        return Path(env_path).expanduser()

    configured_path = os.getenv("MERCHANT_AGENT_ENV_PATH")
    if configured_path:
        return Path(configured_path).expanduser()

    default_external_path = Path.home() / ".merchant-agent" / ".env"
    if default_external_path.exists():
        return default_external_path

    return Path(".env")


def load_local_env(env_path: Optional[str] = None) -> bool:
    env_file = resolve_env_path(env_path)
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


def create_openai_client() -> Any:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 没设置，请先配置可读取到 key 的 .env")

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def validate_plan_schema(payload: Dict[str, Any]) -> Dict[str, Any]:
    allowed_intents = {"diagnosis", "root_cause", "action_recommendation", "irrelevant"}
    allowed_modules = {"health", "gap", "action"}

    intent = payload.get("intent")
    if intent not in allowed_intents:
        raise ValueError(f"invalid intent: {intent}")

    modules = payload.get("modules")
    if not isinstance(modules, list):
        raise ValueError("modules must be a list")
    if any((not isinstance(m, str)) or (m not in allowed_modules) for m in modules):
        raise ValueError(f"invalid modules: {modules}")

    if intent == "irrelevant" and modules:
        raise ValueError("irrelevant intent must have empty modules")
    if intent != "irrelevant" and not modules:
        raise ValueError("non-irrelevant intent must have modules")

    focus_metrics = payload.get("focus_metrics")
    if not isinstance(focus_metrics, list):
        raise ValueError("focus_metrics must be a list")
    if any(not isinstance(x, str) for x in focus_metrics):
        raise ValueError("focus_metrics must be list[str]")

    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("reason must be str")

    return {
        "intent": intent,
        "modules": modules,
        "focus_metrics": focus_metrics,
        "reason": reason.strip(),
    }


def build_planner_prompt(question: str) -> str:
    return f"""
你是商家增长助手的 analysis planner。
请先判断用户问题是否与商家经营分析相关，然后输出分析计划。

只允许以下 intent：
- diagnosis
- root_cause
- action_recommendation
- irrelevant

只允许以下 modules（按需选择）：
- health
- gap
- action

输出必须是 JSON，且仅输出 JSON，不要额外解释。格式如下：
{{
  "intent": "diagnosis|root_cause|action_recommendation|irrelevant",
  "modules": ["health", "gap", "action"],
  "focus_metrics": ["..."],
  "reason": "简短中文理由"
}}

约束：
- 如果 intent 是 irrelevant，modules 必须为空数组，focus_metrics 也应为空或很少。
- 不要编造数据库字段，focus_metrics 尽量使用常见经营指标词汇。

用户问题：{question}
""".strip()


def parse_planner_payload(raw_text: str) -> Dict[str, Any]:
    json_text = extract_json_object(raw_text)
    payload = json.loads(json_text)
    return validate_plan_schema(payload)


def plan_analysis_with_llm(question: str, model: str = "gpt-4.1-mini") -> Dict[str, Any]:
    client = create_openai_client()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "你是商家增长诊断系统的规划器，只返回合法 JSON。"},
            {"role": "user", "content": build_planner_prompt(question)},
        ],
    )
    parsed = parse_planner_payload(response.output_text)
    plan = AnalysisPlan(
        intent=parsed["intent"],
        focus_metrics=parsed["focus_metrics"],
        analysis_modules=parsed["modules"],
        time_range="7d",
    )
    return {
        "intent": parsed["intent"],
        "modules": parsed["modules"],
        "focus_metrics": parsed["focus_metrics"],
        "reason": parsed["reason"],
        "plan": plan,
        "mode": "real_llm",
        "raw_response": response.output_text,
    }


def get_analysis_plan(question: str, mock_mode: bool = True, model: str = "gpt-4.1-mini") -> Dict[str, Any]:
    if mock_mode:
        return fallback_rule_planner(question)

    try:
        return plan_analysis_with_llm(question=question, model=model)
    except Exception as exc:
        fallback = fallback_rule_planner(question)
        fallback["reason"] = f"{fallback['reason']}（LLM planner fallback: {exc}）"
        fallback["mode"] = "fallback_rule"
        return fallback


def fallback_rule_answer(structured_output: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
    framework = build_analysis_framework(structured_output)
    summary = ""
    if framework.get("health"):
        health = framework["health"]
        summary = (
            f"品牌当前健康度状态为 {health['status']}，近7日订单变化 {health['order_change_rate']:.1%}，"
            "建议优先处理影响转化与履约的关键短板。"
        )
    else:
        summary = "当前问题聚焦专项诊断，建议优先按关键 gap 推进动作。"

    problems = []
    if framework.get("gaps"):
        problems = [
            f"{item['metric']} 在 {item['driver']} 维度表现为 {item['label']}"
            for item in framework["gaps"][:3]
        ]
    else:
        problems = ["当前暂无可用 gap 结果。"]

    actions = []
    if framework.get("allowed_actions"):
        actions = [item["action"] for item in framework["allowed_actions"][:3]]
    else:
        actions = ["当前 intent 未要求输出动作建议。"]

    note = "本轮建议基于结构化分析结果，不包含框架外推断。"
    if reason:
        note = f"{note} 系统提示：{reason}"

    text = (
        f"Executive Summary: {summary}\n\n"
        f"Key Problems: {'；'.join(problems)}\n\n"
        f"Recommended Actions: {'；'.join(actions)}\n\n"
        f"Note: {note}"
    )
    return {
        "summary": summary,
        "problems": problems,
        "actions": actions,
        "mode": "fallback_rule",
        "text": text,
        "estimated_input_tokens": None,
        "actual_input_tokens": None,
        "actual_output_tokens": None,
        "actual_total_tokens": None,
    }


def build_answer_prompt(structured_output: Dict[str, Any]) -> str:
    framework = build_analysis_framework(structured_output)
    return f"""
你是商家经营分析助手，面向客户经理输出结论。
你必须严格基于给定结构化结果，不允许编造。

请输出 JSON，字段固定为：
{{
  "summary": "Executive Summary，中文，1-2句",
  "problems": ["Key Problems 列表，2-4条"],
  "actions": ["Recommended Actions 列表，2-4条"]
}}

强约束：
- 不要编造任何新数据或新指标。
- actions 必须优先复用 allowed_actions 的原始动作文案，不要凭空发明动作。
- 若无 allowed_actions，actions 请明确写“当前 intent 未要求输出动作建议”。
- 文风专业、克制、可执行，语言中文。
- 只输出 JSON，不要附加其他解释。

结构化结果：
{json.dumps(framework, ensure_ascii=False, indent=2)}
""".strip()


def parse_answer_payload(raw_text: str) -> Dict[str, Any]:
    payload = json.loads(extract_json_object(raw_text))
    for key in ["summary", "problems", "actions"]:
        if key not in payload:
            raise ValueError(f"missing field: {key}")

    if not isinstance(payload["summary"], str):
        raise ValueError("summary must be str")
    for key in ["problems", "actions"]:
        value = payload[key]
        if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
            raise ValueError(f"{key} must be list[str]")
    return payload


def generate_answer_with_llm(
    structured_output: Dict[str, Any],
    model: str = "gpt-4.1",
    mock_mode: bool = True,
) -> Dict[str, Any]:
    if mock_mode:
        return fallback_rule_answer(structured_output, reason="mock_mode=True")

    prompt = build_answer_prompt(structured_output)
    estimated_tokens = estimate_text_tokens(prompt, model=model)

    try:
        client = create_openai_client()
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "你是商家增长助手，只能返回合法 JSON。"},
                {"role": "user", "content": prompt},
            ],
        )
        parsed = parse_answer_payload(response.output_text)
        usage = getattr(response, "usage", None)
        text = (
            f"Executive Summary: {parsed['summary']}\n\n"
            f"Key Problems: {'；'.join(parsed['problems'])}\n\n"
            f"Recommended Actions: {'；'.join(parsed['actions'])}"
        )
        return {
            "summary": parsed["summary"],
            "problems": parsed["problems"],
            "actions": parsed["actions"],
            "mode": "real_llm",
            "text": text,
            "estimated_input_tokens": estimated_tokens,
            "actual_input_tokens": getattr(usage, "input_tokens", None) if usage else None,
            "actual_output_tokens": getattr(usage, "output_tokens", None) if usage else None,
            "actual_total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        }
    except Exception as exc:
        fallback = fallback_rule_answer(structured_output, reason=f"LLM answer fallback: {exc}")
        fallback["estimated_input_tokens"] = estimated_tokens
        return fallback


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

请只从以下四个 intent 中选择一个：
- irrelevant
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

    client = create_openai_client()
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

    lines.append("（这是 mock 输出，未调用真实 LLM）")
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

    client = create_openai_client()
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
    brand_id: str,
    question: str,
    merchant_df: pd.DataFrame,
    peer_benchmark: Dict[str, np.ndarray],
    health_config: Dict[str, float] = HEALTH_CONFIG,
    gap_config: Dict[str, float] = GAP_CONFIG,
    intent_model: str = "gpt-4.1-mini",
    analysis_model: str = "gpt-4.1",
    mock_mode: bool = True,
) -> Dict[str, Any]:
    load_local_env()
    brand_row = get_brand_row(merchant_df, brand_id)
    plan_result = get_analysis_plan(question=question, mock_mode=mock_mode, model=intent_model)

    if plan_result["intent"] == "irrelevant":
        return {
            "brand_id": brand_id,
            "brand_name": str(brand_row["brand_name"]),
            "question": question,
            "intent_result": {
                "intent": plan_result["intent"],
                "reason": plan_result["reason"],
                "mode": plan_result["mode"],
            },
            "plan_result": {
                "intent": plan_result["intent"],
                "modules": plan_result["modules"],
                "focus_metrics": plan_result["focus_metrics"],
                "reason": plan_result["reason"],
                "mode": plan_result["mode"],
            },
            "intercepted": True,
            "structured_output": None,
            "analysis_result": {
                "summary": IRRELEVANT_RESPONSE_TEXT,
                "problems": [],
                "actions": [],
                "text": IRRELEVANT_RESPONSE_TEXT,
                "estimated_input_tokens": None,
                "actual_input_tokens": None,
                "actual_output_tokens": None,
                "actual_total_tokens": None,
                "mode": "fallback_rule",
            },
        }

    structured_output = run_pipeline(
        merchant_row=brand_row,
        question=question,
        peer_benchmark=peer_benchmark,
        health_config=health_config,
        gap_config=gap_config,
        plan_override=plan_result["plan"],
    )
    analysis_result = generate_answer_with_llm(
        structured_output=structured_output,
        model=analysis_model,
        mock_mode=mock_mode,
    )
    return {
        "brand_id": brand_id,
        "brand_name": str(brand_row["brand_name"]),
        "question": question,
        "intent_result": {
            "intent": plan_result["intent"],
            "reason": plan_result["reason"],
            "mode": plan_result["mode"],
        },
        "plan_result": {
            "intent": plan_result["intent"],
            "modules": plan_result["modules"],
            "focus_metrics": plan_result["focus_metrics"],
            "reason": plan_result["reason"],
            "mode": plan_result["mode"],
        },
        "structured_output": structured_output,
        "analysis_result": analysis_result,
    }
