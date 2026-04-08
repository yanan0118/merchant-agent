import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

ALLOWED_ANALYSIS_TYPES = {
    "diagnosis",
    "root_cause",
    "action_planning",
    "risk_review",
    "opportunity_scan",
    "other",
}
ALLOWED_OVERALL_STATUS = {"healthy", "warning", "risk", "mixed"}
ALLOWED_IMPORTANCE = {"high", "medium", "low"}
ALLOWED_FOCUS_AREAS = {
    "traffic",
    "conversion",
    "ops_readiness",
    "user_experience",
    "supply_quality",
    "other",
}
ALLOWED_EVIDENCE_METRICS = {
    "orders",
    "impressions",
    "conversion_rate",
    "image_coverage",
    "open_hours",
    "accept_time_mins",
    "prep_time_mins",
    "merchant_cancel_rate",
    "active_spu_count",
    "discount_rate",
}


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
    """Rule fallback planner with open content but stable interface."""
    lower_q = question.lower()
    is_irrelevant = any(keyword in question for keyword in IRRELEVANT_KEYWORDS) and not any(
        keyword in question or keyword in lower_q for keyword in BUSINESS_RELATED_KEYWORDS
    )

    if is_irrelevant or not any(keyword in question or keyword in lower_q for keyword in BUSINESS_RELATED_KEYWORDS):
        analysis_type = "other"
        analysis_goal = "当前问题与商家经营分析相关性较低，建议先明确经营目标或具体指标问题。"
        focus_areas = ["other"]
        suggested_evidence_priority = []
        reason = "规则识别为非经营相关问题"
    elif ("为什么" in question) or ("原因" in question) or ("下降" in question):
        analysis_type = "root_cause"
        analysis_goal = "定位订单/增长波动的关键驱动因素并解释因果链路。"
        focus_areas = ["traffic", "conversion", "ops_readiness"]
        suggested_evidence_priority = ["orders", "impressions", "conversion_rate", "merchant_cancel_rate", "prep_time_mins"]
        reason = "规则识别为原因分析问题"
    elif ("建议" in question) or ("怎么做" in question) or ("提升" in question):
        analysis_type = "action_planning"
        analysis_goal = "识别增长短板并给出可执行、可验证的优先级动作建议。"
        focus_areas = ["conversion", "ops_readiness", "supply_quality"]
        suggested_evidence_priority = ["conversion_rate", "image_coverage", "merchant_cancel_rate", "active_spu_count", "discount_rate"]
        reason = "规则识别为动作规划问题"
    elif ("风险" in question) or ("隐患" in question):
        analysis_type = "risk_review"
        analysis_goal = "识别影响短期稳定经营的高风险因素并给出缓释建议。"
        focus_areas = ["ops_readiness", "user_experience"]
        suggested_evidence_priority = ["merchant_cancel_rate", "prep_time_mins", "accept_time_mins", "open_hours"]
        reason = "规则识别为风险评估问题"
    elif ("机会" in question) or ("增长点" in question):
        analysis_type = "opportunity_scan"
        analysis_goal = "识别可放大的增长机会并给出试点动作。"
        focus_areas = ["traffic", "conversion", "supply_quality"]
        suggested_evidence_priority = ["impressions", "conversion_rate", "active_spu_count", "discount_rate"]
        reason = "规则识别为机会扫描问题"
    else:
        analysis_type = "diagnosis"
        analysis_goal = "做整体经营诊断并确定最影响增长的关键问题。"
        focus_areas = ["traffic", "conversion", "ops_readiness"]
        suggested_evidence_priority = ["orders", "impressions", "conversion_rate", "merchant_cancel_rate", "prep_time_mins"]
        reason = "规则识别为整体经营诊断问题"

    return {
        "analysis_type": analysis_type,
        "analysis_goal": analysis_goal,
        "focus_areas": focus_areas[:3],
        "suggested_evidence_priority": suggested_evidence_priority[:5],
        "reason": reason,
        "mode": "fallback_rule",
    }


def plan_analysis(question: str) -> AnalysisPlan:
    """Backward-compatible wrapper for previous callers."""
    plan_result = fallback_rule_planner(question)
    analysis_type = plan_result["analysis_type"]
    if analysis_type == "other":
        return PLAN_LIBRARY["irrelevant"]
    if analysis_type == "root_cause":
        return PLAN_LIBRARY["root_cause"]
    if analysis_type == "action_planning":
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
        "brand_id": merchant_row.get("brand_id", ""),
        "brand_name": merchant_row.get("brand_name", ""),
        "category": merchant_row.get("category", ""),
        "district": merchant_row.get("district", ""),
        "metric_snapshot": {
            "orders_7d": float(merchant_row.get("orders_7d", 0.0)),
            "orders_prev_7d": float(merchant_row.get("orders_prev_7d", 0.0)),
            "impressions": float(merchant_row.get("impressions", 0.0)),
            "conversion_rate": float(merchant_row.get("conversion_rate", 0.0)),
            "image_coverage": float(merchant_row.get("image_coverage", 0.0)),
            "open_hours": float(merchant_row.get("open_hours", 0.0)),
            "accept_time_mins": float(merchant_row.get("accept_time_mins", 0.0)),
            "prep_time_mins": float(merchant_row.get("prep_time_mins", 0.0)),
            "merchant_cancel_rate": float(merchant_row.get("merchant_cancel_rate", 0.0)),
            "active_spu_count": float(merchant_row.get("active_spu_count", 0.0)),
            "discount_rate": float(merchant_row.get("discount_rate", 0.0)),
        },
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
    analysis_type = payload.get("analysis_type")
    if analysis_type not in ALLOWED_ANALYSIS_TYPES:
        raise ValueError(f"invalid analysis_type: {analysis_type}")

    analysis_goal = payload.get("analysis_goal")
    if not isinstance(analysis_goal, str) or not analysis_goal.strip():
        raise ValueError("analysis_goal must be non-empty str")

    focus_areas = payload.get("focus_areas")
    if not isinstance(focus_areas, list):
        raise ValueError("focus_areas must be a list")
    if len(focus_areas) > 3:
        raise ValueError("focus_areas must contain at most 3 items")
    for area in focus_areas:
        if area not in ALLOWED_FOCUS_AREAS:
            raise ValueError(f"invalid focus area: {area}")

    suggested_evidence_priority = payload.get("suggested_evidence_priority")
    if not isinstance(suggested_evidence_priority, list):
        raise ValueError("suggested_evidence_priority must be a list")
    if len(suggested_evidence_priority) > 5:
        raise ValueError("suggested_evidence_priority must contain at most 5 items")
    for metric in suggested_evidence_priority:
        if metric not in ALLOWED_EVIDENCE_METRICS:
            raise ValueError(f"invalid evidence metric: {metric}")

    reason = payload.get("reason")
    if not isinstance(reason, str):
        raise ValueError("reason must be str")

    return {
        "analysis_type": analysis_type,
        "analysis_goal": analysis_goal.strip(),
        "focus_areas": focus_areas,
        "suggested_evidence_priority": suggested_evidence_priority,
        "reason": reason.strip(),
    }


def build_planner_prompt(question: str) -> str:
    return f"""
你是 Merchant Growth Copilot 的 analysis planner。
你可以自主判断分析方向，但必须输出固定 schema 的 JSON。

要求：
- analysis_type 只能从白名单中选择。
- focus_areas 最多 3 个。
- suggested_evidence_priority 最多 5 个，且必须来自允许指标集合。
- 输出只能是 JSON，不要额外文本。

输出格式：
{{
  "analysis_type": "diagnosis|root_cause|action_planning|risk_review|opportunity_scan|other",
  "analysis_goal": "...",
  "focus_areas": ["traffic|conversion|ops_readiness|user_experience|supply_quality|other"],
  "suggested_evidence_priority": [
    "orders|impressions|conversion_rate|image_coverage|open_hours|accept_time_mins|prep_time_mins|merchant_cancel_rate|active_spu_count|discount_rate"
  ],
  "reason": "简短中文理由"
}}

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
    return {
        "analysis_type": parsed["analysis_type"],
        "analysis_goal": parsed["analysis_goal"],
        "focus_areas": parsed["focus_areas"],
        "suggested_evidence_priority": parsed["suggested_evidence_priority"],
        "reason": parsed["reason"],
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


def _planner_to_execution_plan(plan_result: Dict[str, Any]) -> AnalysisPlan:
    """Convert open planner output into a minimal evidence-collection execution plan."""
    analysis_type = plan_result.get("analysis_type", "diagnosis")
    focus_metrics = plan_result.get("suggested_evidence_priority", [])

    if analysis_type == "other":
        return PLAN_LIBRARY["irrelevant"]
    if analysis_type == "root_cause":
        return AnalysisPlan(
            intent="root_cause",
            focus_metrics=focus_metrics,
            analysis_modules=["health", "gap"],
            time_range="7d",
        )
    if analysis_type in {"action_planning", "risk_review", "opportunity_scan"}:
        return AnalysisPlan(
            intent="action_recommendation",
            focus_metrics=focus_metrics,
            analysis_modules=["health", "gap"],
            time_range="7d",
        )
    return AnalysisPlan(
        intent="diagnosis",
        focus_metrics=focus_metrics,
        analysis_modules=["health", "gap"],
        time_range="7d",
    )


def _normalize_level(value: str, allowed: Set[str], default: str = "medium") -> str:
    norm = str(value).strip().lower()
    return norm if norm in allowed else default


def _question_to_analysis_type(question: str) -> str:
    q = question.lower()
    if any(k in question for k in ["原因", "为什么", "下滑"]):
        return "root_cause"
    if any(k in question for k in ["建议", "动作", "提升", "优化"]):
        return "action_planning"
    if any(k in question for k in ["风险", "隐患"]):
        return "risk_review"
    if any(k in question for k in ["机会", "增长点"]):
        return "opportunity_scan"
    if any(k in q for k in ["root cause", "action", "risk", "opportunity"]):
        if "root cause" in q:
            return "root_cause"
        if "risk" in q:
            return "risk_review"
        if "opportunity" in q:
            return "opportunity_scan"
        return "action_planning"
    return "diagnosis"


def build_analysis_evidence(question: str, structured_output: Dict[str, Any]) -> Dict[str, Any]:
    """Build evidence package with raw data as primary source for AI analyst."""
    gaps = structured_output.get("gaps", [])

    peer_comparison = {
        item["metric"]: {
            "value": float(item.get("value", 0.0)),
            "peer_percentile": float(item.get("percentile", 0.0)),
        }
        for item in gaps
    }

    health = structured_output.get("health", {}) or {}
    metrics = structured_output.get("metric_snapshot", {}) or {}
    performance_summary = {
        "orders_7d": metrics.get("orders_7d"),
        "orders_prev_7d": metrics.get("orders_prev_7d"),
        "order_change_rate": health.get("order_change_rate"),
        "health_status": health.get("status"),
        "health_summary": health.get("summary"),
    }

    return {
        "question": question,
        "merchant_profile": {
            "brand_id": structured_output.get("brand_id", ""),
            "brand_name": structured_output.get("brand_name", ""),
            "merchant_name": structured_output.get("merchant_name", ""),
            "category": structured_output.get("category", ""),
            "district": structured_output.get("district", ""),
        },
        "performance_summary": performance_summary,
        "metrics": metrics,
        "peer_comparison": peer_comparison,
        "rule_hints": {
            "note": "以下是系统常用分析视角，仅供参考",
        },
    }


def _decorate_analysis_result(result: Dict[str, Any]) -> Dict[str, Any]:
    overall = result.get("overall_assessment", {}) or {}
    summary = str(overall.get("summary", "")).strip() or "暂无总结。"

    diagnoses = []
    for d in result.get("key_diagnoses", []):
        importance = _normalize_level(d.get("importance", "medium"), ALLOWED_IMPORTANCE)
        title = str(d.get("title", "未命名问题")).strip()
        desc = str(d.get("description", "")).strip()
        if desc:
            diagnoses.append(f"[{importance}] {title}：{desc}")
        else:
            diagnoses.append(f"[{importance}] {title}")
    if not diagnoses:
        diagnoses = ["暂无明确诊断结论。"]

    actions = []
    for a in result.get("recommended_actions", []):
        priority = _normalize_level(a.get("priority", "medium"), ALLOWED_IMPORTANCE)
        action = str(a.get("action", "")).strip()
        why = str(a.get("why_it_matters", "")).strip()
        if action and why:
            actions.append(f"[{priority}] {action}：{why}")
        elif action:
            actions.append(f"[{priority}] {action}")
    if not actions:
        actions = ["暂无动作建议。"]

    talking_points = [
        x for x in result.get("talking_points", [])
        if isinstance(x, str) and x.strip()
    ]
    if not talking_points:
        talking_points = ["建议先与商家确认当前经营目标，再推进优先级最高的动作。"]

    text = (
        f"Executive Summary: {summary}\n\n"
        f"Key Diagnoses: {'；'.join(diagnoses)}\n\n"
        f"Recommended Actions: {'；'.join(actions)}\n\n"
        f"Talking Points: {'；'.join(talking_points)}"
    )
    return {
        **result,
        "summary": summary,
        "diagnoses": diagnoses,
        "problems": diagnoses,
        "actions": actions,
        "talking_points": talking_points,
        "text": text,
    }


def fallback_rule_answer(
    question: str,
    evidence: Dict[str, Any],
    reason: str = "",
) -> Dict[str, Any]:
    """Rule fallback that aligns to the AI schema."""
    health_status = str(
        (evidence.get("performance_summary", {}) or {}).get("health_status", "warning")
    ).strip().lower()
    status = health_status if health_status in ALLOWED_OVERALL_STATUS else "warning"

    peer_cmp = evidence.get("peer_comparison", {}) or {}
    ranked_metrics = sorted(
        peer_cmp.items(),
        key=lambda kv: float((kv[1] or {}).get("peer_percentile", 1.0)),
    )

    key_diagnoses = []
    for metric, info in ranked_metrics[:3]:
        key_diagnoses.append(
            {
                "title": f"{metric} 表现偏弱",
                "description": (
                    f"{metric} 的同类分位为 {float(info.get('peer_percentile', 0.0)):.1%}，"
                    f"当前值 {float(info.get('value', 0.0)):.4g}"
                ),
                "importance": "high" if float(info.get("peer_percentile", 1.0)) < 0.2 else "medium",
                "evidence_refs": [metric],
            }
        )
    if not key_diagnoses:
        key_diagnoses = [
            {
                "title": "经营状态需持续跟踪",
                "description": "当前可用证据有限，建议补充近 14 天关键指标后再做深度诊断。",
                "importance": "medium",
                "evidence_refs": [],
            }
        ]

    recommended_actions = []
    for metric, info in ranked_metrics[:3]:
        percentile = float(info.get("peer_percentile", 0.5))
        if metric == "image_coverage":
            action_text = "优先补齐热销 SKU 图片，并建立新品上架即补图的检查清单。"
        elif metric == "conversion_rate":
            action_text = "重写前 5 个主力商品标题与卖点，并同步优化价格锚点和套餐展示。"
        elif metric == "merchant_cancel_rate":
            action_text = "按高频取消原因做备货白名单与接单阈值，先把商责取消压到目标线。"
        elif metric == "prep_time_mins":
            action_text = "拆分后厨流程并设出餐时钟，优先优化高销量 SKU 的出餐路径。"
        else:
            action_text = f"针对 {metric} 制定 7 天专项优化动作并每日复盘。"

        recommended_actions.append(
            {
                "action": action_text,
                "why_it_matters": (
                    f"{metric} 当前同类分位仅 {percentile:.1%}，是当前增长的主要拖累项之一。"
                ),
                "priority": "high" if percentile < 0.2 else "medium",
                "related_diagnoses": [key_diagnoses[0]["title"]],
            }
        )

    if not recommended_actions:
        recommended_actions.append(
            {
                "action": "先补齐近 14 天关键经营数据，再确定首个高影响优化动作。",
                "why_it_matters": "证据不足时先保证数据质量，避免误判导致资源浪费。",
                "priority": "medium",
                "related_diagnoses": [key_diagnoses[0]["title"]],
            }
        )

    talking_points = [
        "先对齐本周增长目标，再按优先级推进 1 到 2 个高影响动作。",
        "每个动作建议设置 7 天观察窗口，复盘指标变化后再扩量。",
    ]
    if reason:
        talking_points.append(f"系统说明：{reason}")

    fallback = {
        "analysis_type": _question_to_analysis_type(question),
        "overall_assessment": {
            "status": status,
            "summary": (
                f"当前整体状态为 {status}，建议优先处理最弱指标并做小步快跑验证。"
            ),
        },
        "key_diagnoses": key_diagnoses,
        "recommended_actions": recommended_actions,
        "talking_points": talking_points,
        "confidence_note": "该结果来自规则回退，强证据来自指标分位，其他为经验性推断。",
    }
    return {
        **_decorate_analysis_result(fallback),
        "mode": "fallback_rule",
        "estimated_input_tokens": None,
        "actual_input_tokens": None,
        "actual_output_tokens": None,
        "actual_total_tokens": None,
    }


def build_ai_analyst_prompt(question: str, evidence: Dict[str, Any]) -> str:
    return f"""
【SYSTEM】
你是一名资深外卖平台商家增长分析专家（Senior Merchant Growth Analyst）。
你的能力：
- 从复杂数据中识别关键问题
- 判断影响订单增长的核心因素
- 给出具体可执行建议
- 做出取舍（优先级判断）
你不会逐项罗列，而是只关注最重要的问题。

【TASK】
输入：
- 用户问题（question）
- 商家经营数据（evidence）
你的任务是完成一次完整分析：
1. 判断整体经营状态
2. 识别 2-3 个最关键问题
3. 解释原因
4. 提出最有效建议（最多 3 个）
5. 给客户经理话术
你是在做分析与决策，不是解释数据。

【CRITICAL RULES】
1. 必须从所有指标中选择最影响增长的 2-3 个问题，不能逐项分析。
2. 必须解释为什么这些问题影响订单增长（因果链路）。
3. 必须引用 evidence 中的数据，不能编造事实。
4. 可以推断，但要在 confidence_note 中明确不确定性。
5. 禁止空话，动作必须具体可执行。
6. rule_hints 仅供参考，你可以忽略。
7. recommended_actions 最多 3 条。
8. 输出必须为 JSON，且只输出 JSON。

【FEW-SHOT 示例】
输入（简化）：
- conversion_rate 低于同行
- image_coverage 低于同行

错误分析：
- 逐项罗列所有指标，平均给建议。

正确分析：
- 聚焦核心：转化率是关键问题，图片覆盖不足是直接原因之一。
- 建议：优先补齐热销商品图片，并验证 7 天转化率变化。

请模仿“抓重点 + 做取舍”的方式。

【OUTPUT FORMAT】
{{
  "analysis_type": "diagnosis|root_cause|action_planning|risk_review|opportunity_scan|other",
  "overall_assessment": {{
    "status": "healthy|warning|risk|mixed",
    "summary": "一句总体判断"
  }},
  "key_diagnoses": [
    {{
      "title": "问题标题",
      "description": "解释问题",
      "importance": "high|medium|low",
      "evidence_refs": ["metric_name"]
    }}
  ],
  "recommended_actions": [
    {{
      "action": "具体可执行动作",
      "why_it_matters": "为什么重要",
      "priority": "high|medium|low",
      "related_diagnoses": ["问题标题"]
    }}
  ],
  "talking_points": ["客户经理话术"],
  "confidence_note": "强证据与推断说明"
}}

用户问题：
{question}

evidence:
{json.dumps(evidence, ensure_ascii=False, indent=2)}
""".strip()


def validate_ai_output(result: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Soft validation: only enforce structural availability and basic usability."""
    required_top_fields = {
        "analysis_type",
        "overall_assessment",
        "key_diagnoses",
        "recommended_actions",
        "talking_points",
        "confidence_note",
    }
    missing = [x for x in required_top_fields if x not in result]
    if missing:
        raise ValueError(f"missing fields: {missing}")

    analysis_type = str(result.get("analysis_type", "")).strip()
    if analysis_type not in ALLOWED_ANALYSIS_TYPES:
        raise ValueError("invalid analysis_type")

    overall = result.get("overall_assessment")
    if not isinstance(overall, dict):
        raise ValueError("overall_assessment must be object")
    status = str(overall.get("status", "")).strip().lower()
    if status not in ALLOWED_OVERALL_STATUS:
        raise ValueError("invalid overall_assessment.status")
    if not isinstance(overall.get("summary"), str) or not overall.get("summary", "").strip():
        raise ValueError("overall_assessment.summary is required")

    key_diagnoses = result.get("key_diagnoses")
    if not isinstance(key_diagnoses, list):
        raise ValueError("key_diagnoses must be list")
    if len(key_diagnoses) > 5:
        raise ValueError("key_diagnoses must contain at most 5 items")
    allowed_refs = set((evidence.get("metrics") or {}).keys()) | set((evidence.get("peer_comparison") or {}).keys())
    for item in key_diagnoses:
        if not isinstance(item, dict):
            raise ValueError("diagnosis item must be object")
        if not isinstance(item.get("title"), str) or not item.get("title", "").strip():
            raise ValueError("diagnosis.title is required")
        if not isinstance(item.get("description"), str):
            raise ValueError("diagnosis.description must be str")
        item["importance"] = _normalize_level(item.get("importance", "medium"), ALLOWED_IMPORTANCE)
        refs = item.get("evidence_refs", [])
        if not isinstance(refs, list):
            raise ValueError("diagnosis.evidence_refs must be list")
        if any((not isinstance(ref, str)) or (ref not in allowed_refs) for ref in refs):
            raise ValueError("diagnosis.evidence_refs must reference evidence metric names")

    actions = result.get("recommended_actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("recommended_actions must be non-empty list")
    if len(actions) > 3:
        raise ValueError("recommended_actions must contain at most 3 items")
    for item in actions:
        if not isinstance(item, dict):
            raise ValueError("action item must be object")
        if not isinstance(item.get("action"), str) or not item.get("action", "").strip():
            raise ValueError("action.action is required")
        if not isinstance(item.get("why_it_matters"), str) or not item.get("why_it_matters", "").strip():
            raise ValueError("action.why_it_matters is required")
        item["priority"] = _normalize_level(item.get("priority", "medium"), ALLOWED_IMPORTANCE)
        related = item.get("related_diagnoses", [])
        if not isinstance(related, list):
            raise ValueError("action.related_diagnoses must be list")

    talking_points = result.get("talking_points")
    if not isinstance(talking_points, list) or not talking_points:
        raise ValueError("talking_points must be non-empty list")
    if any(not isinstance(x, str) or not x.strip() for x in talking_points):
        raise ValueError("talking_points must be list[str]")

    if not isinstance(result.get("confidence_note"), str) or not result.get("confidence_note", "").strip():
        raise ValueError("confidence_note is required")

    return result


def analyze_with_llm(
    question: str,
    evidence: Dict[str, Any],
    model: str = "gpt-4.1",
    mock_mode: bool = True,
) -> Dict[str, Any]:
    if mock_mode:
        return fallback_rule_answer(question=question, evidence=evidence, reason="mock_mode=True")

    prompt = build_ai_analyst_prompt(question=question, evidence=evidence)
    estimated_tokens = estimate_text_tokens(prompt, model=model)

    try:
        client = create_openai_client()
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "你是一名资深外卖平台商家增长分析专家。"
                        "你只返回合法 JSON，不输出其他文字。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        payload = json.loads(extract_json_object(response.output_text))
        validated = validate_ai_output(payload, evidence=evidence)
        usage = getattr(response, "usage", None)
        return {
            **_decorate_analysis_result(validated),
            "mode": "real_llm",
            "estimated_input_tokens": estimated_tokens,
            "actual_input_tokens": getattr(usage, "input_tokens", None) if usage else None,
            "actual_output_tokens": getattr(usage, "output_tokens", None) if usage else None,
            "actual_total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        }
    except Exception as exc:
        fallback = fallback_rule_answer(
            question=question,
            evidence=evidence,
            reason=f"LLM fallback: {exc}",
        )
        fallback["estimated_input_tokens"] = estimated_tokens
        return fallback


def generate_answer_with_llm(
    question: str,
    structured_output: Dict[str, Any],
    model: str = "gpt-4.1",
    mock_mode: bool = True,
    evidence_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evidence = evidence_override or build_analysis_evidence(question=question, structured_output=structured_output)
    return analyze_with_llm(question=question, evidence=evidence, model=model, mock_mode=mock_mode)


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
    exec_plan = _planner_to_execution_plan(plan_result)

    if plan_result["analysis_type"] == "other":
        return {
            "brand_id": brand_id,
            "brand_name": str(brand_row["brand_name"]),
            "question": question,
            "intent_result": {
                "analysis_type": plan_result["analysis_type"],
                "reason": plan_result["reason"],
                "mode": plan_result["mode"],
            },
            "plan_result": {
                "analysis_type": plan_result["analysis_type"],
                "analysis_goal": plan_result.get("analysis_goal", ""),
                "focus_areas": plan_result.get("focus_areas", []),
                "suggested_evidence_priority": plan_result.get("suggested_evidence_priority", []),
                "reason": plan_result["reason"],
                "mode": plan_result["mode"],
            },
            "intercepted": True,
            "structured_output": None,
            "analysis_result": {
                "analysis_type": "other",
                "overall_assessment": {
                    "status": "warning",
                    "summary": IRRELEVANT_RESPONSE_TEXT,
                },
                "key_diagnoses": [],
                "recommended_actions": [],
                "talking_points": ["请继续提问商家经营相关问题。"],
                "confidence_note": "问题与经营分析无关，未进入 AI 诊断流程。",
                "summary": IRRELEVANT_RESPONSE_TEXT,
                "diagnoses": [],
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

    # LLM-led analysis path: compute evidence first, let AI analyst make decisions.
    structured_output = run_pipeline(
        merchant_row=brand_row,
        question=question,
        peer_benchmark=peer_benchmark,
        health_config=health_config,
        gap_config=gap_config,
        plan_override=exec_plan,
    )
    evidence = build_analysis_evidence(question=question, structured_output=structured_output)
    analysis_result = generate_answer_with_llm(
        question=question,
        structured_output=structured_output,
        model=analysis_model,
        mock_mode=mock_mode,
        evidence_override=evidence,
    )
    return {
        "brand_id": brand_id,
        "brand_name": str(brand_row["brand_name"]),
        "question": question,
        "intent_result": {
            "analysis_type": plan_result["analysis_type"],
            "reason": plan_result["reason"],
            "mode": plan_result["mode"],
        },
        "plan_result": {
            "analysis_type": plan_result["analysis_type"],
            "analysis_goal": plan_result.get("analysis_goal", ""),
            "focus_areas": plan_result.get("focus_areas", []),
            "suggested_evidence_priority": plan_result.get("suggested_evidence_priority", []),
            "reason": plan_result["reason"],
            "mode": plan_result["mode"],
        },
        "structured_output": structured_output,
        "analysis_evidence": evidence,
        "ai_analyst_output": {
            "analysis_type": analysis_result.get("analysis_type"),
            "overall_assessment": analysis_result.get("overall_assessment", {}),
            "key_diagnoses": analysis_result.get("key_diagnoses", []),
            "recommended_actions": analysis_result.get("recommended_actions", []),
            "talking_points": analysis_result.get("talking_points", []),
            "confidence_note": analysis_result.get("confidence_note", ""),
            "mode": analysis_result.get("mode"),
        },
        "analysis_result": analysis_result,
    }
