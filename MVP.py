from merchant_growth_mvp import (
    GAP_CONFIG,
    HEALTH_CONFIG,
    create_mock_inputs,
    get_merchant_row,
    render_report,
    run_llm_orchestrated_pipeline,
)


def main() -> None:
    merchant_df, peer_benchmark = create_mock_inputs()
    merchant_id = "M001"
    question = "这家商家最近经营怎么样？我应该建议他做什么？"

    merchant_row = get_merchant_row(merchant_df, merchant_id)
    result = run_llm_orchestrated_pipeline(
        merchant_row=merchant_row,
        question=question,
        peer_benchmark=peer_benchmark,
        health_config=HEALTH_CONFIG,
        gap_config=GAP_CONFIG,
        mock_mode=True,
    )

    print("=== Intent Result ===")
    print(result["intent_result"])
    print()

    print("=== Structured Output Report ===")
    render_report(
        {
            "merchant_id": result["structured_output"]["merchant_id"],
            "merchant_name": result["structured_output"]["merchant_name"],
            "question": result["structured_output"]["question"],
            "health": result["structured_output"].get(
                "health",
                {
                    "status": "N/A",
                    "summary": "当前 intent 未要求输出 health 模块。",
                },
            ),
            "gaps_by_driver": {
                driver: [
                    {
                        "metric": item["metric"],
                        "value": item["value"],
                        "percentile": item["percentile"],
                        "label": item["label"],
                    }
                    for item in result["structured_output"].get("gaps", [])
                    if item["driver"] == driver
                ]
                for driver in ["traffic", "ops_readiness", "user_experience", "supply_quality"]
            },
            "recommended_actions": result["structured_output"].get("actions", []),
        }
    )
    print()

    print("=== Controlled LLM Output ===")
    print(result["analysis_result"]["text"])
    print()
    print("Estimated input tokens:", result["analysis_result"]["estimated_input_tokens"])
    print("Mode:", result["analysis_result"]["mode"])


if __name__ == "__main__":
    main()
