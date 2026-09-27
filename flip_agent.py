"""Screen fictional DFW flip opportunities; optional LLM tool-using analyst."""
import argparse
import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    value = row.get(key, "").strip()
    if not value:
        raise ValueError(f"{row.get('id', 'row')}: missing {key}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{row.get('id', 'row')}: invalid {key}")
    return result


def comparable_sales(property_row, sales, as_of, max_age_days=180):
    """Prefer same ZIP, beds and similar area; widen to city only if necessary."""
    eligible = []
    blocked = []
    area = number(property_row, "sqft")
    if area <= 0:
        raise ValueError("Property square footage must be positive")
    for sale in sales:
        if sale.get("status", "").strip().upper() != "SOLD":
            blocked.append({"id": sale["id"], "reason": "Not marked SOLD"})
            continue
        if not sale.get("sold_price", "").strip():
            blocked.append({"id": sale["id"], "reason": "Verified closing price unavailable; last list price is not a substitute"})
            continue
        sold = date.fromisoformat(sale["sold_date"])
        age = (as_of - sold).days
        if not 0 <= age <= max_age_days:
            continue
        sale_area = number(sale, "sqft")
        if sale_area <= 0 or not 0.8 * area <= sale_area <= 1.2 * area:
            continue
        if sale["city"].casefold() != property_row["city"].casefold():
            continue
        if sale["beds"] != property_row["beds"]:
            continue
        eligible.append(sale)
    exact = [s for s in eligible if s["zip"] == property_row["zip"]]
    chosen = exact if len(exact) >= 3 else eligible
    return chosen, "same ZIP" if len(exact) >= 3 else "same city (ZIP widened)", blocked


def underwrite(row, sales, settings, as_of):
    if row.get("property_type", "").strip().lower() != "single_family":
        return {"id": row["id"], "address": row["address"], "status": "OUT_OF_SCOPE",
                "reason": "This model screens existing single-family homes only"}
    comps, comp_scope, blocked = comparable_sales(row, sales, as_of)
    findings = {"id": row["id"], "city": row["city"], "address": row["address"],
                "listing_url": row.get("listing_url", ""), "comps_used": [s["id"] for s in comps],
                "comp_scope": comp_scope, "excluded_comp_rows": blocked, "as_of": as_of.isoformat()}
    if len(comps) < 3:
        return {**findings, "status": "NEEDS_DATA", "reason": "Fewer than 3 recent, size- and bedroom-matched sold comps"}

    ask = number(row, "asking_price")
    sqft = number(row, "sqft")
    rehab = number(row, "renovation_estimate")
    annual_tax = number(row, "annual_property_tax")
    insurance_monthly = number(row, "insurance_monthly")
    hoa_monthly = number(row, "hoa_monthly")
    utilities_monthly = number(row, "utilities_monthly")
    months = float(settings["hold_months"])
    purchase_close_rate = float(settings["purchase_closing_pct"])
    sale_cost_rate = float(settings["selling_cost_pct"])
    contingency_rate = float(settings["rehab_contingency_pct"])
    loan_fraction = float(settings["loan_to_purchase_pct"])
    annual_interest = float(settings["annual_interest_pct"])
    points_rate = float(settings["loan_points_pct"])
    min_profit = float(settings["min_profit"])
    min_roi = float(settings["min_roi_pct"])
    if not 0 <= loan_fraction <= 1 or not 0 <= sale_cost_rate < 1:
        raise ValueError("Invalid financing or selling cost percentage")
    if months <= 0 or min_profit < 0 or min_roi < 0:
        raise ValueError("Invalid hold months or return target")

    prices_per_sqft = [number(s, "sold_price") / number(s, "sqft") for s in comps]
    arv = median(prices_per_sqft) * sqft
    rehab_total = rehab * (1 + contingency_rate)
    holding = months * (annual_tax / 12 + insurance_monthly + hoa_monthly + utilities_monthly)

    def deal(price, resale):
        loan = price * loan_fraction
        closing = price * purchase_close_rate
        financing = loan * (annual_interest * months / 12 + points_rate)
        selling = resale * sale_cost_rate
        total_cost = price + closing + rehab_total + holding + financing + selling
        profit = resale - total_cost
        # Cash invested is simplified: loan principal is repaid at sale; interest and points are costs.
        cash_in = price - loan + closing + rehab_total + holding + financing
        roi = profit / cash_in if cash_in > 0 else None
        return {"profit": round(profit), "roi_pct": round(100 * roi, 1) if roi is not None else None,
                "total_cost": round(total_cost), "cash_in": round(cash_in)}

    base = deal(ask, arv)
    downside = deal(ask, 0.9 * arv)
    # Linear cost in offer. Solve the offer at the minimum absolute profit target.
    offer_cost_multiplier = 1 + purchase_close_rate + loan_fraction * (annual_interest * months / 12 + points_rate)
    max_offer = ((1 - sale_cost_rate) * arv - rehab_total - holding - min_profit) / offer_cost_multiplier
    if base["profit"] >= min_profit and base["roi_pct"] is not None and base["roi_pct"] >= 100 * min_roi:
        status = "REVIEW"
    else:
        status = "PASS_AT_ASK"
    if row.get("red_flags", "").strip():
        status = "DUE_DILIGENCE" if status == "REVIEW" else status
    return {**findings, "status": status, "ask": round(ask), "arv_estimate": round(arv),
            "arv_basis": "Median sold price per sqft of selected comps × subject sqft",
            "rehab_with_contingency": round(rehab_total), "holding_cost": round(holding),
            "max_offer_for_min_profit": round(max_offer), "base": base, "resale_10pct_lower": downside,
            "red_flags": row.get("red_flags", ""), "comp_sales_dates": [s["sold_date"] for s in comps]}


def run_analysis(properties_path, sales_path, settings_path, as_of):
    properties = read_csv(properties_path)
    sales = read_csv(sales_path)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    ids = [p["id"] for p in properties]
    if len(ids) != len(set(ids)):
        raise ValueError("Property IDs must be unique")
    return [underwrite(p, sales, settings, as_of) for p in properties]


def ai_review(results):
    """Model chooses tools for evidence and writes a triage memo; never does the math."""
    from agents import Agent, Runner, function_tool
    by_id = {r["id"]: r for r in results}

    @function_tool
    def list_properties() -> str:
        """List property IDs and computed screening statuses."""
        return json.dumps([{"id": r["id"], "status": r["status"]} for r in results])

    @function_tool
    def inspect_underwriting(property_id: str) -> str:
        """Get deterministic underwriting, comparable evidence and flags for a property ID."""
        return json.dumps(by_id.get(property_id, {"error": "Unknown property ID"}))

    analyst = Agent(
        name="DFW Flip Triage Analyst",
        model="gpt-4.1-mini",
        instructions=("You screen hypothetical DFW flips for a human. First list_properties, "
                      "then inspect_underwriting for each candidate. Make no new dollar calculations. "
                      "Cite the property IDs and comp IDs supplied by tools. Treat ARV as an estimate, "
                      "not an appraisal. Never recommend buying without inspections, scope, financing, "
                      "tax, title, insurance, and verified sold comps. Rank REVIEW ahead of PASS_AT_ASK; "
                      "explain downside and exact missing evidence. Do not invent listings or values."),
        tools=[list_properties, inspect_underwriting],
    )
    return Runner.run_sync(analyst, "Prepare a short acquisition screening memo for these properties.").final_output


def main():
    parser = argparse.ArgumentParser(description="DFW flip underwriting with optional AI triage")
    parser.add_argument("--properties", type=Path, default=ROOT / "sample_properties.csv")
    parser.add_argument("--sales", type=Path, default=ROOT / "sample_sold_comps.csv")
    parser.add_argument("--settings", type=Path, default=ROOT / "assumptions.json")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--ai", action="store_true", help="Call OpenAI after deterministic analysis")
    parser.add_argument("--output", type=Path, default=ROOT / "flip_results.json")
    args = parser.parse_args()
    results = run_analysis(args.properties, args.sales, args.settings, args.as_of)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for r in results:
        if "base" in r:
            print(f"{r['id']}: {r['status']} | ask ${r['ask']:,} | estimated ARV ${r['arv_estimate']:,} | estimated profit ${r['base']['profit']:,}")
        else:
            print(f"{r['id']}: {r['status']} | {r['reason']}")
    print(f"Saved {args.output}")
    if args.ai:
        print("\nAI TRIAGE MEMO\n" + str(ai_review(results)))


if __name__ == "__main__":
    main()
