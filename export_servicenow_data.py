"""
export_servicenow_data.py
--------------------------
Exports your OWN resolved/closed tickets from ServiceNow
into a CSV file ready for model training.

Pulls these fields per ticket:
  short_description, description, category, subcategory,
  business_service, priority, assignment_group

Run:
    python export_servicenow_data.py
    python export_servicenow_data.py --limit 2000
    python export_servicenow_data.py --output data/my_tickets.csv
"""

import csv
import os
import sys
import time
import yaml
import argparse
import requests
from requests.exceptions import RequestException

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

INSTANCE = config["servicenow"]["instance_url"].rstrip("/")
AUTH     = (config["servicenow"]["username"], config["servicenow"]["password"])
HEADERS  = {"Accept": "application/json", "Content-Type": "application/json"}

FIELDS = [
    "short_description", "description", "category",
    "subcategory", "business_service", "priority", "assignment_group"
]

CSV_COLUMNS = FIELDS  # same order in output CSV


def test_connection():
    print(f"\n{BOLD}Connecting to {INSTANCE}...{RESET}")
    try:
        r = requests.get(
            f"{INSTANCE}/api/now/table/incident",
            auth=AUTH, headers=HEADERS,
            params={"sysparm_limit": 1}, timeout=15
        )
        if r.status_code == 200:
            print(f"  {GREEN}✅ Connected{RESET}")
            return True
        print(f"  {RED}❌ HTTP {r.status_code}: {r.text[:200]}{RESET}")
    except RequestException as e:
        print(f"  {RED}❌ {e}{RESET}")
    return False


def fetch_page(offset: int, limit: int, state_filter: str) -> list:
    """Fetch one page of resolved/closed tickets."""
    params = {
        "sysparm_query":         f"{state_filter}^assignment_groupISNOTEMPTY",
        "sysparm_fields":        ",".join(FIELDS),
        "sysparm_limit":         limit,
        "sysparm_offset":        offset,
        "sysparm_display_value": "true",
        "sysparm_exclude_reference_link": "true",
    }
    r = requests.get(
        f"{INSTANCE}/api/now/table/incident",
        auth=AUTH, headers=HEADERS, params=params, timeout=30
    )
    r.raise_for_status()
    return r.json().get("result", [])


def clean(value) -> str:
    """Extract display value if dict, strip whitespace."""
    if isinstance(value, dict):
        return (value.get("display_value") or value.get("value") or "").strip()
    return str(value or "").strip()


def export(output_path: str, max_records: int):
    print(f"\n{BOLD}{'═'*58}{RESET}")
    print(f"{BOLD}  ServiceNow Training Data Exporter{RESET}")
    print(f"{BOLD}{'═'*58}{RESET}")
    print(f"  Instance    : {BOLD}{INSTANCE}{RESET}")
    print(f"  Output file : {BOLD}{output_path}{RESET}")
    print(f"  Max records : {BOLD}{max_records}{RESET}")
    print()

    # ServiceNow state codes: 6=Resolved, 7=Closed
    state_filter = "state=6^ORstate=7"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total_written = 0
    page_size     = 100
    offset        = 0
    skipped       = 0
    group_counts  = {}

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        print(f"  Fetching tickets (page size={page_size})...\n")

        while total_written < max_records:
            batch = fetch_page(offset, page_size, state_filter)
            if not batch:
                break

            for raw in batch:
                if total_written >= max_records:
                    break

                short_desc = clean(raw.get("short_description", ""))
                group      = clean(raw.get("assignment_group", ""))

                # Skip rows missing essential fields
                if not short_desc or not group:
                    skipped += 1
                    continue

                row = {col: clean(raw.get(col, "")) for col in CSV_COLUMNS}
                writer.writerow(row)
                total_written += 1
                group_counts[group] = group_counts.get(group, 0) + 1

            offset += page_size
            pct = min(100, int(total_written / max_records * 100))
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {total_written} records", end="", flush=True)

            if len(batch) < page_size:
                break   # Last page

            time.sleep(0.1)  # Be polite to the API

    print(f"\n\n  {GREEN}{'═'*56}{RESET}")
    print(f"  {GREEN}{BOLD}Export complete!{RESET}")
    print(f"  Records exported : {GREEN}{BOLD}{total_written}{RESET}")
    print(f"  Rows skipped     : {YELLOW}{skipped}{RESET}  (missing fields)")
    print()
    print(f"  {BOLD}Assignment Group Breakdown:{RESET}")
    for grp, cnt in sorted(group_counts.items(), key=lambda x: -x[1]):
        bar_len = int(cnt / max(group_counts.values()) * 30)
        bar = "█" * bar_len
        print(f"    {grp:<35} {bar:<30} {cnt}")

    if total_written < 50:
        print(f"\n  {YELLOW}⚠️  Only {total_written} records found.{RESET}")
        print(f"  {YELLOW}   The developer instance may only have the 15 tickets you just created.{RESET}")
        print(f"  {YELLOW}   See instructions below for adding more training data.{RESET}")
    elif total_written < 200:
        print(f"\n  {YELLOW}⚠️  {total_written} records is small — model accuracy may be limited.{RESET}")
        print(f"  {YELLOW}   Aim for 200+ resolved tickets per assignment group for best results.{RESET}")
    else:
        print(f"\n  {GREEN}✅  Good dataset size for training.{RESET}")

    print(f"\n  {BOLD}Next step:{RESET}")
    print(f"    {CYAN}python train_own_model.py --csv {output_path}{RESET}")
    print()

    return total_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ServiceNow tickets to CSV for training")
    parser.add_argument("--output", default="data/my_servicenow_tickets.csv")
    parser.add_argument("--limit",  type=int, default=5000, help="Max records to export")
    args = parser.parse_args()

    if not test_connection():
        sys.exit(1)

    export(args.output, args.limit)
