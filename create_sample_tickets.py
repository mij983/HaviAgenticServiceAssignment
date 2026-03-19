import sys
import time
import yaml
import requests
from requests.exceptions import RequestException

sys.path.insert(0, __import__("os").path.dirname(__file__))

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

INSTANCE = cfg["servicenow"]["instance_url"].rstrip("/")
AUTH     = (cfg["servicenow"]["username"], cfg["servicenow"]["password"])
HEADERS  = {"Content-Type": "application/json", "Accept": "application/json"}

SAMPLE_TICKETS = [
    # IT-SC Operations Application Support
    {"short_description": "VPN connection dropping after 10 minutes", "description": "VPN connection dropping after 10 minutes", "priority": "3"},
]


def test_connection():
    try:
        r = requests.get(INSTANCE + "/api/now/table/incident",
            auth=AUTH, headers=HEADERS, params={"sysparm_limit": 1}, timeout=15)
        if r.status_code == 200:
            print("  [OK] Connected to " + INSTANCE)
            return True
        print("  [ERROR] Status " + str(r.status_code))
    except RequestException as e:
        print("  [ERROR] " + str(e))
    return False


def create_ticket(ticket, index, total):
    payload = {
        "short_description": ticket["short_description"],
        "description":       ticket["description"],
        "category":          "Applications and Software",
        "subcategory":       "Business Application",
        "priority":          ticket.get("priority", "3"),
        "state":             "1",
    }
    try:
        r = requests.post(INSTANCE + "/api/now/table/incident",
            auth=AUTH, headers=HEADERS, json=payload, timeout=15)
        r.raise_for_status()
        number = r.json()["result"].get("number", "N/A")
        print("  [{:02d}/{:02d}] [OK]  {}  {}".format(index, total, number, ticket["short_description"][:60]))
        return True
    except Exception as e:
        print("  [{:02d}/{:02d}] [ERROR]  FAILED -- {}".format(index, total, e))
        return False


def main():
    print("")
    print("=" * 60)
    print("  ServiceNow Ticket Creator -- 50 Real Team Tickets")
    print("  Routing by Short Description + Description Only")
    print("=" * 60)
    print("  Instance : " + INSTANCE)
    print("")

    if not test_connection():
        sys.exit(1)

    print("")
    print("  Creating " + str(len(SAMPLE_TICKETS)) + " tickets (all unassigned)...")
    print("")
    ok = 0
    for i, t in enumerate(SAMPLE_TICKETS, 1):
        if create_ticket(t, i, len(SAMPLE_TICKETS)):
            ok += 1
        time.sleep(0.3)

    failed = len(SAMPLE_TICKETS) - ok
    print("")
    print("  " + str(ok) + " created  /  " + str(failed) + " failed")
    print("")
    print("  Next:  python run_agent.py")
    print("")


if __name__ == "__main__":
    main()
