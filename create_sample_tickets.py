"""
create_sample_tickets.py
-------------------------
Creates all 50 tickets from your CSV into ServiceNow — all UNASSIGNED.
The AI agent will read each short_description + description and route
them to the correct assignment team.

Short descriptions are taken VERBATIM from the CSV.
Configuration Item excluded as requested.

Run:
    python create_sample_tickets.py
"""

import sys, time, yaml, requests
from requests.exceptions import RequestException

GREEN="\033[92m"; RED="\033[91m"; CYAN="\033[96m"; BOLD="\033[1m"; RESET="\033[0m"

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

INSTANCE = cfg["servicenow"]["instance_url"].rstrip("/")
AUTH     = (cfg["servicenow"]["username"], cfg["servicenow"]["password"])
HEADERS  = {"Content-Type": "application/json", "Accept": "application/json"}


# ── All 50 tickets verbatim from CSV (SD exact, description cleaned) ──────────
# Expected team in comment → AI must predict this from SD + description only.
SAMPLE_TICKETS = [

    # ── IT-SC Operations Application Support ─────────────────────────────────
    {
        "short_description": "INFOR: Issues on routes 0267R2611 and 5076R2611",
        "description": "Two sales orders were allocated into route 0 instead of route 7062R2611. Crossdocking orders for DC Porto did not go through. Load and invoice blocked. Issue at DC Porto but needs solving at DC Azambuja too.",
        "priority": "3",
    },  # → IT-SC Operations Application Support
    {
        "short_description": "Kisoft - Panel weight",
        "description": "Issues with weight of Kisoft panels. Panel P-1290-3200-2350-4-1-1-P at location 4003110001. Max capacity 4000kg, available 834kg. Error when storing SSCC of 640kg even though within limit.",
        "priority": "4",
    },  # → IT-SC Operations Application Support
    {
        "short_description": "Reopening REQ0263861 - INFOR USER ROCIO MENOR",
        "description": "Ticket REQ0263861 was opened for user creation of ROCIO MENOR. User now appears blocked. Please unblock the user in INFOR WMS.",
        "priority": "4",
    },  # → IT-SC Operations Application Support
    {
        "short_description": "INFOR: Issues on route 5322R2611",
        "description": "Route 5322R2611 did not close correctly. Load shows closed but shipment orders remain in loaded status. ILOS did not receive load closed information.",
        "priority": "3",
    },  # → IT-SC Operations Application Support
    {
        "short_description": "Infor muito lento",
        "description": "WMS INFOR application is extremely slow. Workers cannot perform tasks. Orders sent at 16h and more than half not processed after two hours.",
        "priority": "3",
    },  # → IT-SC Operations Application Support
    {
        "short_description": "HPT-500-INFOR-006",
        "description": "Device factory reset was performed incorrectly. Device deleted from Scale Fusion. EUS Team has not responded for a week. Devices INC0830833 and INC0816583 stopped.",
        "priority": "3",
    },  # → IT-SC Operations Application Support
    {
        "short_description": "KISOFT 3216 needs to be opened again by KNAPP",
        "description": "Blocking 3216 in Kisoft was closed without releasing goods. Please ask KNAPP to reopen blocking 3216 so goods can be unblocked.",
        "priority": "3",
    },  # → IT-SC Operations Application Support
    {
        "short_description": "ES SAN FERNANDO ERROR KISOFT STAGE DAR UBICACION ARTICULO tipo stock",
        "description": "Cannot assign location to a new article for a new customer. When selecting stock type, Kisoft closes directly without saving.",
        "priority": "3",
    },  # → IT-SC Operations Application Support
    {
        "short_description": "ES SAN FERNANDO - XML MESSAGE KISOFT SENDS TO SYSTORE TO REPLENISH P-LOC 4050410001 IS NOT WORKING",
        "description": "XML message Kisoft sends to Systore for replenishing p-location 4050410001 shows wrong value in target location field. Should be PD141_A but shows incorrect value.",
        "priority": "3",
    },  # → IT-SC Operations Application Support
    {
        "short_description": "ICH KANN NICHT MEHR AUS KISOFT SEIT 3 TAGEN DRUCKEN!!",
        "description": "Cannot print from Kisoft for 3 days. I CAN'T PRINT FROM KISOFT FOR 3 DAYS ANYMORE!! All print jobs failing.",
        "priority": "4",
    },  # → IT-SC Operations Application Support

    # ── IT-SC GBS App Support ─────────────────────────────────────────────────
    {
        "short_description": "Unable to Access Dims on my Laptop",
        "description": "Seeking assistance to check my Dims application concern. Cannot access the application on laptop.",
        "priority": "4",
    },  # → IT-SC GBS App Support
    {
        "short_description": "DIMS",
        "description": "Requesting access to shared path to proceed with printing. Cannot print document when finishing inputting details in DIMS.",
        "priority": "4",
    },  # → IT-SC GBS App Support
    {
        "short_description": "Mcdo Dims",
        "description": "Good morning. My Mcdo Dims have some module are not working.",
        "priority": "4",
    },  # → IT-SC GBS App Support
    {
        "short_description": "Dims error message",
        "description": "Cannot access local application DIMS. Not just DC Plaridel, also DC Cabuyao and users working at home in Philippines.",
        "priority": "4",
    },  # → IT-SC GBS App Support
    {
        "short_description": "can not open my dims on a desk top mostly all user here in plaridel",
        "description": "Cannot open DIMS on desktop. All users at Plaridel site affected.",
        "priority": "3",
    },  # → IT-SC GBS App Support
    {
        "short_description": "Incorrect username in ILOS",
        "description": "Issue with employee signature in ILOS 1.0. Log in with KRA091 but different signature KRZ is displayed inside the application.",
        "priority": "4",
    },  # → IT-SC GBS App Support
    {
        "short_description": "Die Tour 1542 vom 18.02.26 ist noch im System in ILOS. Die Kunden koennen keinen Wareneingang machen. In KI-Soft wurde die Tour bestaetigt.",
        "description": "Tour 1542 from February 18 2026 is still in the ILOS system. Customers are unable to process a goods receipt. The tour has been confirmed in KiSoft. There appears to be a communication problem between the two programs.",
        "priority": "3",
    },  # → IT-SC GBS App Support
    {
        "short_description": "Boomi ILOS Sourcing : Error received while processing startLoad from ILOS to INFOR for LoadID :5107R2601",
        "description": "Boomi encountered an error in the StartLoad process ILOS-HDE-500-DistStartLoad-5107R2601. Message: An invalid OrderKey value was supplied. Pick Order 3191278 failed in INFOR. QtyPick value is negative in XML from ILOS.",
        "priority": "4",
    },  # → IT-SC GBS App Support
    {
        "short_description": "ILOS",
        "description": "Urgently need sales orders 59257006, 59257099, 59257145 recorded under client 170 transferred to client 1700. Truck cannot leave without correct invoice for the delivery.",
        "priority": "3",
    },  # → IT-SC GBS App Support
    {
        "short_description": "Open routes in the INFOR",
        "description": "Routes in INFOR are closed but remain open in ILOS. Unable to invoice. Problem at Azambuja DC.",
        "priority": "3",
    },  # → IT-SC GBS App Support

    # ── IT-Asia-SOA-Central-App-Support ──────────────────────────────────────
    {
        "short_description": "IT Daily Sales report for McD need exclude item category 9900045",
        "description": "Please set up MCD04 to exclude non-item 99000 in SOA IT Daily Sales Report.",
        "priority": "3",
    },  # → IT-Asia-SOA-Central-App-Support
    {
        "short_description": "No data downloaded in SOA",
        "description": "No data is downloaded on the Item Store Delivery Report in SOA Reports menu.",
        "priority": "3",
    },  # → IT-Asia-SOA-Central-App-Support
    {
        "short_description": "TH DN EDI missed reference number",
        "description": "HAVI Thailand identified issue with Delivery Note in SOA system. Last page does not display reference document number when it has no continued content. Documentation team cannot map document back.",
        "priority": "4",
    },  # → IT-Asia-SOA-Central-App-Support
    {
        "short_description": "need your help to check this TIO # TI2-02695(SOA) was not appear at WMS system",
        "description": "TIO TI2-02695 from SOA not appearing in WMS system. Issue may be due to 15 day date range before and after ETA request date.",
        "priority": "4",
    },  # → IT-Asia-SOA-Central-App-Support
    {
        "short_description": "DN number MCDDN1-090767697 can't download pdf at menu Print DN/Invoice in SOA",
        "description": "DN number MCDDN1-090767697 when printing PDF in SOA on menu Print DN/Invoice shows blank result.",
        "priority": "3",
    },  # → IT-Asia-SOA-Central-App-Support

    # ── IT-SC-EPAM-SAP Workflow ───────────────────────────────────────────────
    {
        "short_description": "Workflow Error in VIM",
        "description": "Error when trying to approve an invoice in VIM. Error in determining the next approver. Vendor: Caritas Werkstaetten Niederrhein 1900033931.",
        "priority": "3",
    },  # → IT-SC-EPAM-SAP Workflow
    {
        "short_description": "VIM sehr langsam",
        "description": "Every loading operation in VIM (save or approve) takes approximately 30 to 60 seconds to process.",
        "priority": "3",
    },  # → IT-SC-EPAM-SAP Workflow
    {
        "short_description": "invoices from IC4S are not submitted to VIM",
        "description": "After validation, invoices from IC4S are not submitted to VIM. Please solve the problem.",
        "priority": "3",
    },  # → IT-SC-EPAM-SAP Workflow
    {
        "short_description": "VIM 1008 - id 3314851 - workflow to be cancelled",
        "description": "Please cancel the workflow of ID 3314851 or reassign to AP team. Invoice already cleared but still in VIM. User cannot approve or reject it.",
        "priority": "4",
    },  # → IT-SC-EPAM-SAP Workflow
    {
        "short_description": "URGENT Invoices blocked in VIM after booking, invoices are not coming to payment",
        "description": "Invoices stuck in VIM after booking. Coding is correct. SAP settings for profile MBA842 need checking. Invoices are not coming to payment.",
        "priority": "3",
    },  # → IT-SC-EPAM-SAP Workflow

    # ── IT-SC-EPAM-SAP-AMS-Support ────────────────────────────────────────────
    {
        "short_description": "NO ZEDI OUTPUT in SAP",
        "description": "No ZEDI output being generated from SAP PC4 for debtor 6266 since 26.1.2026. Regenerated via VF31 for past invoices but automatic ZEDI still not generating.",
        "priority": "4",
    },  # → IT-SC-EPAM-SAP-AMS-Support
    {
        "short_description": "Shell price mass upload - 56 termek Hell",
        "description": "Please upload the attached prices to SAP and ILOS. Assign the incident to AMS IT Team.",
        "priority": "3",
    },  # → IT-SC-EPAM-SAP-AMS-Support
    {
        "short_description": "I cannot generate mass invoices from SAP",
        "description": "Cannot generate mass invoices from SAP. Feature not working.",
        "priority": "3",
    },  # → IT-SC-EPAM-SAP-AMS-Support
    {
        "short_description": "SAP - Invoices with ZEDI output processed but not delivered to Archiva",
        "description": "ZEDI output for several invoices was processed but not delivered to Archiva. Please check B2Bi configuration.",
        "priority": "4",
    },  # → IT-SC-EPAM-SAP-AMS-Support
    {
        "short_description": "Bitte Ticket in SAP C4C",
        "description": "Bitte Ticket in SAP C4C - Ticket Nr. 7813759 entpersonalisieren. Wir sind verpflichtet die Daten zu loeschen.",
        "priority": "4",
    },  # → IT-SC-EPAM-SAP-AMS-Support

    # ── IT-SC-EPAM-SAP Hybris CRM Support ────────────────────────────────────
    {
        "short_description": "Bitte Ticket in SAP C4C - Ticket Nr. 7813759 entpersonalisieren. Wir sind verpflichtet die Daten zu loeschen.",
        "description": "Bitte Ticket in SAP C4C - Ticket Nr. 7813759 entpersonalisieren. Wir sind verpflichtet die Daten zu loeschen (Anhaenge).",
        "priority": "4",
    },  # → IT-SC-EPAM-SAP Hybris CRM Support
    {
        "short_description": "Delete/remove C4C Ticket 7804716 completely. Contains sensitive data.",
        "description": "Please delete and remove C4C Ticket 7804716 completely. Contains sensitive data.",
        "priority": "4",
    },  # → IT-SC-EPAM-SAP Hybris CRM Support
    {
        "short_description": "Delete/remove C4C Ticket 7794713 completely. Contains sensitive data.",
        "description": "Delete and remove C4C Ticket 7794713 completely. Contains sensitive data.",
        "priority": "4",
    },  # → IT-SC-EPAM-SAP Hybris CRM Support
    {
        "short_description": "HAVI C4H Active User Report",
        "description": "Latest HAVI C4H Active User Report is missing the last logon date column. Please send new report adding this column.",
        "priority": "4",
    },  # → IT-SC-EPAM-SAP Hybris CRM Support
    {
        "short_description": "C4C email function not working in the Nordics",
        "description": "Emails sent from C4C are not visible in the errands. Cannot see sent emails. When a response arrives, a new ticket is opened. Affects all Nordic markets.",
        "priority": "3",
    },  # → IT-SC-EPAM-SAP Hybris CRM Support

    # ── IT-Portal-Central ─────────────────────────────────────────────────────
    {
        "short_description": "Haviconnectestaurant 12050 zabok",
        "description": "Restaurant 12050 zabok in HaviConnect submits claims but they are not sent via email like other restaurants. Please configure email notifications for this restaurant.",
        "priority": "4",
    },  # → IT-Portal-Central
    {
        "short_description": "Issue with HaviConnect Website",
        "description": "Cannot enter the HaviConnect website. Neither internal users nor customers can access the portal.",
        "priority": "3",
    },  # → IT-Portal-Central
    {
        "short_description": "HAVIConnect - Order proposal corrections not transferred into final order (multiple items)",
        "description": "Restaurant DE MCD #1784 reports manually corrected order proposal quantities in HAVIConnect were not transferred into the final order. Restaurant received incorrect quantities.",
        "priority": "4",
    },  # → IT-Portal-Central
    {
        "short_description": "Password HAVI Connect Portugal (Claim +)",
        "description": "Need password to access HAVI Connect supplier claims+ portal for Portugal suppliers.",
        "priority": "3",
    },  # → IT-Portal-Central
    {
        "short_description": "HAVI CONNECT FUNCTIONS disabled for McD SLO customer",
        "description": "Some HAVI CONNECT functions are not working for McD SLO customer users. Need immediate assistance to enable applications.",
        "priority": "3",
    },  # → IT-Portal-Central

    # ── IT-SCM-EPAM L2 ────────────────────────────────────────────────────────
    {
        "short_description": "Missing PMIX data for Portugal Market",
        "description": "75 stores in Portugal did not send PMIX data today. 20 stores have minimal PMIX data. Check if McD sent data correctly.",
        "priority": "3",
    },  # → IT-SCM-EPAM L2
    {
        "short_description": "HLPL - Missing PMIX data - stores 435, 775, 794",
        "description": "Missing PMIX data in PIR and JDA for stores 435, 775, 794 across several days in December. McD IT confirmed data was resent. Please verify and upload correct PMIX data.",
        "priority": "4",
    },  # → IT-SCM-EPAM L2
    {
        "short_description": "JDA - Wrong HIST data for PL-GC",
        "description": "Missing data in JDA HIST tables for PL-GC for 9th December 2025. PMIX data was resent by McD IT. Please verify and upload correct data.",
        "priority": "4",
    },  # → IT-SCM-EPAM L2
    {
        "short_description": "Store 248 - CFM/PMIX",
        "description": "New store 248 opening. McD IT already sending CFM/PMIX data. Please check if data is being received correctly.",
        "priority": "3",
    },  # → IT-SCM-EPAM L2
    {
        "short_description": "Update history in JDA/PIR according to new product numbers (SMI) in PMIX - PL market.",
        "description": "New product numbers SMI added to PMIX today for Poland market. Please update history in JDA/PIR with new SMI product numbers since 26.11.2025.",
        "priority": "3",
    },  # → IT-SCM-EPAM L2
]


def test_connection():
    try:
        r = requests.get(f"{INSTANCE}/api/now/table/incident",
            auth=AUTH, headers=HEADERS, params={"sysparm_limit": 1}, timeout=15)
        if r.status_code == 200:
            print(f"  {GREEN}✅  Connected to {INSTANCE}{RESET}")
            return True
        print(f"  {RED}❌  Status {r.status_code}{RESET}")
    except RequestException as e:
        print(f"  {RED}❌  {e}{RESET}")
    return False


def create_ticket(ticket, index, total):
    payload = {
        "short_description": ticket["short_description"],
        "description":       ticket["description"],
        "category":          "Applications and Software",
        "subcategory":       "Business Application",
        "priority":          ticket.get("priority", "3"),
        "state":             "1",   # New — unassigned, AI routes this
    }
    try:
        r = requests.post(f"{INSTANCE}/api/now/table/incident",
            auth=AUTH, headers=HEADERS, json=payload, timeout=15)
        r.raise_for_status()
        number = r.json()["result"].get("number", "N/A")
        print(f"  [{index:02d}/{total}] {GREEN}✅{RESET}  {number}  {ticket['short_description'][:65]}")
        return True
    except Exception as e:
        print(f"  [{index:02d}/{total}] {RED}❌{RESET}  FAILED — {e}")
        return False


def main():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗
║   ServiceNow Ticket Creator — 50 Real Team Tickets       ║
║   Routing by Short Description + Description Only        ║
╚══════════════════════════════════════════════════════════╝{RESET}
  Instance : {BOLD}{INSTANCE}{RESET}
""")

    if not test_connection():
        sys.exit(1)

    print(f"\n  {BOLD}Creating {len(SAMPLE_TICKETS)} tickets (all unassigned)...{RESET}\n")
    ok = 0
    for i, t in enumerate(SAMPLE_TICKETS, 1):
        if create_ticket(t, i, len(SAMPLE_TICKETS)):
            ok += 1
        time.sleep(0.3)

    failed = len(SAMPLE_TICKETS) - ok
    print(f"\n  {GREEN}{ok} created{RESET}", end="")
    if failed:
        print(f"  {RED}{failed} failed{RESET}", end="")
    print(f"\n\n  {BOLD}Next:{RESET}  python run_agent.py\n")


if __name__ == "__main__":
    main()
