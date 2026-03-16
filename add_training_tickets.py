"""
add_training_tickets.py
------------------------
Adds a LARGE set of realistic resolved tickets to your ServiceNow
developer instance — pre-assigned to groups so they can be exported
and used as training data for your own model.

These tickets are already RESOLVED with assignment_group filled in,
so they show up in export_servicenow_data.py as training examples.

Usage:
    python add_training_tickets.py              # adds ~60 resolved tickets
    python add_training_tickets.py --preview    # show tickets without creating
"""

import sys
import time
import argparse
import yaml
import requests
from requests.exceptions import RequestException

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

INSTANCE = config["servicenow"]["instance_url"].rstrip("/")
AUTH     = (config["servicenow"]["username"], config["servicenow"]["password"])
HEADERS  = {"Content-Type": "application/json", "Accept": "application/json"}

# ── 60 pre-resolved training tickets (20 per group) ──────────────────────────
TRAINING_TICKETS = [

    # ════════════════════════════════════════════════════════
    # GROUP: Network Support  (20 tickets)
    # ════════════════════════════════════════════════════════
    {
        "short_description": "VPN connection dropping after 10 minutes",
        "description": "Corporate VPN disconnects exactly 10 minutes after connecting. Cisco AnyConnect client shows timeout error. Reproducible across multiple users.",
        "category": "Network", "subcategory": "VPN",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "Updated VPN session timeout settings on ASA firewall from 600s to 3600s. Issue resolved.",
        "priority": "2", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "Cannot reach internal file share from remote office",
        "description": "Users at the Manchester office cannot access \\\\fileserver01\\shared. Ping to the server IP fails. VPN tunnel appears to be up.",
        "category": "Network", "subcategory": "Routing",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "Missing static route on edge router for 192.168.5.0/24 subnet. Added route and verified connectivity.",
        "priority": "2", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "Internet speed extremely slow on 3rd floor",
        "description": "All users on the 3rd floor reporting internet speeds below 1 Mbps. Other floors unaffected. Started after this morning's network maintenance.",
        "category": "Network", "subcategory": "Performance",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "Found duplex mismatch on switch uplink port. Set both sides to full-duplex 1Gbps. Speed restored.",
        "priority": "3", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "Network printer offline across entire department",
        "description": "HP LaserJet M507 (IP 192.168.1.45) showing as offline. Print jobs queuing up. Device physically powered on.",
        "category": "Network", "subcategory": "Printer",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "Switch port 24 on access switch SW-FL2-01 was in error-disabled state due to BPDU guard. Cleared and re-enabled.",
        "priority": "3", "impact": "3", "urgency": "3",
    },
    {
        "short_description": "SSL VPN certificate error on login page",
        "description": "Users getting browser certificate warning when accessing SSL VPN portal. Certificate shows as expired since yesterday.",
        "category": "Network", "subcategory": "VPN",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "Renewed SSL certificate on VPN concentrator. Updated certificate chain. Verified no browser warnings.",
        "priority": "2", "impact": "1", "urgency": "1",
    },
    {
        "short_description": "DNS lookup failing for external websites",
        "description": "Users cannot resolve external domain names. Internal hostnames work. Nslookup to 8.8.8.8 works but corporate DNS server not forwarding.",
        "category": "Network", "subcategory": "DNS",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "DNS forwarder configuration was lost after server reboot. Re-added forwarders to internal DNS servers.",
        "priority": "1", "impact": "1", "urgency": "1",
    },
    {
        "short_description": "Wireless access point offline in conference room B",
        "description": "WAP in conference room B not broadcasting SSID. PoE switch port shows connected but AP management page unreachable.",
        "category": "Network", "subcategory": "Wireless",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "AP firmware corrupted after power outage. Factory reset and re-provisioned from WLC. SSID broadcasting.",
        "priority": "3", "impact": "3", "urgency": "3",
    },
    {
        "short_description": "Firewall rule blocking new SaaS application",
        "description": "New HR SaaS tool cannot connect. Traffic to *.workday.com is being blocked by perimeter firewall. All 200 HR staff affected.",
        "category": "Network", "subcategory": "Firewall",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "Added application-specific firewall rules to allow HTTPS traffic to Workday IP ranges. Verified access.",
        "priority": "2", "impact": "1", "urgency": "2",
    },
    {
        "short_description": "Network latency spikes causing VoIP call drops",
        "description": "VoIP calls dropping every 5-10 minutes. Network monitoring shows latency spikes up to 800ms on core switch. Affects 50 phone extensions.",
        "category": "Network", "subcategory": "VoIP",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "QoS policy misconfigured after recent switch upgrade. Re-applied DSCP marking for voice traffic. Latency stable at <20ms.",
        "priority": "2", "impact": "2", "urgency": "1",
    },
    {
        "short_description": "Switch port not coming up after cable replacement",
        "description": "Server team replaced patch cable on SRV-APP-03 but port still shows down. Server NIC showing link but switch port in err-disabled.",
        "category": "Network", "subcategory": "Hardware",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "Port was in err-disabled state due to port security MAC violation. Cleared violation and updated MAC whitelist.",
        "priority": "2", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "Routing loop causing network instability",
        "description": "Core network experiencing packet loss and high CPU on routers. Traceroute showing packets looping between two routers.",
        "category": "Network", "subcategory": "Routing",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "Duplicate OSPF area 0 configuration found after last week's change. Removed duplicate and redistributed routes.",
        "priority": "1", "impact": "1", "urgency": "1",
    },
    {
        "short_description": "Proxy server blocking legitimate business websites",
        "description": "Web proxy blocking access to several approved vendor sites. Category filter incorrectly classifying them as malicious.",
        "category": "Network", "subcategory": "Proxy",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "Added URL exceptions for 5 vendor domains in proxy policy. Updated threat intelligence feed categorisation.",
        "priority": "3", "impact": "2", "urgency": "3",
    },
    {
        "short_description": "DHCP pool exhausted on guest wifi network",
        "description": "Guest WiFi users unable to get IP addresses. DHCP server logs show pool 192.168.100.0/24 fully allocated.",
        "category": "Network", "subcategory": "DHCP",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "Reduced DHCP lease time from 24 hours to 2 hours. Expanded guest pool to /23. Also found rogue DHCP device — removed.",
        "priority": "3", "impact": "3", "urgency": "3",
    },
    {
        "short_description": "WAN link to DR site flapping",
        "description": "MPLS WAN link to disaster recovery site going up and down every few minutes. BGP session not stable. DR replication affected.",
        "category": "Network", "subcategory": "WAN",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "Coordinated with ISP — faulty SFP on their CPE device. Replaced by ISP engineer. Link stable for 24 hours.",
        "priority": "1", "impact": "1", "urgency": "1",
    },
    {
        "short_description": "Network monitoring alerts not sending emails",
        "description": "Nagios monitoring stopped sending alert emails. Network team not receiving any notifications for the last 6 hours.",
        "category": "Network", "subcategory": "Monitoring",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "SMTP relay configuration changed after mail server migration. Updated Nagios SMTP settings with new relay server.",
        "priority": "3", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "IPSec tunnel to partner company dropped",
        "description": "Site-to-site IPSec VPN tunnel to partner organisation has been down for 2 hours. EDI file transfers failing.",
        "category": "Network", "subcategory": "VPN",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "Partner changed their public IP without notifying us. Updated IKE peer address on our firewall. Tunnel re-established.",
        "priority": "2", "impact": "2", "urgency": "1",
    },
    {
        "short_description": "High packet loss on internet connection",
        "description": "30% packet loss observed on primary internet uplink. Affecting all cloud services and internet browsing.",
        "category": "Network", "subcategory": "Performance",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "ISP confirmed line fault 2km from our premises. Repair completed by ISP. Failover to secondary link maintained service.",
        "priority": "1", "impact": "1", "urgency": "1",
    },
    {
        "short_description": "Port security violation on trading floor switch",
        "description": "Multiple ports on trading floor showing security violations. Users cannot connect. MAC address table full.",
        "category": "Network", "subcategory": "Security",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "Network scan found rogue hub connected to port 12. Removed and increased sticky MAC limit. Reviewed all port security policies.",
        "priority": "2", "impact": "2", "urgency": "1",
    },
    {
        "short_description": "Load balancer not distributing traffic evenly",
        "description": "F5 load balancer sending 90% of traffic to one web server node. Other nodes idle. Application response times degraded.",
        "category": "Network", "subcategory": "Load Balancer",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "Round-robin algorithm accidentally changed to least-connections after F5 upgrade. Reverted algorithm. Traffic balanced.",
        "priority": "2", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "Network time sync failing across all servers",
        "description": "All servers drifting from correct time. Kerberos authentication failures starting to occur. NTP server unreachable.",
        "category": "Network", "subcategory": "NTP",
        "assignment_group": "Network Support", "state": "6",
        "close_notes": "Primary NTP server had disk failure and was offline. Promoted secondary NTP server. Synchronised all servers.",
        "priority": "2", "impact": "1", "urgency": "2",
    },

    # ════════════════════════════════════════════════════════
    # GROUP: Application Support  (20 tickets)
    # ════════════════════════════════════════════════════════
    {
        "short_description": "ERP system throwing null pointer exception on order entry",
        "description": "Users get NullPointerException when saving new customer orders in the ERP system. Error in Order.java line 342. Affects all order entry staff.",
        "category": "Application", "subcategory": "Error",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Bug in v3.4.1 release — product ID field not validated before save. Applied hotfix patch 3.4.1-HF2. Tested and deployed.",
        "priority": "1", "impact": "1", "urgency": "1",
    },
    {
        "short_description": "User permissions missing after Active Directory migration",
        "description": "50 users lost access to multiple applications after AD migration last weekend. Roles not mapped correctly to new OU structure.",
        "category": "Software", "subcategory": "Permissions",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Created new group policy mapping for migrated OU. Re-provisioned application roles for all affected users.",
        "priority": "2", "impact": "1", "urgency": "2",
    },
    {
        "short_description": "Report generation timing out in BI platform",
        "description": "Large financial reports in Tableau timing out after 30 seconds. Reports that used to take 5 seconds now fail. Started after database upgrade.",
        "category": "Application", "subcategory": "Performance",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Missing index on fact_sales table after DB migration. Added composite index on date_key and region_id. Reports now run in 3s.",
        "priority": "2", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "Email notifications not sending from CRM",
        "description": "Salesforce CRM stopped sending automated email notifications to customers 2 days ago. Outbound email queue has 2000+ stuck messages.",
        "category": "Application", "subcategory": "Email",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Sendgrid API key expired. Renewed API key and updated in Salesforce Connected App settings. Queue processed.",
        "priority": "2", "impact": "1", "urgency": "2",
    },
    {
        "short_description": "Login page blank after browser update",
        "description": "Application login page appears completely blank on Chrome 120. Console shows JS error: Cannot read property of undefined. IE11 works.",
        "category": "Software", "subcategory": "Browser",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "App uses deprecated window.event API removed in Chrome 120. Updated JavaScript to use event parameter instead. Tested on all browsers.",
        "priority": "2", "impact": "2", "urgency": "1",
    },
    {
        "short_description": "Data import failing with encoding error",
        "description": "CSV import tool throwing UnicodeDecodeError on files with special characters. Affects all European locale files with accented characters.",
        "category": "Application", "subcategory": "Data",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Import function hardcoded to ASCII. Updated to UTF-8 with BOM handling. Tested with German, French, Spanish sample files.",
        "priority": "3", "impact": "2", "urgency": "3",
    },
    {
        "short_description": "Two-factor authentication not working for mobile users",
        "description": "TOTP codes from Google Authenticator rejected for 15% of users. Time-based codes expiring too quickly.",
        "category": "Software", "subcategory": "Authentication",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Auth server clock drifted by 45 seconds. Re-synced with NTP. Extended TOTP window tolerance to 90 seconds as interim fix.",
        "priority": "2", "impact": "2", "urgency": "1",
    },
    {
        "short_description": "Application log files filling up disk",
        "description": "Application server disk at 100%. Root cause: verbose DEBUG logging left enabled after troubleshooting session 3 weeks ago.",
        "category": "Application", "subcategory": "Infrastructure",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Changed log level from DEBUG to WARN. Cleared 45GB of log files. Implemented log rotation policy with 7-day retention.",
        "priority": "2", "impact": "2", "urgency": "1",
    },
    {
        "short_description": "PDF export feature generating corrupted files",
        "description": "Document management system PDF export producing files that cannot be opened. Error: 'File is damaged and could not be repaired'.",
        "category": "Application", "subcategory": "Feature",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "iText library version incompatibility after Java upgrade to 17. Upgraded iText from 5.5 to 8.0. PDF export working correctly.",
        "priority": "2", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "Session timeout too aggressive on trading application",
        "description": "Trading application logging users out after 5 minutes of inactivity. Traders losing work. Business requires 30-minute timeout.",
        "category": "Application", "subcategory": "Configuration",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Session timeout misconfigured in application.properties after security review. Updated session.timeout from 300 to 1800 seconds.",
        "priority": "3", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "Search functionality returning no results",
        "description": "Full-text search in knowledge base returning empty results for all queries since last night's maintenance. Index appears corrupt.",
        "category": "Application", "subcategory": "Search",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Elasticsearch index was deleted during maintenance by mistake. Rebuilt index from database. Search fully functional.",
        "priority": "2", "impact": "2", "urgency": "1",
    },
    {
        "short_description": "Mobile app crashing on iOS 17 devices",
        "description": "Company mobile app crashing immediately on launch for all iPhone users who updated to iOS 17. Android unaffected.",
        "category": "Application", "subcategory": "Mobile",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "iOS 17 deprecated UIWebView API. Migrated to WKWebView. Released v4.2.1 to App Store. Users prompted to update.",
        "priority": "1", "impact": "2", "urgency": "1",
    },
    {
        "short_description": "Database connection pool exhausted during peak hours",
        "description": "Application throwing 'Unable to acquire JDBC Connection' errors every day at 9-10am. Max connections reached.",
        "category": "Application", "subcategory": "Database",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Connection pool max-size set to 10 — too low. Increased to 50. Also found 3 connection leaks in transaction code — fixed.",
        "priority": "1", "impact": "1", "urgency": "1",
    },
    {
        "short_description": "Wrong VAT rate applied in billing system",
        "description": "Billing system applying 20% VAT to zero-rated items. Compliance issue discovered during audit. Affects 300 invoices.",
        "category": "Application", "subcategory": "Finance",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Tax rule configuration table had incorrect mapping after last upgrade. Updated tax codes and ran correction script for affected invoices.",
        "priority": "1", "impact": "1", "urgency": "1",
    },
    {
        "short_description": "API rate limiting blocking integration partner",
        "description": "Partner integration failing with HTTP 429 Too Many Requests. Our API gateway throttling partner's legitimate calls.",
        "category": "Application", "subcategory": "API",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Partner's API key had default rate limit of 100 req/min. Increased to 1000 req/min based on SLA. Updated documentation.",
        "priority": "2", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "User unable to reset password — link expires instantly",
        "description": "Password reset email arrives but clicking the link shows 'Link expired'. Token valid for 15 minutes but users say link expires immediately.",
        "category": "Software", "subcategory": "Authentication",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Token expiry was being calculated in UTC but compared against local time. Fixed timezone handling in auth service.",
        "priority": "2", "impact": "3", "urgency": "2",
    },
    {
        "short_description": "Scheduled batch job not running overnight",
        "description": "Nightly data reconciliation batch job has not run for 3 nights. Cron job shows as scheduled but no execution logs.",
        "category": "Application", "subcategory": "Batch",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Cron job user account password expired. Updated service account to non-expiring password. Batch job executed successfully.",
        "priority": "2", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "Application displaying incorrect currency conversion",
        "description": "E-commerce platform showing EUR prices calculated incorrectly. Exchange rate last updated 6 months ago. Revenue impact.",
        "category": "Application", "subcategory": "Finance",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Currency feed API key expired. Renewed subscription and implemented daily automated rate refresh with alerting.",
        "priority": "1", "impact": "1", "urgency": "1",
    },
    {
        "short_description": "Drag and drop file upload not working in Firefox",
        "description": "Document portal drag-and-drop upload broken in Firefox 118. Works in Chrome. JS console: DragEvent is not defined.",
        "category": "Application", "subcategory": "Browser",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Firefox 118 changed DragEvent handling. Applied polyfill and updated event listener to use both native and polyfill API.",
        "priority": "3", "impact": "3", "urgency": "3",
    },
    {
        "short_description": "Application health check endpoint returning 503",
        "description": "Load balancer health check failing — /health endpoint returning HTTP 503. Application appears functional but is being removed from rotation.",
        "category": "Application", "subcategory": "Infrastructure",
        "assignment_group": "Application Support", "state": "6",
        "close_notes": "Health check endpoint was querying database. Database maintenance window caused temporary 503. Added DB circuit breaker to health check.",
        "priority": "1", "impact": "1", "urgency": "1",
    },

    # ════════════════════════════════════════════════════════
    # GROUP: Cloud Operations  (20 tickets)
    # ════════════════════════════════════════════════════════
    {
        "short_description": "Production Kubernetes cluster nodes not scheduling pods",
        "description": "New pods stuck in Pending state. kubectl describe shows: 0/6 nodes available: 6 Insufficient memory. Cluster was upgraded last night.",
        "category": "Cloud", "subcategory": "Kubernetes",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Resource limits increased without corresponding node capacity. Added 2 worker nodes to cluster. All pending pods scheduled.",
        "priority": "1", "impact": "1", "urgency": "1",
    },
    {
        "short_description": "Azure App Service scaling not triggering under load",
        "description": "App Service not auto-scaling despite CPU at 100%. Scale-out rule configured to trigger at 70% CPU but not activating.",
        "category": "Cloud", "subcategory": "Azure",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Scale-out cooldown period was set to 60 minutes. Reduced to 5 minutes. Also found scale rule pointing to wrong metric namespace.",
        "priority": "2", "impact": "2", "urgency": "1",
    },
    {
        "short_description": "S3 bucket public access accidentally enabled",
        "description": "Security scan flagged S3 bucket prod-documents as publicly accessible. Contains sensitive customer documents. Immediate remediation required.",
        "category": "Cloud", "subcategory": "AWS",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Removed public ACL and enabled Block Public Access settings. Implemented S3 bucket policy restricting access to VPC endpoint only.",
        "priority": "1", "impact": "1", "urgency": "1",
    },
    {
        "short_description": "Docker image registry disk space full",
        "description": "Harbor container registry disk at 100%. New image pushes failing. CI/CD pipelines blocked across all teams.",
        "category": "Cloud", "subcategory": "Docker",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Cleaned up 200+ untagged images and images older than 90 days. Implemented retention policy — keep last 10 tags per repo.",
        "priority": "2", "impact": "1", "urgency": "1",
    },
    {
        "short_description": "Terraform state file locked and blocking deployments",
        "description": "Terraform state in S3 showing as locked for 6 hours. Previous pipeline run failed mid-apply and did not release lock.",
        "category": "Cloud", "subcategory": "Infrastructure",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Manually removed stale lock from DynamoDB state lock table. Verified failed deployment rolled back correctly. Re-ran pipeline.",
        "priority": "2", "impact": "1", "urgency": "2",
    },
    {
        "short_description": "Cloud VM snapshot backup jobs failing overnight",
        "description": "Azure Backup reporting 15 VM snapshot failures. Policy configured but jobs timing out. No backups completed in 5 days.",
        "category": "Cloud", "subcategory": "Backup",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Backup vault storage limit reached. Expanded vault capacity and deleted expired recovery points. Backups completing successfully.",
        "priority": "2", "impact": "1", "urgency": "2",
    },
    {
        "short_description": "Helm chart deployment failing with ImagePullBackOff",
        "description": "New microservice deployment failing across all environments. Kubernetes pods stuck in ImagePullBackOff. Image exists in registry.",
        "category": "Cloud", "subcategory": "Kubernetes",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Registry authentication secret expired in namespaces. Rotated credentials and updated imagePullSecrets in all namespaces.",
        "priority": "2", "impact": "2", "urgency": "1",
    },
    {
        "short_description": "Cloud cost spike — $15,000 over budget this month",
        "description": "Azure cost alert triggered. Monthly spend on track to exceed budget by $15,000. Cost analysis shows EC2 dev instances not shut down over weekend.",
        "category": "Cloud", "subcategory": "Cost",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Identified 45 dev/test VMs running 24/7. Implemented auto-shutdown schedule for non-prod at 7pm weekdays and all weekend.",
        "priority": "2", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "CDN caching stale content after website deployment",
        "description": "Customers seeing old version of website 4 hours after deployment. CloudFront serving cached content. Deployment includes critical bug fix.",
        "category": "Cloud", "subcategory": "CDN",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Created CloudFront invalidation for /* path. Future deployments now include automated invalidation step in CI/CD pipeline.",
        "priority": "2", "impact": "2", "urgency": "1",
    },
    {
        "short_description": "Database RDS instance storage auto-expansion not working",
        "description": "RDS PostgreSQL instance disk at 95% and still growing. Storage auto-scaling enabled but not triggering. Risk of database going read-only.",
        "category": "Cloud", "subcategory": "Database",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Auto-scaling maximum was set equal to current size. Increased max storage from 100GB to 500GB. Also archived 40GB of old audit logs.",
        "priority": "1", "impact": "1", "urgency": "1",
    },
    {
        "short_description": "Azure DevOps pipeline timing out on build step",
        "description": "Build pipeline timing out after 60 minutes on unit test step. Tests were passing in 15 minutes previously. Started after dependency update.",
        "category": "Cloud", "subcategory": "CI/CD",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "New test library introduced infinite loop in edge case. Identified and fixed loop. Added test timeout of 30 minutes as safeguard.",
        "priority": "2", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "GCP Cloud Run service returning 429 quota exceeded",
        "description": "Cloud Run service hitting GCP project quotas. All requests returning 429 during business hours. Auto-scaling not effective.",
        "category": "Cloud", "subcategory": "GCP",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Requested quota increase for Cloud Run concurrent requests from 1000 to 5000. Approved and applied. Also optimised cold start time.",
        "priority": "1", "impact": "1", "urgency": "1",
    },
    {
        "short_description": "Kubernetes ingress not routing traffic to new service",
        "description": "New API service deployed but external traffic not reaching pods. Ingress controller configured, service and pods running, but requests return 404.",
        "category": "Cloud", "subcategory": "Kubernetes",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Ingress path annotation missing trailing path prefix. Added nginx.ingress.kubernetes.io/rewrite-target annotation. Traffic routing correctly.",
        "priority": "2", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "Lambda cold start latency causing API gateway timeouts",
        "description": "API Gateway timing out 30% of requests during off-peak hours due to Lambda cold starts. SLA requires <500ms response time.",
        "category": "Cloud", "subcategory": "Serverless",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Enabled Lambda Provisioned Concurrency for production functions. Configured 5 warm instances. P99 latency reduced from 4.2s to 180ms.",
        "priority": "2", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "SSL certificate expiry causing service outage",
        "description": "Production HTTPS certificate expired at midnight. All API calls returning SSL handshake error. Certificate auto-renewal failed silently.",
        "category": "Cloud", "subcategory": "Security",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Let's Encrypt renewal failing due to expired IAM role permissions. Renewed manually immediately. Updated IAM policy and tested auto-renewal.",
        "priority": "1", "impact": "1", "urgency": "1",
    },
    {
        "short_description": "Cloud storage bucket replication lagging by 48 hours",
        "description": "Azure Storage geo-replication showing 48-hour lag. RPO breached. Replication was working fine until 3 days ago.",
        "category": "Cloud", "subcategory": "Storage",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Large batch of 10TB data uploaded without triggering replication correctly. Root cause: replication rule excluded files over 5GB. Updated rule.",
        "priority": "2", "impact": "1", "urgency": "2",
    },
    {
        "short_description": "Monitoring dashboards showing no metrics",
        "description": "Grafana dashboards blank for all cloud resources. No metrics visible for last 6 hours. CloudWatch exporter pod crashing.",
        "category": "Cloud", "subcategory": "Monitoring",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "CloudWatch exporter IAM role had expired temporary credentials. Rotated to permanent IAM role. Metrics resumed. Historical gap noted.",
        "priority": "2", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "Elasticsearch cluster red status — unassigned shards",
        "description": "Production Elasticsearch cluster showing RED status. 45 unassigned shards. Cluster has been degraded for 12 hours.",
        "category": "Cloud", "subcategory": "Database",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "One data node had disk failure. Replaced node with new EC2 instance. Shards rebalanced automatically over 3 hours. Cluster GREEN.",
        "priority": "1", "impact": "1", "urgency": "1",
    },
    {
        "short_description": "VPC peering not routing traffic between accounts",
        "description": "New VPC peering connection between prod and analytics AWS accounts created but traffic not flowing. Confirmed peering accepted.",
        "category": "Cloud", "subcategory": "AWS",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Route tables in both VPCs not updated to include peering routes. Added CIDR routes pointing to peering connection in both accounts.",
        "priority": "2", "impact": "2", "urgency": "2",
    },
    {
        "short_description": "Container resource limits causing OOM kills",
        "description": "Application pods being OOMKilled every few hours. Container memory limit set too low for current workload.",
        "category": "Cloud", "subcategory": "Kubernetes",
        "assignment_group": "Cloud Operations", "state": "6",
        "close_notes": "Memory limit was 512Mi for a Java app with 400Mi heap. Increased limits to 1.5Gi and adjusted JVM heap to 1Gi. OOM kills stopped.",
        "priority": "2", "impact": "2", "urgency": "2",
    },
]


def test_connection():
    try:
        r = requests.get(
            f"{INSTANCE}/api/now/table/incident",
            auth=AUTH, headers=HEADERS,
            params={"sysparm_limit": 1}, timeout=15
        )
        return r.status_code == 200
    except RequestException:
        return False


def get_group_sys_id(group_name: str) -> str | None:
    """Look up the sys_id of an assignment group by name."""
    r = requests.get(
        f"{INSTANCE}/api/now/table/sys_user_group",
        auth=AUTH, headers=HEADERS,
        params={"sysparm_query": f"name={group_name}", "sysparm_fields": "sys_id,name"},
        timeout=10
    )
    results = r.json().get("result", [])
    return results[0]["sys_id"] if results else None


def create_ticket(ticket: dict, index: int, total: int, group_cache: dict) -> bool:
    group_name = ticket["assignment_group"]

    # Resolve group sys_id (cached)
    if group_name not in group_cache:
        group_cache[group_name] = get_group_sys_id(group_name)

    group_sys_id = group_cache.get(group_name)

    payload = {
        "short_description": ticket["short_description"],
        "description":       ticket["description"],
        "category":          ticket["category"],
        "subcategory":       ticket["subcategory"],
        "priority":          ticket.get("priority", "3"),
        "impact":            ticket.get("impact", "3"),
        "urgency":           ticket.get("urgency", "3"),
        "state":             ticket.get("state", "6"),   # 6 = Resolved
        "close_code":        "Solved (Permanently)",
        "close_notes":       ticket.get("close_notes", "Resolved by support team."),
    }

    if group_sys_id:
        payload["assignment_group"] = group_sys_id
    else:
        payload["assignment_group.name"] = group_name

    try:
        r = requests.post(
            f"{INSTANCE}/api/now/table/incident",
            auth=AUTH, headers=HEADERS,
            json=payload, timeout=15
        )
        r.raise_for_status()
        number = r.json()["result"].get("number", "N/A")
        print(f"  [{index:02d}/{total}] {GREEN}✅{RESET}  {number}  [{group_name}]  {ticket['short_description'][:50]}")
        return True
    except Exception as e:
        print(f"  [{index:02d}/{total}] {RED}❌{RESET}  FAILED  {ticket['short_description'][:50]}  ({e})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Add resolved training tickets to ServiceNow")
    parser.add_argument("--preview", action="store_true", help="Show tickets without creating them")
    args = parser.parse_args()

    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗
║   Add Training Data Tickets to ServiceNow                ║
║   Creates {len(TRAINING_TICKETS)} resolved incidents for model training       ║
╚══════════════════════════════════════════════════════════╝{RESET}
  Instance : {BOLD}{INSTANCE}{RESET}
""")

    if args.preview:
        from collections import Counter
        groups = Counter(t["assignment_group"] for t in TRAINING_TICKETS)
        print(f"  {BOLD}Tickets that will be created:{RESET}\n")
        for grp, cnt in groups.items():
            print(f"  {CYAN}{grp:<35}{RESET}  {cnt} tickets")
        print(f"\n  Total: {BOLD}{len(TRAINING_TICKETS)}{RESET} tickets (all in Resolved/Closed state)")
        print(f"\n  Run without --preview to actually create them.\n")
        return

    print(f"  {BOLD}Testing connection...{RESET}", end=" ")
    if not test_connection():
        print(f"{RED}FAILED{RESET}")
        print(f"\n  {RED}Cannot connect to {INSTANCE}{RESET}")
        print(f"  Check config/config.yaml credentials.\n")
        sys.exit(1)
    print(f"{GREEN}OK{RESET}")

    print(f"\n  {BOLD}Creating {len(TRAINING_TICKETS)} resolved training tickets...{RESET}\n")

    group_cache = {}
    success = 0
    failed  = 0

    for i, ticket in enumerate(TRAINING_TICKETS, 1):
        ok = create_ticket(ticket, i, len(TRAINING_TICKETS), group_cache)
        if ok:
            success += 1
        else:
            failed += 1
        time.sleep(0.25)

    # Group breakdown
    from collections import Counter
    groups = Counter(t["assignment_group"] for t in TRAINING_TICKETS)

    print(f"""
  {'═'*56}
  {GREEN}{BOLD}Done!{RESET}
  Created  : {GREEN}{BOLD}{success}{RESET} tickets
  Failed   : {RED}{failed}{RESET} tickets

  {BOLD}Tickets per group:{RESET}""")
    for grp, cnt in groups.items():
        print(f"    {grp:<35}  {cnt} tickets")

    print(f"""
  {BOLD}These are RESOLVED tickets — the AI can learn from them.{RESET}

  {BOLD}Next steps:{RESET}
    1. Export your data:
       {CYAN}python export_servicenow_data.py{RESET}

    2. Train your own model:
       {CYAN}python train_own_model.py --csv data/my_servicenow_tickets.csv{RESET}

    3. Run the agent:
       {CYAN}python run_agent.py{RESET}
  {'═'*56}
""")


if __name__ == "__main__":
    main()
