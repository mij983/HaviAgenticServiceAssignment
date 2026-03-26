TEAM: IT-Network Support
# VPN Troubleshooting Guide

This guide covers common VPN issues reported by end users and the steps to resolve them.

---

## Issue: VPN Not Connecting

**Symptoms:**
- User cannot establish VPN connection from home or remote office
- Error message: "Unable to connect to VPN server"
- VPN client shows "Timeout" or "Authentication failed"

**Resolution Steps:**
1. Ask the user to restart the VPN client (GlobalProtect / Cisco AnyConnect)
2. Verify the user's internet connection is working (can they browse the web?)
3. Check if the VPN gateway is reachable: ping vpn.company.com
4. Ask the user to re-enter their credentials — password may have expired
5. If MFA is enabled, confirm the user is approving the push notification
6. If the issue persists, escalate to IT-Network Support with the VPN client logs

**Common Causes:**
- Expired Active Directory password
- MFA token not approved
- ISP blocking UDP port 4501 (used by GlobalProtect)
- Corporate firewall blocking the home IP

---

## Issue: VPN Drops Frequently

**Symptoms:**
- VPN connects but disconnects after a few minutes
- User reports intermittent connectivity during video calls

**Resolution Steps:**
1. Check the user's home internet stability (run a speed test)
2. Ask the user to switch from Wi-Fi to a wired ethernet connection
3. Update the VPN client to the latest version
4. Change VPN protocol from IPSec to SSL (or vice versa) in client settings
5. If using a home router — ask the user to disable VPN passthrough setting

**Escalation:**
- If the issue affects multiple users in the same region, escalate to IT-Network Support immediately as it may indicate a gateway issue.

---

## Issue: SSL VPN Certificate Error

**Symptoms:**
- Browser or VPN client shows "Certificate not trusted" warning
- Error: "SSL handshake failed"

**Resolution Steps:**
1. Check if the corporate root CA certificate is installed on the user's machine
2. Verify the system clock is correct — certificate validation fails if clock is wrong
3. Push the latest certificate via Intune (raise request to IT-Intune Support if needed)

---

## Related Teams

| Issue Type                    | Route To                          |
|-------------------------------|-----------------------------------|
| VPN client not connecting     | IT-Network Support                |
| MFA / Authentication issues   | IT-Access Management              |
| Certificate / PKI issues      | IT-Security EDR                   |
| Intune device compliance      | IT-Intune Support                 |
| Network infrastructure faults | IT-Network Security               |
