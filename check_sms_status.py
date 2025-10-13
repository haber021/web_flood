#!/usr/bin/env python
"""
Command-line script to check SMS sending status for flood alerts.
Shows which recipients received messages and which ones failed.
"""
import os
import django
import sys
from datetime import timedelta

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flood_monitoring.settings')
sys.path.append(os.path.dirname(__file__))
django.setup()

from django.utils import timezone
from core.models import FloodAlert, NotificationLog

def check_sms_status(hours=24):
    """Check SMS status for alerts in the last N hours"""
    print(f"Checking SMS status for alerts in the last {hours} hours...")
    print("=" * 60)

    # Get alerts from the last N hours
    since = timezone.now() - timedelta(hours=hours)
    alerts = FloodAlert.objects.filter(issued_at__gte=since).order_by('-issued_at')

    if not alerts.exists():
        print(f"No alerts found in the last {hours} hours.")
        return

    total_alerts = alerts.count()
    total_sms_sent = 0
    total_sms_failed = 0
    total_sms_pending = 0

    for alert in alerts:
        print(f"\nAlert: {alert.title}")
        print(f"Severity: {alert.get_severity_level_display()}")
        print(f"Issued: {alert.issued_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Active: {alert.active}")
        print("-" * 40)

        # Get notification logs for this alert
        sms_logs = NotificationLog.objects.filter(
            alert=alert,
            notification_type='sms'
        ).order_by('sent_at')

        if not sms_logs.exists():
            print("No SMS notifications sent for this alert.")
            continue

        alert_sms_sent = 0
        alert_sms_failed = 0
        alert_sms_pending = 0

        for log in sms_logs:
            status_icon = {
                'sent': '[SENT]',
                'failed': '[FAILED]',
                'pending': '[PENDING]'
            }.get(log.status, '[UNKNOWN]')

            print(f"{status_icon} {log.recipient} - {log.status.upper()}")

            if log.status == 'sent':
                alert_sms_sent += 1
            elif log.status == 'failed':
                alert_sms_failed += 1
            elif log.status == 'pending':
                alert_sms_pending += 1

        print(f"\nSMS Summary for this alert: {alert_sms_sent} sent, {alert_sms_failed} failed, {alert_sms_pending} pending")

        total_sms_sent += alert_sms_sent
        total_sms_failed += alert_sms_failed
        total_sms_pending += alert_sms_pending

    print("\n" + "=" * 60)
    print(f"OVERALL SUMMARY (Last {hours} hours):")
    print(f"Total Alerts: {total_alerts}")
    print(f"SMS Sent: {total_sms_sent}")
    print(f"SMS Failed: {total_sms_failed}")
    print(f"SMS Pending: {total_sms_pending}")

    # Check SMS configuration
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
    sms_enabled = os.getenv('SMS_ENABLED', 'true').lower() == 'true'
    print(f"\nSMS Configuration: {'ENABLED' if sms_enabled else 'DISABLED'}")

    if not sms_enabled and total_sms_pending > 0:
        print("TIP: Set SMS_ENABLED=true in .env to send pending SMS messages")

if __name__ == "__main__":
    # Allow custom hours via command line argument
    hours = 24  # default
    if len(sys.argv) > 1:
        try:
            hours = int(sys.argv[1])
        except ValueError:
            print("Usage: python check_sms_status.py [hours]")
            print("Example: python check_sms_status.py 48")
            sys.exit(1)

    check_sms_status(hours)