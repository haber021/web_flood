from django.core.management.base import BaseCommand
from django.utils import timezone
from core.models import FloodAlert
from core.notifications import dispatch_notifications_for_alert


class Command(BaseCommand):
    help = 'Send scheduled alerts that are due'

    def handle(self, *args, **options):
        current_time = timezone.now()
        self.stdout.write(f'Checking for scheduled alerts at {current_time}')

        # Find alerts that are scheduled to be sent now or in the past
        scheduled_alerts = FloodAlert.objects.filter(
            scheduled_send_time__lte=current_time,
            active=True
        ).exclude(
            # Exclude alerts that have already been sent
            last_notification_sent_at__isnull=False
        )

        if not scheduled_alerts:
            self.stdout.write('No scheduled alerts to send.')
            return

        sent_count = 0
        for alert in scheduled_alerts:
            try:
                self.stdout.write(f'Sending scheduled alert: {alert.title}')
                dispatch_notifications_for_alert(alert)
                alert.last_notification_sent_at = current_time
                alert.save(update_fields=['last_notification_sent_at'])
                sent_count += 1
                self.stdout.write(f'  ✓ Sent notifications for alert: {alert.title}')
            except Exception as e:
                self.stdout.write(f'  ✗ Failed to send alert {alert.title}: {str(e)}')

        self.stdout.write(f'Successfully sent {sent_count} scheduled alerts.')
