from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from core.models import Barangay, FloodAlert, SensorData, ThresholdSetting
from core.notifications import dispatch_notifications_for_alert
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Send automatic alerts for barangays every 3 hours based on sensor data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be done without actually sending alerts',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        current_time = timezone.now()

        self.stdout.write(f'Checking for automatic barangay alerts at {current_time}')

        # Get all barangays with auto alerts enabled
        barangays = Barangay.objects.filter(auto_alert_enabled=True)

        if not barangays:
            self.stdout.write('No barangays found.')
            return

        # Get threshold settings
        thresholds = {t.parameter: t for t in ThresholdSetting.objects.all()}

        if not thresholds:
            self.stdout.write('No threshold settings found. Please configure thresholds first.')
            return

        alerts_created = 0
        alerts_sent = 0

        for barangay in barangays:
            try:
                # Check if enough time has passed since last alert (3 hours minimum)
                if barangay.last_auto_alert_sent:
                    time_since_last_alert = current_time - barangay.last_auto_alert_sent
                    if time_since_last_alert < timedelta(hours=3):
                        continue  # Skip this barangay, not enough time has passed

                # Check if we should send an alert for this barangay
                alert_needed, severity_level, alert_title, alert_description = self.should_send_alert(barangay, thresholds)

                if alert_needed:
                    if dry_run:
                        self.stdout.write(f'[DRY RUN] Would create alert for {barangay.name}: {alert_title}')
                        continue

                    # Create the alert
                    alert = FloodAlert.objects.create(
                        title=alert_title,
                        description=alert_description,
                        severity_level=severity_level,
                        active=True,
                        issued_by=None,  # System-generated
                        scheduled_send_time=current_time  # Send immediately
                    )

                    # Add the barangay to affected barangays
                    alert.affected_barangays.add(barangay)

                    self.stdout.write(f'Created alert for {barangay.name}: {alert.title}')

                    # Send notifications immediately
                    try:
                        dispatch_notifications_for_alert(alert)
                        alert.last_notification_sent_at = current_time
                        alert.save(update_fields=['last_notification_sent_at'])

                        # Update barangay's last auto alert timestamp
                        barangay.last_auto_alert_sent = current_time
                        barangay.save(update_fields=['last_auto_alert_sent'])

                        alerts_sent += 1
                        self.stdout.write(f'  ✓ Sent notifications for {barangay.name}')
                    except Exception as e:
                        self.stdout.write(f'  ✗ Failed to send notifications for {barangay.name}: {str(e)}')

                    alerts_created += 1

            except Exception as e:
                self.stdout.write(f'Error processing {barangay.name}: {str(e)}')
                logger.error(f'Error processing automatic alert for barangay {barangay.id}: {str(e)}')

        if dry_run:
            self.stdout.write(f'Dry run completed. Would create {alerts_created} alerts.')
        else:
            self.stdout.write(f'Created {alerts_created} automatic alerts, sent {alerts_sent} notifications.')

    def should_send_alert(self, barangay, thresholds):
        """
        Determine if an alert should be sent for this barangay based on recent sensor data.
        Returns: (should_send, severity_level, title, description)
        """
        current_time = timezone.now()
        three_hours_ago = current_time - timedelta(hours=3)

        # Get the latest sensor readings for this barangay in the last 3 hours
        sensor_data = SensorData.objects.filter(
            sensor__barangay=barangay,
            timestamp__gte=three_hours_ago,
            timestamp__lte=current_time
        ).select_related('sensor')

        if not sensor_data:
            return False, 0, "", ""

        # Analyze sensor data to determine if alert is needed
        max_severity = 0
        alert_messages = []

        for data in sensor_data:
            sensor_type = data.sensor.sensor_type
            value = data.value

            if sensor_type in thresholds:
                threshold = thresholds[sensor_type]
                severity = self.get_severity_level(value, threshold)

                if severity > max_severity:
                    max_severity = severity

                if severity >= 3:  # Warning or higher
                    alert_messages.append(
                        f"{sensor_type.title()}: {value}{threshold.unit} (threshold: {self.get_threshold_value(severity, threshold)}{threshold.unit})"
                    )

        if max_severity >= 2:  # At least Watch level
            title = self.get_alert_title(max_severity, barangay.name)
            description = self.get_alert_description(max_severity, barangay.name, alert_messages)
            return True, max_severity, title, description

        return False, 0, "", ""

    def get_severity_level(self, value, threshold):
        """Determine severity level based on value and thresholds"""
        if value >= threshold.catastrophic_threshold:
            return 5
        elif value >= threshold.emergency_threshold:
            return 4
        elif value >= threshold.warning_threshold:
            return 3
        elif value >= threshold.watch_threshold:
            return 2
        elif value >= threshold.advisory_threshold:
            return 1
        return 0

    def get_threshold_value(self, severity_level, threshold):
        """Get the threshold value for a given severity level"""
        if severity_level == 5:
            return threshold.catastrophic_threshold
        elif severity_level == 4:
            return threshold.emergency_threshold
        elif severity_level == 3:
            return threshold.warning_threshold
        elif severity_level == 2:
            return threshold.watch_threshold
        elif severity_level == 1:
            return threshold.advisory_threshold
        return 0

    def get_alert_title(self, severity_level, barangay_name):
        """Generate alert title based on severity"""
        severity_names = {
            1: "ADVISORY",
            2: "WATCH",
            3: "WARNING",
            4: "EMERGENCY",
            5: "CATASTROPHIC ALERT"
        }
        severity_name = severity_names.get(severity_level, "ALERT")
        return f"AUTOMATIC {severity_name}: {barangay_name}"

    def get_alert_description(self, severity_level, barangay_name, alert_messages):
        """Generate alert description"""
        base_messages = {
            1: f"Environmental conditions in {barangay_name} are approaching advisory levels.",
            2: f"Environmental conditions in {barangay_name} require monitoring.",
            3: f"Warning: Environmental conditions in {barangay_name} may pose risks.",
            4: f"Emergency: Critical environmental conditions detected in {barangay_name}.",
            5: f"Catastrophic: Extreme environmental conditions in {barangay_name}. Immediate action required."
        }

        description = base_messages.get(severity_level, f"Alert for {barangay_name}")

        if alert_messages:
            description += "\n\nSensor readings:\n" + "\n".join(f"• {msg}" for msg in alert_messages)

        description += "\n\nThis is an automated alert generated by the flood monitoring system."

        return description