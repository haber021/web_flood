from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db.models import Q
from datetime import timedelta

from core.models import (
    Barangay,
    Sensor,
    SensorData,
    ThresholdSetting,
    FloodAlert,
)
from core.notifications import dispatch_notifications_for_alert


def evaluate_severity(value: float, ts: ThresholdSetting) -> int:
    """Return severity level 0-5 for a value against a ThresholdSetting.
    0 means below advisory (no alert). 1..5 correspond to Advisory..Catastrophic.
    """
    if value is None or ts is None:
        return 0
    if value >= ts.catastrophic_threshold:
        return 5
    if value >= ts.emergency_threshold:
        return 4
    if value >= ts.warning_threshold:
        return 3
    if value >= ts.watch_threshold:
        return 2
    if value >= ts.advisory_threshold:
        return 1
    return 0


class Command(BaseCommand):
    help = "Apply threshold evaluation across all barangays and create/update targeted alerts."

    def add_arguments(self, parser):
        parser.add_argument(
            "--barangay-id",
            type=int,
            help="Only evaluate a single barangay by ID.",
        )
        parser.add_argument(
            "--municipality-id",
            type=int,
            help="Only evaluate barangays within a municipality ID.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Do not create/update alerts; only log what would happen.",
        )

    def handle(self, *args, **options):
        barangay_id = options.get("barangay_id")
        municipality_id = options.get("municipality_id")
        dry_run = options.get("dry_run", False)

        # Load all configured thresholds indexed by parameter
        thresholds = {t.parameter: t for t in ThresholdSetting.objects.all()}
        if not thresholds:
            self.stdout.write(self.style.WARNING("No ThresholdSetting records found. Nothing to apply."))
            return

        # Build barangay queryset per filters
        b_qs = Barangay.objects.all()
        if barangay_id:
            b_qs = b_qs.filter(id=barangay_id)
        if municipality_id:
            b_qs = b_qs.filter(municipality_id=municipality_id)

        total_processed = 0
        total_alerts_created = 0
        total_alerts_updated = 0
        now = timezone.now()

        for b in b_qs.iterator():
            total_processed += 1
            exceeded_details = []  # list of dicts {parameter, value, unit, severity}
            highest_severity = 0

            # For each configured parameter, get latest reading for sensors under this barangay
            for param, ts in thresholds.items():
                latest = (
                    SensorData.objects.filter(
                        sensor__sensor_type=param,
                        sensor__barangay=b,
                    )
                    .order_by("-timestamp")
                    .first()
                )
                if not latest:
                    continue

                sev = evaluate_severity(latest.value, ts)
                if sev > 0:
                    exceeded_details.append(
                        {
                            "parameter": param,
                            "value": latest.value,
                            "unit": ts.unit,
                            "severity": sev,
                        }
                    )
                    highest_severity = max(highest_severity, sev)

            if highest_severity == 0:
                # Nothing exceeded for this barangay
                self.stdout.write(
                    f"Barangay {b.id} - {b.name}: no thresholds exceeded."
                )
                continue

            # Build description
            details_lines = []
            sev_name = {
                1: "Advisory",
                2: "Watch",
                3: "Warning",
                4: "Emergency",
                5: "Catastrophic",
            }
            for d in sorted(exceeded_details, key=lambda x: (-x["severity"], x["parameter"])):
                details_lines.append(
                    f"- {d['parameter'].replace('_', ' ').title()}: {d['value']} {d['unit']} (> {sev_name[d['severity']]})"
                )
            description = (
                f"Automated threshold evaluation at {now.strftime('%Y-%m-%d %H:%M:%S %Z')} for {b.name}.\n"
                f"Highest severity: {sev_name[highest_severity]}.\n"
                f"Exceeded parameters:\n" + ("\n".join(details_lines))
            )

            title_prefix = f"Automated Alert for {b.name}"

            # Find existing active automated alert targeting this barangay
            existing = (
                FloodAlert.objects.filter(
                    active=True,
                    affected_barangays=b,
                    title__startswith=title_prefix,
                )
                .order_by("-issued_at")
                .first()
            )

            if dry_run:
                if existing:
                    action = (
                        "update (severity increase)"
                        if highest_severity > existing.severity_level
                        else "no change"
                    )
                else:
                    action = "create"
                self.stdout.write(
                    self.style.NOTICE(
                        f"[DRY-RUN] {title_prefix}: {action}; highest={highest_severity}"
                    )
                )
                continue

            if existing:
                # Always update the alert details
                severity_changed = existing.severity_level != highest_severity
                description_changed = existing.description != description
                if severity_changed or description_changed:
                    existing.severity_level = highest_severity
                    existing.title = f"{title_prefix}: {sev_name[highest_severity]}"
                    existing.description = description
                    existing.updated_at = timezone.now()
                    existing.save()

                # Decide whether to dispatch notifications
                should_dispatch = False
                if severity_changed and highest_severity > existing.severity_level:
                    # Severity increased, always dispatch
                    should_dispatch = True
                elif existing.last_notification_sent_at is None:
                    # Never sent before, dispatch
                    should_dispatch = True
                elif (now - existing.last_notification_sent_at) >= timedelta(hours=3):
                    # At least 3 hours since last notification, dispatch update
                    should_dispatch = True

                if should_dispatch:
                    dispatch_notifications_for_alert(existing)
                    existing.last_notification_sent_at = now
                    existing.save()
                    total_alerts_updated += 1
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"Updated alert for {b.name} to severity {existing.severity_level} and sent notifications."
                        )
                    )
                else:
                    self.stdout.write(f"No notification sent for {b.name}; within cooldown period.")
            else:
                # Create new alert
                alert = FloodAlert.objects.create(
                    title=f"{title_prefix}: {sev_name[highest_severity]}",
                    description=description,
                    severity_level=highest_severity,
                    active=True,
                )
                alert.affected_barangays.set([b])
                dispatch_notifications_for_alert(alert)
                alert.last_notification_sent_at = now
                alert.save()
                total_alerts_created += 1
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Created alert for {b.name} with severity {highest_severity}."
                    )
                )

        self.stdout.write(
            self.style.SUCCESS(
                f"Done. Barangays processed: {total_processed}. Created: {total_alerts_created}, Updated: {total_alerts_updated}."
            )
        )
