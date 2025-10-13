#!/usr/bin/env python
"""
Test script to verify that notifications are properly triggered when thresholds are exceeded.
This script simulates adding sensor data that exceeds thresholds and checks if notifications are sent.
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flood_3.settings')
django.setup()

from core.models import Sensor, SensorData, ThresholdSetting, FloodAlert, NotificationLog, Barangay, Municipality
from core.notifications import dispatch_notifications_for_alert
from django.utils import timezone
from django.core.management import call_command

def test_notification_system():
    """Test the notification system by simulating threshold exceedances"""

    print("🧪 Testing Notification System...")

    # Create test data if it doesn't exist
    municipality, created = Municipality.objects.get_or_create(
        name="Test Municipality",
        defaults={
            'province': 'Test Province',
            'population': 10000,
            'area_sqkm': 100.0,
            'latitude': 14.5995,
            'longitude': 120.9842,
            'is_active': True
        }
    )

    barangay, created = Barangay.objects.get_or_create(
        name="Test Barangay",
        defaults={
            'municipality': municipality,
            'population': 1000,
            'area_sqkm': 10.0,
            'latitude': 14.5995,
            'longitude': 120.9842,
            'is_featured': True
        }
    )

    # Create test sensor
    sensor, created = Sensor.objects.get_or_create(
        name="Test Water Level Sensor",
        defaults={
            'sensor_type': 'water_level',
            'latitude': 14.5995,
            'longitude': 120.9842,
            'active': True,
            'barangay': barangay
        }
    )

    # Create threshold setting if it doesn't exist
    threshold, created = ThresholdSetting.objects.get_or_create(
        parameter='water_level',
        defaults={
            'advisory_threshold': 1.0,
            'watch_threshold': 2.0,
            'warning_threshold': 3.0,
            'emergency_threshold': 4.0,
            'catastrophic_threshold': 5.0,
            'unit': 'm'
        }
    )

    print(f"✅ Test data created: Municipality={municipality.name}, Barangay={barangay.name}, Sensor={sensor.name}")

    # Test 1: Add sensor data that exceeds advisory threshold
    print("\n📊 Test 1: Adding sensor data that exceeds advisory threshold (1.5m)...")
    sensor_data = SensorData.objects.create(
        sensor=sensor,
        value=1.5,  # Exceeds advisory threshold of 1.0m
        timestamp=timezone.now()
    )

    # Wait a moment for the signal handler to process
    import time
    time.sleep(3)

    # Check if any alerts were created
    alerts = FloodAlert.objects.filter(active=True)
    print(f"📋 Active alerts found: {alerts.count()}")

    for alert in alerts:
        print(f"  - {alert.title} (Severity: {alert.get_severity_level_display()})")

    # Check if notifications were sent
    notifications = NotificationLog.objects.all()
    print(f"📱 Notifications sent: {notifications.count()}")

    for notification in notifications:
        print(f"  - {notification.notification_type} to {notification.recipient} ({notification.status})")

    # Test 2: Add sensor data that exceeds emergency threshold
    print("\n📊 Test 2: Adding sensor data that exceeds emergency threshold (4.5m)...")
    sensor_data2 = SensorData.objects.create(
        sensor=sensor,
        value=4.5,  # Exceeds emergency threshold of 4.0m
        timestamp=timezone.now()
    )

    # Wait for processing
    time.sleep(3)

    # Check updated alerts
    updated_alerts = FloodAlert.objects.filter(active=True)
    print(f"📋 Updated active alerts found: {updated_alerts.count()}")

    for alert in updated_alerts:
        print(f"  - {alert.title} (Severity: {alert.get_severity_level_display()})")

    # Check new notifications
    new_notifications = NotificationLog.objects.all()
    print(f"📱 Total notifications sent: {new_notifications.count()}")

    # Test 3: Run threshold evaluation command manually
    print("\n🔄 Test 3: Running threshold evaluation command manually...")
    try:
        call_command('apply_thresholds_all', verbosity=2)
        print("✅ Threshold evaluation command completed successfully")
    except Exception as e:
        print(f"❌ Error running threshold evaluation: {e}")

    # Final summary
    print("\n📊 Final Summary:")
    print(f"  - Active alerts: {FloodAlert.objects.filter(active=True).count()}")
    print(f"  - Total notifications: {NotificationLog.objects.count()}")
    print(f"  - Threshold settings: {ThresholdSetting.objects.count()}")

    print("\n🎉 Notification system test completed!")

if __name__ == '__main__':
    test_notification_system()
