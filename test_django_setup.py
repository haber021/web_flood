#!/usr/bin/env python
"""
Test script to diagnose Django setup issues
"""

import os
import sys
import traceback

# Add the current directory to Python path
sys.path.insert(0, 'c:\\Users\\PC\\Desktop\\flood_3')

# Set Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flood_monitoring.settings')

try:
    print("1. Importing Django...")
    import django
    print("   ✓ Django imported successfully")

    print("2. Setting up Django...")
    django.setup()
    print("   ✓ Django setup successful")

    print("3. Importing models...")
    from core.models import Sensor, SensorData
    print("   ✓ Models imported successfully")

    print("4. Testing database connection...")
    from django.db import connection
    cursor = connection.cursor()
    print("   ✓ Database connection successful")

    print("\n🎉 All tests passed! Django is working correctly.")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
