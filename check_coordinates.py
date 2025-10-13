#!/usr/bin/env python
import os
import sys
import django

# Add the project directory to the Python path
sys.path.append('c:/Users/PC/Desktop/flood_02')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flood_monitoring.settings')

# Setup Django
django.setup()

from core.models import Barangay, Municipality

print("Checking coordinates in database...")
print("=" * 50)

# Check barangays
print("Barangays:")
barangays = Barangay.objects.all()[:5]
for b in barangays:
    print(f"  {b.name}: lat={b.latitude}, lng={b.longitude}")

print("\nMunicipalities:")
municipalities = Municipality.objects.all()[:5]
for m in municipalities:
    print(f"  {m.name}: lat={m.latitude}, lng={m.longitude}")

print("\n" + "=" * 50)
print("Checking for invalid coordinates...")

# Check for invalid coordinates
invalid_barangays = Barangay.objects.filter(
    Q(latitude__isnull=True) | Q(longitude__isnull=True) |
    Q(latitude=0) | Q(longitude=0)
)[:5]

if invalid_barangays:
    print(f"Found {len(invalid_barangays)} barangays with invalid coordinates:")
    for b in invalid_barangays:
        print(f"  {b.name}: lat={b.latitude}, lng={b.longitude}")
else:
    print("No barangays with obviously invalid coordinates found.")
