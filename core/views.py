from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.views import LoginView, PasswordResetView
from django.contrib.auth import login, logout
from django.contrib.auth.forms import PasswordResetForm
from django.contrib.auth.models import User, Group
from django.contrib import messages
from django.db.models import Avg, Max, Min, Q, Count
from django.utils import timezone
from django.http import JsonResponse, HttpResponseForbidden
from django.core.paginator import Paginator
from datetime import timedelta
from django.db.models.functions import TruncYear
import json
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
from .models import (
    Sensor, SensorData, Barangay, FloodRiskZone, Municipality,
    FloodAlert, ThresholdSetting, NotificationLog, EmergencyContact, UserProfile,
    ResilienceScore
)
from .forms import FloodAlertForm, ThresholdSettingForm, BarangaySearchForm, RegisterForm, UserProfileForm
from .forms import SensorForm
from .notifications import dispatch_notifications_for_alert


def _iso_timestamp(dt):
    """Return an ISO 8601 UTC timestamp string for a datetime `dt`.
    Handles naive datetimes by making them aware using the project's default timezone.
    """
    if not dt:
        return None
    # Ensure aware
    try:
        if timezone.is_naive(dt):
            dt = timezone.make_aware(dt, timezone.get_default_timezone())
    except Exception:
        # If any issue, fall back to the object's isoformat
        try:
            return dt.isoformat()
        except Exception:
            return None

    # Convert to UTC for a consistent client-side conversion target
    try:
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return dt.isoformat()


def _manila_str(dt, include_seconds=True, include_date=True):
    """Return a human-readable Manila time string in 24-hour format.
    Example: '15 Sep 2025 15:11:20' or '15 Sep 2025 15:11' depending on include_seconds.
    """
    if not dt:
        return None
    try:
        if timezone.is_naive(dt):
            dt = timezone.make_aware(dt, timezone.get_default_timezone())
    except Exception:
        pass

    try:
        if ZoneInfo:
            manila_tz = ZoneInfo('Asia/Manila')
            local_dt = dt.astimezone(manila_tz)
        else:
            # Fallback: use django timezone convert to offset (may use system tz)
            local_dt = timezone.localtime(dt)
    except Exception:
        local_dt = dt

    if include_date:
        if include_seconds:
            return local_dt.strftime('%d %b %Y %H:%M:%S')
        return local_dt.strftime('%d %b %Y %H:%M')
    else:
        if include_seconds:
            return local_dt.strftime('%H:%M:%S')
        return local_dt.strftime('%H:%M')

class CustomLoginView(LoginView):
    template_name = 'login.html'
    redirect_authenticated_user = True

class CustomPasswordResetView(PasswordResetView):
    template_name = 'password_reset.html'
    form_class = PasswordResetForm
    success_url = '/login/'

def register(request):
    """User registration view"""
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Log the user in immediately after registration
            login(request, user)
            messages.success(request, f'Account created for {user.username}!')
            return redirect('dashboard')
    else:
        form = RegisterForm()
    
    return render(request, 'register.html', {'form': form})

@login_required
def logout_view(request):
    """Log the user out and redirect to login page"""
    logout(request)
    messages.success(request, 'You have been logged out.')
    return redirect('login')

@login_required
def dashboard(request):
    """Main dashboard view showing all real-time data and visualizations"""
    # Get latest sensor readings
    latest_temperature = SensorData.objects.filter(
        sensor__sensor_type='temperature'
    ).order_by('-timestamp').first()
    
    latest_humidity = SensorData.objects.filter(
        sensor__sensor_type='humidity'
    ).order_by('-timestamp').first()
    
    latest_rainfall = SensorData.objects.filter(
        sensor__sensor_type='rainfall'
    ).order_by('-timestamp').first()
    
    latest_water_level = SensorData.objects.filter(
        sensor__sensor_type='water_level'
    ).order_by('-timestamp').first()
    
    latest_wind_speed = SensorData.objects.filter(
        sensor__sensor_type='wind_speed'
    ).order_by('-timestamp').first()
    
    # Get active alerts
    active_alerts = FloodAlert.objects.filter(active=True).order_by('-severity_level')
    
    # Get sensor locations for map
    sensors = Sensor.objects.filter(active=True)
    
    # Get flood risk zones for map
    flood_zones = FloodRiskZone.objects.all()
    
    context = {
        'latest_temperature': latest_temperature,
        'latest_humidity': latest_humidity,
        'latest_rainfall': latest_rainfall,
        'latest_water_level': latest_water_level,
        'latest_wind_speed': latest_wind_speed,
        'active_alerts': active_alerts,
        'sensors': sensors,
        'flood_zones': flood_zones,
        'page': 'dashboard',
        # Server-rendered Manila time for the header (24-hour)
        'current_time_manila': _manila_str(timezone.now(), include_seconds=True, include_date=True)
    }
    
    return render(request, 'dashboard.html', context)

@login_required
def prediction_page(request):
    """Flood prediction page view"""
    # Get historic data for predictions
    end_date = timezone.now()
    start_date = end_date - timedelta(days=7)
    
    # Get rainfall data for the past week
    rainfall_data = SensorData.objects.filter(
        sensor__sensor_type='rainfall',
        timestamp__gte=start_date,
        timestamp__lte=end_date
    ).order_by('timestamp')
    
    # Get water level data for the past week
    water_level_data = SensorData.objects.filter(
        sensor__sensor_type='water_level',
        timestamp__gte=start_date,
        timestamp__lte=end_date
    ).order_by('timestamp')
    
    # Current active alerts
    active_alerts = FloodAlert.objects.filter(active=True)
    
    # Get all barangays for the alert form
    barangays = Barangay.objects.all()
    
    # Create new alert form
    form = FloodAlertForm()
    
    context = {
        'rainfall_data': rainfall_data,
        'water_level_data': water_level_data,
        'active_alerts': active_alerts,
        'barangays': barangays,
        'form': form,
        'page': 'prediction'
    }
    
    return render(request, 'prediction.html', context)

@login_required
def create_alert(request):
    """Create a new flood alert"""
    if request.method == 'POST':
        form = FloodAlertForm(request.POST)
        if form.is_valid():
            alert = form.save(commit=False)
            alert.issued_by = request.user
            alert.save()
            # Save many-to-many relationships
            form.save_m2m()

            # Check SMS configuration before dispatching
            import os
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
            sms_enabled = os.getenv('SMS_ENABLED', 'true').lower() == 'true'

            if sms_enabled:
                # Dispatch notifications to affected contacts
                dispatch_notifications_for_alert(alert)

                # Check notification results
                from .models import NotificationLog
                sms_logs = NotificationLog.objects.filter(alert=alert, notification_type='sms')
                sent_count = sms_logs.filter(status='sent').count()
                failed_count = sms_logs.filter(status='failed').count()
                pending_count = sms_logs.filter(status='pending').count()

                if failed_count > 0:
                    messages.warning(request, f"Alert created successfully! {sent_count} SMS sent, {failed_count} failed (possible rate limits or insufficient credits). Your Twilio credentials are valid but may have reached daily limits.")
                elif pending_count > 0:
                    messages.success(request, f"Alert created successfully! SMS notifications are pending (SMS currently disabled). Your Twilio credentials are valid and ready to send messages.")
                else:
                    messages.success(request, f"Alert created successfully! {sent_count} SMS notifications sent to barangay contacts. Your Twilio credentials are valid and working properly.")
            else:
                messages.success(request, "Alert created successfully! SMS notifications are currently disabled. Your Twilio credentials are valid and ready to send messages when enabled.")

            return redirect('prediction_page')
        else:
            messages.error(request, "Error creating alert. Please check the form.")

    return redirect('prediction_page')

@login_required
def barangays_page(request):
    """Barangays management page view"""
    form = BarangaySearchForm(request.GET or None)
    barangays = Barangay.objects.all()
    
    # Apply search and filtering if form is valid
    if form.is_valid():
        name_query = form.cleaned_data.get('name')
        severity_level = form.cleaned_data.get('severity_level')
        
        if name_query:
            barangays = barangays.filter(name__icontains=name_query)
        
        if severity_level:
            # Filter barangays affected by alerts with the given severity level
            barangays = barangays.filter(flood_alerts__severity_level=severity_level).distinct()
    
    context = {
        'barangays': barangays,
        'form': form,
        'page': 'barangays'
    }
    
    return render(request, 'barangays.html', context)

@login_required
def barangay_detail(request, barangay_id):
    """Detailed view for a specific barangay"""
    barangay = get_object_or_404(Barangay, id=barangay_id)
    
    # Get flood alerts affecting this barangay
    alerts = FloodAlert.objects.filter(affected_barangays=barangay).order_by('-issued_at')
    
    # Get emergency contacts for this barangay
    contacts = EmergencyContact.objects.filter(barangay=barangay)
    
    # Get flood history for the chart (last 5 years)
    current_year = timezone.now().year
    years_range = range(current_year - 4, current_year + 1)
    
    # Get counts of alerts per year
    alert_counts_by_year = (
        alerts.annotate(year=TruncYear('issued_at'))
        .values('year')
        .annotate(count=Count('id'))
        .order_by('year')
    )
    
    # Prepare data for the chart, ensuring all years in the range are present
    history_data = {str(year): 0 for year in years_range}
    for item in alert_counts_by_year:
        history_data[str(item['year'].year)] = item['count']
        
    context = {
        'barangay': barangay,
        'alerts': alerts,
        'contacts': contacts,
        'page': 'barangays',
        'flood_history_json': json.dumps(history_data)
    }
    
    return render(request, 'barangay_detail.html', context)

@login_required
def notifications_page(request):
    """Notifications center page view"""
    # Get all alerts
    alerts = FloodAlert.objects.all().order_by('-issued_at')

    # Get notification logs (latest 10 only)
    notifications = NotificationLog.objects.all().order_by('-sent_at')[:10]

    # Get emergency contacts
    contacts = EmergencyContact.objects.all()

    # Calculate analytics for all notifications (not just the displayed 10)
    all_notifications = NotificationLog.objects.all()
    total = all_notifications.count()
    delivered = all_notifications.filter(status='delivered').count()
    delivery_rate = (delivered / total * 100) if total > 0 else 0
    notification_counts = {
        'total': total,
        'sms': all_notifications.filter(notification_type='sms').count(),
        'email': all_notifications.filter(notification_type='email').count(),
        'app': all_notifications.filter(notification_type='app').count(),
        'delivered': delivered,
        'sent': all_notifications.filter(status='sent').count(),
        'failed': all_notifications.filter(status='failed').count(),
        'pending': all_notifications.filter(status='pending').count(),
        'delivery_rate': delivery_rate,
    }

    context = {
        'alerts': alerts,
        'notifications': notifications,
        'contacts': contacts,
        'notification_counts': notification_counts,
        'page': 'notifications'
    }

    return render(request, 'notifications.html', context)

@login_required
def config_page(request):
    """System configuration page view"""
    # Get all threshold settings
    thresholds = ThresholdSetting.objects.all()
    
    # Create form for updating thresholds
    form = ThresholdSettingForm()
    
    if request.method == 'POST':
        form = ThresholdSettingForm(request.POST)
        if form.is_valid():
            threshold = form.save(commit=False)
            threshold.last_updated_by = request.user
            
            # Check if a threshold for this parameter already exists
            existing = ThresholdSetting.objects.filter(parameter=threshold.parameter).first()
            if existing:
                # Update existing threshold
                existing.advisory_threshold = threshold.advisory_threshold
                existing.watch_threshold = threshold.watch_threshold
                existing.warning_threshold = threshold.warning_threshold
                existing.emergency_threshold = threshold.emergency_threshold
                existing.catastrophic_threshold = threshold.catastrophic_threshold
                existing.unit = threshold.unit
                existing.last_updated_by = request.user
                existing.save()
                messages.success(request, f"{threshold.parameter} thresholds updated successfully.")
            else:
                # Save new threshold
                threshold.save()
                messages.success(request, f"{threshold.parameter} thresholds created successfully.")
            
            return redirect('config_page')
    
    context = {
        'thresholds': thresholds,
        'form': form,
        'page': 'config'
    }
    
    return render(request, 'config.html', context)

def get_chart_data(request):
    """API endpoint to get chart data for the dashboard (no login required).
    Supports two modes:
    - Range mode (default): last `days` days of data
    - Limit mode: if `limit` is provided (>0), returns the latest N data points
    Both modes respect optional municipality_id and barangay_id filters.
    """
    chart_type = request.GET.get('type', 'temperature')
    # Optional: fetch latest N readings or by time window
    # Backwards/forwards compatibility: support 'range' (latest|1w|1m|1y)
    range_param = (request.GET.get('range') or '').strip().lower()
    try:
        limit = int(request.GET.get('limit')) if request.GET.get('limit') is not None else None
    except Exception:
        limit = None
    try:
        days = int(request.GET.get('days')) if request.GET.get('days') is not None else None
    except Exception:
        days = None
    # Map range -> limit/days only if explicit limit/days were not provided
    if (limit is None and days is None) and range_param:
        if range_param == 'latest':
            limit = 10
        elif range_param == '1w':
            days = 7
        elif range_param == '1m':
            days = 30
        elif range_param == '1y':
            days = 365
    # Final fallbacks/guards
    if limit is not None and limit <= 0:
        limit = None
    if days is None:
        days = 1
    elif days <= 0:
        days = 1
    historical = request.GET.get('historical', 'false').lower() == 'true'
    
    # Get location filters if provided
    municipality_id = request.GET.get('municipality_id')
    barangay_id = request.GET.get('barangay_id')
    
    # Base queryset with optional location filters
    base_filters = {
        'sensor__sensor_type': chart_type,
    }
    if municipality_id:
        base_filters['sensor__municipality_id'] = municipality_id
    if barangay_id:
        base_filters['sensor__barangay_id'] = barangay_id

    # Determine the most specific sensor query set based on filters
    sensor_qs = Sensor.objects.filter(sensor_type=chart_type)
    if barangay_id:
        sensor_qs = sensor_qs.filter(barangay_id=barangay_id)
    elif municipality_id:
        sensor_qs = sensor_qs.filter(municipality_id=municipality_id)

    if limit and limit > 0:
        # Limit mode: choose a single representative sensor (with most recent reading)
        # to avoid mixing multiple sensors in one line chart.

        # Pick the sensor with the most recent reading
        try:
            top_sensor = (
                sensor_qs
                .annotate(last_ts=Max('sensordata__timestamp'))
                .order_by('-last_ts')
                .first()
            )
        except Exception:
            top_sensor = None

        if top_sensor:
            qs = SensorData.objects.filter(sensor=top_sensor).order_by('-timestamp')
        else:
            qs = SensorData.objects.filter(sensor__sensor_type=chart_type).order_by('-timestamp')

        rows = list(qs[:limit])
        rows.reverse()  # chronological order for chart
        data = rows
    else:
        # Range mode: last `days`
        end_date = timezone.now()
        start_date = end_date - timedelta(days=days)

        try:
            top_sensor = (
                sensor_qs
                .annotate(last_ts=Max('sensordata__timestamp'))
                .order_by('-last_ts')
                .first()
            )
        except Exception:
            top_sensor = None

        if top_sensor:
            filters = {
                'sensor': top_sensor,
                'timestamp__gte': start_date,
                'timestamp__lte': end_date
            }
            data = SensorData.objects.filter(**filters).order_by('timestamp')
        else:
            global_filters = {
                'sensor__sensor_type': chart_type,
                'timestamp__gte': start_date,
                'timestamp__lte': end_date
            }
            data = SensorData.objects.filter(**global_filters).order_by('timestamp')

    # Backend fallbacks when no data collected so far
    try:
        if not data:
            if limit and limit > 0:
                # Retry without location filters across all sensors of this type
                alt = (
                    SensorData.objects
                    .filter(sensor__sensor_type=chart_type)
                    .order_by('-timestamp')[:limit]
                )
                data = list(alt)
                data.reverse()
            else:
                # Retry across all sensors of this type in the time window
                alt = (
                    SensorData.objects
                    .filter(sensor__sensor_type=chart_type, timestamp__gte=start_date, timestamp__lte=end_date)
                    .order_by('timestamp')
                )
                data = list(alt)
    except Exception:
        pass
    
    # Format timestamps based on the time range
    if days >= 30:  # Monthly view
        date_format = '%b %d'
    elif days >= 7:  # Weekly view
        date_format = '%b %d'
    elif days > 1:  # Multi-day view
        date_format = '%m/%d %H:%M'
    else:  # Single day view
        date_format = '%H:%M'
    
    # Return timestamps in ISO 8601 (UTC) so frontend can convert to Asia/Manila reliably
    chart_data = {
        # ISO timestamps (UTC) for programmatic parsing on the client
        'labels': [_iso_timestamp(reading.timestamp) for reading in data],
        # Human-readable Manila strings for quick display (24-hour)
        'labels_manila': [_manila_str(reading.timestamp, include_seconds=True, include_date=(days!=1)) for reading in data],
        'values': [reading.value for reading in data],
    }
    
    # Add historical comparison data if requested
    if historical:
        # In a real app, we would query historical data from previous years
        # For this example, we'll use a simplified approach
        chart_data['historical_values'] = [float(f"{value * 0.85:.2f}") for value in chart_data['values']]
        
        # Add threshold values based on sensor type
        if chart_type == 'water_level':
            chart_data['threshold_value'] = 1.5  # Default flood threshold in meters
        elif chart_type == 'rainfall':
            chart_data['threshold_value'] = 25.0  # Heavy rainfall threshold in mm
    
    return JsonResponse(chart_data)

def get_map_data(request):
    """API endpoint to get map data (no login required)"""
    # This API endpoint is accessible without login for map visualization
    # Optional filters
    barangay_id = request.GET.get('barangay_id', None)
    municipality_id = request.GET.get('municipality_id', None)
    
    # Initialize empty lists to use in JSON response
    sensor_data = []
    zone_data = []
    barangay_data = []
    
    try:
        # Get sensors for map
        sensors = Sensor.objects.filter(active=True)
        if municipality_id:
            # Filter sensors by municipality if requested
            sensors = sensors.filter(Q(municipality_id=municipality_id) | Q(municipality=None))
            
        # Process each sensor
        for sensor in sensors:
            try:
                latest_reading = SensorData.objects.filter(sensor=sensor).order_by('-timestamp').first()
                value = latest_reading.value if latest_reading else None
                
                sensor_data.append({
                    'id': sensor.id,
                    'name': sensor.name,
                    'type': sensor.sensor_type,
                    'lat': sensor.latitude,
                    'lng': sensor.longitude,
                    'value': value,
                    'unit': get_unit_for_sensor_type(sensor.sensor_type),
                })
            except Exception as e:
                print(f"Error processing sensor {sensor.id}: {str(e)}")
                # Continue to next sensor
        
        # Get flood risk zones
        zones = FloodRiskZone.objects.all()
        
        for zone in zones:
            zone_data.append({
                'id': zone.id,
                'name': zone.name,
                'severity': zone.severity_level,
                'geojson': zone.geojson,
            })
            
        # Get all barangays, filter by municipality if provided
        barangay_queryset = Barangay.objects.all()
        if municipality_id:
            barangay_queryset = barangay_queryset.filter(municipality_id=municipality_id)
        
        # First, get all active alerts
        active_alerts = FloodAlert.objects.filter(active=True)
        if municipality_id:
            active_alerts = active_alerts.filter(affected_barangays__municipality_id=municipality_id).distinct()
        
        # Get affected barangays with their severities
        alert_severity_by_barangay = {}
        for alert in active_alerts:
            for barangay in alert.affected_barangays.all():
                # Track the highest severity level for each barangay
                current_severity = alert_severity_by_barangay.get(barangay.id, 0)
                alert_severity_by_barangay[barangay.id] = max(current_severity, alert.severity_level)

        # Load all threshold settings for parameter severity calculation
        thresholds = {t.parameter: t for t in ThresholdSetting.objects.all()}

        # Build barangay data including all barangays
        for barangay in barangay_queryset:
            # Use the highest severity from alerts, or 0 if not affected
            severity = alert_severity_by_barangay.get(barangay.id, 0)

            # Calculate parameter-specific severities
            param_severities = {}
            for param, ts in thresholds.items():
                # Find the latest sensor reading relevant to this barangay
                # Fallback order: barangay-specific -> municipality-wide -> global
                latest_reading = SensorData.objects.filter(
                    sensor__sensor_type=param, sensor__barangay=barangay
                ).order_by('-timestamp').first()

                if not latest_reading and barangay.municipality_id:
                    latest_reading = SensorData.objects.filter(
                        sensor__sensor_type=param, sensor__municipality_id=barangay.municipality_id
                    ).order_by('-timestamp').first()

                if not latest_reading:
                    latest_reading = SensorData.objects.filter(
                        sensor__sensor_type=param
                    ).order_by('-timestamp').first()

                if latest_reading:
                    value = latest_reading.value
                    level = 0
                    if value >= ts.catastrophic_threshold:
                        level = 5
                    elif value >= ts.emergency_threshold:
                        level = 4
                    elif value >= ts.warning_threshold:
                        level = 3
                    elif value >= ts.watch_threshold:
                        level = 2
                    elif value >= ts.advisory_threshold:
                        level = 1
                    param_severities[param] = level
                else:
                    param_severities[param] = 0

            # Include municipality information
            municipality_name = barangay.municipality.name if barangay.municipality else "-"

            barangay_data.append({
                'id': barangay.id,
                'name': barangay.name,
                'population': barangay.population,
                'municipality_id': barangay.municipality_id,
                'municipality_name': municipality_name,
                'lat': barangay.latitude,
                'lng': barangay.longitude,
                'severity': severity,
                'param_severities': param_severities,
                # Add extra data
                'contact_person': barangay.contact_person,
                'contact_number': barangay.contact_number
            })
        
    except Exception as e:
        # Log the error but still return what we have
        print(f"Error in get_map_data: {str(e)}")
    
    # Return map data with whatever we've collected
    map_data = {
        'sensors': sensor_data,
        'zones': zone_data,
        'barangays': barangay_data,
    }
    
    return JsonResponse(map_data)


def get_heatmap_points(request):
    """API endpoint to get data points for heatmap visualization based on threshold breaches."""
    municipality_id = request.GET.get('municipality_id')
    barangay_id = request.GET.get('barangay_id')

    # Load all threshold settings into a dictionary for quick lookups
    thresholds = {t.parameter: t for t in ThresholdSetting.objects.all()}
    if not thresholds:
        return JsonResponse({'points': []}) # No thresholds, no heatmap

    # Filter barangays based on request
    barangay_qs = Barangay.objects.all()
    if municipality_id:
        barangay_qs = barangay_qs.filter(municipality_id=municipality_id)
    if barangay_id:
        barangay_qs = barangay_qs.filter(id=barangay_id)

    heat_points = []
    for barangay in barangay_qs:
        if not (barangay.latitude and barangay.longitude):
            continue

        # For each barangay, calculate a total risk score based on sensor readings
        total_risk_score = 0
        for param, ts in thresholds.items():
            # Find the latest sensor reading relevant to this barangay
            # Fallback order: barangay-specific -> municipality-wide -> global
            latest_reading = SensorData.objects.filter(
                sensor__sensor_type=param, sensor__barangay=barangay
            ).order_by('-timestamp').first()

            if not latest_reading and barangay.municipality_id:
                latest_reading = SensorData.objects.filter(
                    sensor__sensor_type=param, sensor__municipality_id=barangay.municipality_id
                ).order_by('-timestamp').first()

            if not latest_reading:
                latest_reading = SensorData.objects.filter(
                    sensor__sensor_type=param
                ).order_by('-timestamp').first()

            if not latest_reading:
                continue

            # Check value against thresholds to get a severity level (0-5)
            value = latest_reading.value
            if value >= ts.catastrophic_threshold: total_risk_score += 5
            elif value >= ts.emergency_threshold:    total_risk_score += 4
            elif value >= ts.warning_threshold:      total_risk_score += 3
            elif value >= ts.watch_threshold:        total_risk_score += 2
            elif value >= ts.advisory_threshold:     total_risk_score += 1

        # Normalize the total risk score to an intensity value (0.0 to 1.0)
        # Max possible score is 5 params * 5 severity = 25
        intensity = min(1.0, total_risk_score / 15.0) # Normalize against a score of 15 for good visual spread

        if intensity > 0.1: # Only show areas with some level of risk
            heat_points.append([barangay.latitude, barangay.longitude, round(intensity, 3)])

    return JsonResponse({'points': heat_points})


def weather_dashboard(request):
    """Public page: Polished Weather Conditions dashboard UI."""
    return render(request, 'weather_dashboard.html')


@login_required
def add_sensor(request):
    """View to add a new Sensor via a form (admin/manager only)"""
    # Runtime permission check (avoids decorator ordering issues)
    if not is_admin_or_manager(request.user):
        return HttpResponseForbidden('You do not have permission to add sensors.')
    if request.method == 'POST':
        form = SensorForm(request.POST)
        if form.is_valid():
            sensor = form.save()
            messages.success(request, f"Sensor '{sensor.name}' created successfully.")
            return redirect('dashboard')
    else:
        form = SensorForm()

    context = {
        'form': form,
        'page': 'add_sensor'
    }
    return render(request, 'add_sensor.html', context)


@login_required
def add_municipality(request):
    """View to add a new Municipality (admin/manager only)"""
    if not is_admin_or_manager(request.user):
        return HttpResponseForbidden('You do not have permission to add municipalities.')

    from .forms import MunicipalityForm

    if request.method == 'POST':
        form = MunicipalityForm(request.POST)
        if form.is_valid():
            # Some smaller creation UIs (AJAX/modal) may only provide name/province/is_active.
            # Municipality model requires population, area_sqkm, latitude, longitude (non-nullable).
            # Save with commit=False and set safe defaults for missing numeric fields to avoid IntegrityError.
            m = form.save(commit=False)
            # Set defaults if not provided
            if not m.population:
                m.population = 0
            if not m.area_sqkm:
                m.area_sqkm = 0.0
            if not m.latitude:
                m.latitude = 0.0
            if not m.longitude:
                m.longitude = 0.0
            m.save()
            messages.success(request, f"Municipality '{m.name}' created successfully.")
            return redirect('dashboard')
    else:
        form = MunicipalityForm()

    return render(request, 'add_municipality.html', {'form': form, 'page': 'add_municipality'})


@login_required
def add_barangay(request):
    """View to add a new Barangay (admin/manager only)"""
    if not is_admin_or_manager(request.user):
        return HttpResponseForbidden('You do not have permission to add barangays.')

    from .forms import BarangayForm

    if request.method == 'POST':
        form = BarangayForm(request.POST)
        if form.is_valid():
            # Prevent duplicates: if a barangay with the same name (case-insensitive)
            # already exists for the selected municipality, return/redirect to avoid duplicate rows.
            name = form.cleaned_data.get('name', '').strip()
            municipality = form.cleaned_data.get('municipality')

            existing = None
            if name:
                qs = Barangay.objects.filter(name__iexact=name)
                if municipality:
                    qs = qs.filter(municipality=municipality)
                existing = qs.first()

            if existing:
                messages.info(request, f"Barangay '{existing.name}' already exists.")
                return redirect('dashboard')

            # Save safely with defaults for numeric fields when missing
            b = form.save(commit=False)
            if not getattr(b, 'population', None):
                b.population = 0
            if not getattr(b, 'area_sqkm', None):
                b.area_sqkm = 0.0
            if not getattr(b, 'latitude', None):
                b.latitude = 0.0
            if not getattr(b, 'longitude', None):
                b.longitude = 0.0
            b.save()
            messages.success(request, f"Barangay '{b.name}' created successfully.")
            return redirect('dashboard')
    else:
        form = BarangayForm()

    return render(request, 'add_barangay.html', {'form': form, 'page': 'add_barangay'})

def get_unit_for_sensor_type(sensor_type):
    """Helper function to get the unit for a sensor type"""
    units = {
        'temperature': '°C',
        'humidity': '%',
        'rainfall': 'mm',
        'water_level': 'm',
        'wind_speed': 'km/h',
    }
    return units.get(sensor_type, '')

def get_latest_sensor_data(request):
    """API endpoint to get the latest sensor data (no login required)"""
    # This API endpoint is accessible without login for dashboard visualization
    # Get limit parameter (default to 5)
    limit = int(request.GET.get('limit', 5))
    
    # Get municipality filter if provided
    municipality_id = request.GET.get('municipality_id')
    
    # Get the latest reading for each sensor type
    latest_readings = []
    
    for sensor_type in ['temperature', 'humidity', 'rainfall', 'water_level', 'wind_speed']:
        # Build the filter
        filters = {'sensor__sensor_type': sensor_type}
        
        # Add municipality filter if provided
        if municipality_id:
            filters['sensor__municipality_id'] = municipality_id
        
        # Get the reading with filters
        reading = SensorData.objects.filter(**filters).order_by('-timestamp').first()
        
        # If no municipality-specific reading, try fallback to global sensors
        if not reading and municipality_id:
            print(f"No {sensor_type} data found for municipality {municipality_id}, using global data")
            reading = SensorData.objects.filter(sensor__sensor_type=sensor_type).order_by('-timestamp').first()
        
        if reading:
            latest_readings.append({
                'id': reading.id,
                'sensor_id': reading.sensor.id,
                'sensor_name': reading.sensor.name,
                'sensor_type': reading.sensor.sensor_type,
                'value': reading.value,
                'timestamp': _iso_timestamp(reading.timestamp),
                'timestamp_manila': _manila_str(reading.timestamp, include_seconds=True, include_date=True),
                'municipality_id': reading.sensor.municipality_id if reading.sensor.municipality else None,
                'municipality_name': reading.sensor.municipality.name if reading.sensor.municipality else 'Global',
                'unit': get_unit_for_sensor_type(reading.sensor.sensor_type)
            })
    
    # Return as JSON
    return JsonResponse({
        'count': len(latest_readings),
        'results': latest_readings[:limit]
    })

def get_flood_alerts(request):
    """API endpoint to get flood alerts (no login required)"""
    # This API endpoint is accessible without login for alerts visualization
    # Check if we only want active alerts
    active_only = request.GET.get('active', 'false').lower() == 'true'
    
    # Get alerts, filtered if needed
    if active_only:
        alerts = FloodAlert.objects.filter(active=True).order_by('-severity_level', '-issued_at')
    else:
        alerts = FloodAlert.objects.all().order_by('-issued_at')
    
    # Format alerts for JSON response
    alert_data = []
    for alert in alerts:
        alert_data.append({
            'id': alert.id,
            'title': alert.title,
            'description': alert.description,
            'severity_level': alert.severity_level,
            'active': alert.active,
            'predicted_flood_time': alert.predicted_flood_time,
            'issued_at': alert.issued_at,
            'updated_at': alert.updated_at,
            'issued_by_username': alert.issued_by.username if alert.issued_by else 'System',
            'affected_barangay_count': alert.affected_barangays.count(),
            'affected_barangays': list(alert.affected_barangays.values_list('id', flat=True))
        })
    
    # Return as JSON
    return JsonResponse({
        'count': len(alert_data),
        'results': alert_data
    })

def get_municipality_detail(request, municipality_id):
    """API endpoint to get details for a single municipality."""
    try:
        municipality = get_object_or_404(Municipality, pk=municipality_id)
        data = {
            'id': municipality.id,
            'name': municipality.name,
            'province': municipality.province,
            'population': municipality.population,
            'area_sqkm': municipality.area_sqkm,
            'latitude': municipality.latitude,
            'longitude': municipality.longitude,
            'is_active': municipality.is_active,
            'contact_person': municipality.contact_person,
            'contact_number': municipality.contact_number,
        }
        return JsonResponse(data)
    except Municipality.DoesNotExist:
        return JsonResponse({'error': 'Municipality not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def api_create_municipality(request):
    """API endpoint to create a municipality via AJAX (admin/manager only)."""
    if not is_admin_or_manager(request.user):
        return JsonResponse({'success': False, 'message': 'Permission denied.'}, status=403)

    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'POST required.'}, status=400)

    from .forms import MunicipalityForm

    form = MunicipalityForm(request.POST)
    if form.is_valid():
        # As above, ensure required numeric fields have safe defaults before saving
        m = form.save(commit=False)
        if not m.population:
            m.population = 0
        if not m.area_sqkm:
            m.area_sqkm = 0.0
        if not m.latitude:
            m.latitude = 0.0
        if not m.longitude:
            m.longitude = 0.0
        m.save()
        return JsonResponse({'success': True, 'id': m.id, 'name': m.name, 'message': 'Municipality created.'})
    else:
        # Return form errors
        errors = {k: v for k, v in form.errors.items()}
        return JsonResponse({'success': False, 'errors': errors}, status=400)


@login_required
def api_create_barangay(request):
    """API endpoint to create a barangay via AJAX (admin/manager only)."""
    if not is_admin_or_manager(request.user):
        return JsonResponse({'success': False, 'message': 'Permission denied.'}, status=403)

    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'POST required.'}, status=400)

    from .forms import BarangayForm

    form = BarangayForm(request.POST)
    if form.is_valid():
        # Prevent duplicates for AJAX pathway as well
        name = form.cleaned_data.get('name', '').strip()
        municipality = form.cleaned_data.get('municipality')
        existing = None
        if name:
            qs = Barangay.objects.filter(name__iexact=name)
            if municipality:
                qs = qs.filter(municipality=municipality)
            existing = qs.first()

        if existing:
            return JsonResponse({'success': True, 'id': existing.id, 'name': existing.name, 'message': 'Barangay already exists.'})

        # Ensure required numeric fields have safe defaults before saving
        b = form.save(commit=False)
        if not getattr(b, 'population', None):
            b.population = 0
        if not getattr(b, 'area_sqkm', None):
            b.area_sqkm = 0.0
        if not getattr(b, 'latitude', None):
            b.latitude = 0.0
        if not getattr(b, 'longitude', None):
            b.longitude = 0.0
        b.save()
        return JsonResponse({'success': True, 'id': b.id, 'name': b.name, 'message': 'Barangay created.'})
    else:
        errors = {k: v for k, v in form.errors.items()}
        return JsonResponse({'success': False, 'errors': errors}, status=400)


# This function has been moved up to avoid duplication


# User Management and Profile Views

@login_required
def profile(request):
    """User profile page view"""
    user = request.user
    profile = user.profile
    
    context = {
        'user': user,
        'profile': profile,
        'page': 'profile'
    }
    
    return render(request, 'profile.html', context)

@login_required
def edit_profile(request):
    """Edit user profile view"""
    user = request.user
    
    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=user.profile, user=user)
        if form.is_valid():
            form.save(user=user)
            messages.success(request, 'Your profile has been updated!')
            return redirect('profile')
    else:
        form = UserProfileForm(instance=user.profile, user=user)
    
    context = {
        'form': form,
        'page': 'profile'
    }
    
    return render(request, 'edit_profile.html', context)

# Helper function to check if user is admin or manager
def is_admin_or_manager(user):
    """Check if the user is an admin or flood manager"""
    if user.is_superuser:
        return True
    if hasattr(user, 'profile'):
        return user.profile.role in ['admin', 'manager']
    return False

@login_required
@user_passes_test(is_admin_or_manager)
def user_management(request):
    """User management page for admins and managers"""
    # Get all users
    users = User.objects.all().select_related('profile').order_by('username')
    
    # Handle search and filtering
    search_query = request.GET.get('search', '')
    role_filter = request.GET.get('role', '')
    municipality_filter = request.GET.get('municipality', '')
    
    if search_query:
        users = users.filter(Q(username__icontains=search_query) | 
                            Q(first_name__icontains=search_query) | 
                            Q(last_name__icontains=search_query) | 
                            Q(email__icontains=search_query))
    
    if role_filter:
        users = users.filter(profile__role=role_filter)
    
    if municipality_filter:
        users = users.filter(profile__municipality_id=municipality_filter)
    
    # Pagination
    paginator = Paginator(users, 10)  # 10 users per page
    page_number = request.GET.get('page', 1)
    users_page = paginator.get_page(page_number)
    
    # Get all municipalities for the filter
    municipalities = Municipality.objects.filter(is_active=True).order_by('name')
    
    context = {
        'users': users_page,
        'search_query': search_query,
        'role_filter': role_filter,
        'municipality_filter': municipality_filter,
        'municipalities': municipalities,
        'user_roles': UserProfile.USER_ROLES,
        'page': 'user_management'
    }
    
    return render(request, 'user_management.html', context)

@login_required
@user_passes_test(is_admin_or_manager)
def view_user(request, user_id):
    """View details of a specific user"""
    viewed_user = get_object_or_404(User.objects.select_related('profile'), id=user_id)
    
    context = {
        'viewed_user': viewed_user,
        'page': 'user_management'
    }
    
    return render(request, 'view_user.html', context)

@login_required
@user_passes_test(is_admin_or_manager)
def edit_user(request, user_id):
    """Edit a specific user (admin only)"""
    viewed_user = get_object_or_404(User.objects.select_related('profile'), id=user_id)
    
    # Only superuser can edit other superusers
    if viewed_user.is_superuser and not request.user.is_superuser:
        messages.error(request, 'You do not have permission to edit this user.')
        return redirect('user_management')
    
    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=viewed_user.profile, user=viewed_user)
        if form.is_valid():
            form.save(user=viewed_user)
            messages.success(request, f'User {viewed_user.username} has been updated!')
            return redirect('view_user', user_id=user_id)
    else:
        form = UserProfileForm(instance=viewed_user.profile, user=viewed_user)
    
    context = {
        'form': form,
        'viewed_user': viewed_user,
        'page': 'user_management'
    }
    
    return render(request, 'edit_user.html', context)

@login_required
def resilience_scores_page(request):
    """Community resilience scores view"""
    # Get all resilience scores, prioritizing current scores
    resilience_scores = ResilienceScore.objects.all().order_by('-is_current', '-assessment_date')
    
    # Get location filters from request
    municipality_id = request.GET.get('municipality_id', None)
    barangay_id = request.GET.get('barangay_id', None)
    show_only_current = request.GET.get('current', 'true').lower() == 'true'
    
    # Filter by municipality if specified
    if municipality_id:
        resilience_scores = resilience_scores.filter(municipality_id=municipality_id)
    
    # Filter by barangay if specified
    if barangay_id:
        resilience_scores = resilience_scores.filter(barangay_id=barangay_id)
    
    # Filter to show only current assessments if specified
    if show_only_current:
        resilience_scores = resilience_scores.filter(is_current=True)
    
    # Get all municipalities for filtering
    municipalities = Municipality.objects.all().order_by('name')
    
    # Get barangays belonging to the selected municipality or all if no municipality is selected
    barangay_filter = {}
    if municipality_id:
        barangay_filter['municipality_id'] = municipality_id
    
    barangays = Barangay.objects.filter(**barangay_filter).order_by('name')
    
    # Calculate summary statistics
    avg_scores = {}
    if resilience_scores.exists():
        # Overall average
        avg_scores['overall'] = resilience_scores.aggregate(Avg('overall_score'))['overall_score__avg']
        
        # By component
        avg_scores['infrastructure'] = resilience_scores.aggregate(Avg('infrastructure_score'))['infrastructure_score__avg']
        avg_scores['social'] = resilience_scores.aggregate(Avg('social_capital_score'))['social_capital_score__avg']
        avg_scores['institutional'] = resilience_scores.aggregate(Avg('institutional_score'))['institutional_score__avg']
        avg_scores['economic'] = resilience_scores.aggregate(Avg('economic_score'))['economic_score__avg']
        avg_scores['environmental'] = resilience_scores.aggregate(Avg('environmental_score'))['environmental_score__avg']
    
    context = {
        'resilience_scores': resilience_scores,
        'municipalities': municipalities,
        'barangays': barangays,
        'avg_scores': avg_scores,
        'selected_municipality_id': municipality_id,
        'selected_barangay_id': barangay_id,
        'show_only_current': show_only_current,
        'page': 'resilience'
    }
    
    return render(request, 'resilience_scores.html', context)

def get_flood_history(request):
    """API endpoint to get historical flood alerts for a given location."""
    municipality_id = request.GET.get('municipality_id')
    barangay_id = request.GET.get('barangay_id')
    search_query = request.GET.get('search', '').strip()
    severity_filter = request.GET.get('severity', '').strip()

    alerts_qs = FloodAlert.objects.all().order_by('-issued_at')

    if barangay_id:
        alerts_qs = alerts_qs.filter(affected_barangays__id=barangay_id)
    elif municipality_id:
        alerts_qs = alerts_qs.filter(affected_barangays__municipality_id=municipality_id)

    # Apply search filter
    if search_query:
        alerts_qs = alerts_qs.filter(Q(title__icontains=search_query) | Q(description__icontains=search_query))

    # Apply severity filter
    if severity_filter and severity_filter.isdigit():
        alerts_qs = alerts_qs.filter(severity_level=int(severity_filter))

    # Paginate the results
    paginator = Paginator(alerts_qs.distinct(), 10) # 10 history items per page
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    history_data = []
    for alert in page_obj:
        history_data.append({
            'id': alert.id,
            'title': alert.title,
            'description': alert.description,
            'severity_level': alert.severity_level,
            'severity_display': alert.get_severity_level_display(),
            'issued_at': _manila_str(alert.issued_at, include_seconds=False),
            'is_active': alert.active,
        })

    return JsonResponse({
        'history': history_data,
        'pagination': {
            'has_next': page_obj.has_next(),
            'has_previous': page_obj.has_previous(),
            'next_page_number': page_obj.next_page_number() if page_obj.has_next() else None,
            'previous_page_number': page_obj.previous_page_number() if page_obj.has_previous() else None,
            'current_page': page_obj.number,
            'total_pages': paginator.num_pages,
        }
    })
