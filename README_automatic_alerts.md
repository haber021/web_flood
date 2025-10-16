# Automatic Barangay Alert System

This system automatically sends flood alerts for barangays every 3 hours based on sensor data readings.

## Features

- **Automatic Alert Generation**: Monitors sensor data and creates alerts when thresholds are exceeded
- **3-Hour Intervals**: Ensures alerts are not sent too frequently (minimum 3 hours between alerts per barangay)
- **Configurable**: Each barangay can have automatic alerts enabled/disabled
- **Threshold-Based**: Uses existing threshold settings to determine alert severity levels
- **SMS/Email Notifications**: Integrates with existing notification system

## Setup

### 1. Database Migration
The system adds two new fields to the Barangay model:
- `auto_alert_enabled`: Boolean field to enable/disable automatic alerts (default: True)
- `last_auto_alert_sent`: Timestamp of the last automatic alert sent

Run migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```

### 2. Management Command
Use the `send_automatic_barangay_alerts` management command:

```bash
# Run the command manually
python manage.py send_automatic_barangay_alerts

# Dry run (shows what would be done without sending alerts)
python manage.py send_automatic_barangay_alerts --dry-run
```

### 3. Cron Job Setup
Set up a cron job to run the command every 3 hours:

```bash
# Edit crontab
crontab -e

# Add this line (adjust paths as needed):
0 */3 * * * cd /path/to/your/project && /path/to/venv/bin/python manage.py send_automatic_barangay_alerts
```

Or use the provided setup script:
```bash
# Edit the script with correct paths first
chmod +x setup_automatic_alerts_cron.sh
./setup_automatic_alerts_cron.sh
```

## How It Works

1. **Filtering**: Only barangays with `auto_alert_enabled=True` are considered
2. **Timing Check**: Skips barangays that received an alert within the last 3 hours
3. **Sensor Analysis**: Checks sensor data from the last 3 hours for the barangay
4. **Threshold Evaluation**: Determines if any sensor readings exceed configured thresholds
5. **Alert Creation**: Creates alerts with appropriate severity levels (Watch, Warning, Emergency, etc.)
6. **Notification**: Sends SMS/email notifications to emergency contacts
7. **Tracking**: Updates `last_auto_alert_sent` timestamp to prevent spam

## Alert Severity Levels

- **Advisory (1)**: Readings approach advisory thresholds
- **Watch (2)**: Readings exceed watch thresholds
- **Warning (3)**: Readings exceed warning thresholds
- **Emergency (4)**: Readings exceed emergency thresholds
- **Catastrophic (5)**: Readings exceed catastrophic thresholds

## Configuration

### Enable/Disable for Specific Barangays
In Django admin, edit barangay records to:
- Set `auto_alert_enabled` to control automatic alerts
- View `last_auto_alert_sent` to see when the last alert was sent

### Threshold Configuration
Configure alert thresholds in the admin panel under "Threshold settings" to control when alerts are triggered.

## Testing

Use the `--dry-run` option to test without sending actual notifications:

```bash
python manage.py send_automatic_barangay_alerts --dry-run
```

This will show which barangays would receive alerts without actually creating them or sending notifications.

## Troubleshooting

### No Alerts Being Sent
1. Check that `auto_alert_enabled` is True for barangays
2. Verify threshold settings are configured
3. Ensure sensor data exists for the barangay
4. Check that 3+ hours have passed since last alert

### Cron Job Issues
1. Verify cron service is running: `sudo service cron status`
2. Check cron logs: `grep CRON /var/log/syslog`
3. Test the command manually first
4. Ensure correct paths in crontab

### Permission Issues
Make sure the user running the cron job has:
- Access to the project directory
- Access to the virtual environment
- Database access permissions