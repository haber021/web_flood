#!/bin/bash

# Setup automatic barangay alerts cron job
# This script sets up a cron job to run automatic alerts every 3 hours

# Path to your Django project
PROJECT_PATH="/path/to/your/flood_monitoring_project"
VENV_PATH="/path/to/your/virtualenv"

# Cron job command
CRON_COMMAND="0 */3 * * * cd $PROJECT_PATH && $VENV_PATH/bin/python manage.py send_automatic_barangay_alerts"

# Add to crontab (this will overwrite existing crontab, be careful!)
echo "Setting up cron job for automatic barangay alerts..."
echo "Command: $CRON_COMMAND"
echo ""

# Check if cron job already exists
if crontab -l | grep -q "send_automatic_barangay_alerts"; then
    echo "Cron job already exists. Updating..."
    # Remove existing cron job
    crontab -l | grep -v "send_automatic_barangay_alerts" | crontab -
fi

# Add new cron job
(crontab -l ; echo "$CRON_COMMAND") | crontab -

echo "Cron job setup complete!"
echo "The system will now automatically check for and send barangay alerts every 3 hours."
echo ""
echo "To view current cron jobs: crontab -l"
echo "To edit cron jobs manually: crontab -e"
echo "To remove this cron job: crontab -l | grep -v send_automatic_barangay_alerts | crontab -"