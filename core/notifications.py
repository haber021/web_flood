import logging
from typing import List
from django.db.models import Q
from .models import FloodAlert, EmergencyContact, NotificationLog, UserProfile
from twilio.rest import Client
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def send_sms(to_number, message_body):
    """
    Send an SMS using Twilio

    Args:
        to_number (str): The recipient's phone number in E.164 format (e.g., "+1234567890")
        message_body (str): The message to send

    Returns:
        str: Message SID if successful, None otherwise
    """
    try:
        # Load environment variables from .env file in project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(project_root, '.env')
        load_dotenv(env_path)

        # Your Account SID and Auth Token from Twilio console
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        twilio_number = os.getenv('TWILIO_PHONE_NUMBER')

        if not all([account_sid, auth_token, twilio_number]):
            logger.error(f"Error: Missing Twilio credentials in environment variables. Looking for .env at: {env_path}")
            logger.error(f"TWILIO_ACCOUNT_SID: {account_sid is not None}, TWILIO_AUTH_TOKEN: {auth_token is not None}, TWILIO_PHONE_NUMBER: {twilio_number is not None}")
            return None

        # Check if SMS sending is enabled (for trial accounts or testing)
        sms_enabled = os.getenv('SMS_ENABLED', 'true').lower() == 'true'
        if not sms_enabled:
            logger.info(f"SMS sending is disabled. Would send to {to_number}: {message_body[:50]}...")
            return 'DISABLED'

        # Initialize the Twilio client
        client = Client(account_sid, auth_token)

        # Send the message
        message = client.messages.create(
            body=message_body,
            from_=twilio_number,
            to=to_number
        )

        logger.info(f"Message sent successfully! SID: {message.sid}")
        return message.sid

    except Exception as e:
        error_str = str(e)
        # Handle specific Twilio errors
        if '63038' in error_str or 'exceeded the' in error_str.lower():
            logger.warning(f"Twilio rate limit exceeded. SMS not sent to {to_number}. Error: {error_str}")
        elif '400' in error_str or 'Invalid' in error_str:
            logger.error(f"Invalid phone number or Twilio configuration error for {to_number}: {error_str}")
        else:
            logger.error(f"Error sending SMS to {to_number}: {error_str}")
        return None


def dispatch_notifications_for_alert(alert: FloodAlert):
    """
    Finds relevant contacts for a given FloodAlert and creates NotificationLog entries.
    This simulates sending SMS and email notifications.

    Args:
        alert (FloodAlert): The alert for which to send notifications.
    """
    if not alert.active:
        logger.info(f"Alert {alert.id} is not active. Skipping notification dispatch.")
        return

    # Get all barangays affected by the alert
    affected_barangays = alert.affected_barangays.all()
    if not affected_barangays.exists():
        logger.warning(f"Alert {alert.id} has no affected barangays. No notifications sent.")
        return

    # --- Find Recipients ---
    # 1. Barangay contact numbers for the affected barangays
    barangay_contacts = []
    for barangay in affected_barangays:
        if barangay.contact_number:
            barangay_contacts.append({
                'name': barangay.contact_person or f'Barangay {barangay.name}',
                'phone': barangay.contact_number,
                'barangay': barangay.name
            })

    # 2. UserProfile entries for users assigned to the affected locations
    # Find users assigned to the specific barangays or their parent municipalities
    barangay_ids = [b.id for b in affected_barangays]
    municipality_ids = [b.municipality_id for b in affected_barangays if b.municipality_id]

    users = UserProfile.objects.filter(
        Q(barangay_id__in=barangay_ids) | Q(municipality_id__in=municipality_ids)
    ).select_related('user')

    # --- Prepare and "Send" Notifications ---
    message_body = f"Flood Alert: {alert.get_severity_level_display()} - {alert.title}. Description: {alert.description}"
    recipients_notified = set()

    # Notify Barangay Contacts
    for contact in barangay_contacts:
        if contact['phone'] and contact['phone'] not in recipients_notified:
            # Actually send SMS via Twilio
            sms_sid = send_sms(contact['phone'], message_body)
            if sms_sid == 'DISABLED':
                # SMS sending is disabled
                NotificationLog.objects.create(
                    alert=alert, notification_type='sms', recipient=contact['phone'], status='pending'
                )
                recipients_notified.add(contact['phone'])
                logger.info(f"SMS sending disabled. Would send to barangay contact {contact['name']} ({contact['barangay']}) at {contact['phone']} for alert {alert.id}")
            elif sms_sid:
                NotificationLog.objects.create(
                    alert=alert, notification_type='sms', recipient=contact['phone'], status='sent'
                )
                recipients_notified.add(contact['phone'])
                logger.info(f"SMS sent to barangay contact {contact['name']} ({contact['barangay']}) at {contact['phone']} for alert {alert.id}, SID: {sms_sid}")
            else:
                NotificationLog.objects.create(
                    alert=alert, notification_type='sms', recipient=contact['phone'], status='failed'
                )
                logger.error(f"Failed to send SMS to barangay contact {contact['name']} ({contact['barangay']}) at {contact['phone']} for alert {alert.id}")

    # Notify UserProfiles
    for profile in users:
        # Notify via Email
        if profile.receive_email and profile.user.email and profile.user.email not in recipients_notified:
            NotificationLog.objects.create(
                alert=alert, notification_type='email', recipient=profile.user.email, status='sent'
            )
            recipients_notified.add(profile.user.email)
            logger.info(f"Simulated Email sent to user {profile.user.username} at {profile.user.email} for alert {alert.id}")
        # Notify via SMS
        if profile.receive_sms and profile.phone_number and profile.phone_number not in recipients_notified:
            sms_sid = send_sms(profile.phone_number, message_body)
            if sms_sid == 'DISABLED':
                # SMS sending is disabled
                NotificationLog.objects.create(
                    alert=alert, notification_type='sms', recipient=profile.phone_number, status='pending'
                )
                recipients_notified.add(profile.phone_number)
                logger.info(f"SMS sending disabled. Would send to user {profile.user.username} at {profile.phone_number} for alert {alert.id}")
            elif sms_sid:
                NotificationLog.objects.create(
                    alert=alert, notification_type='sms', recipient=profile.phone_number, status='sent'
                )
                recipients_notified.add(profile.phone_number)
                logger.info(f"SMS sent to user {profile.user.username} at {profile.phone_number} for alert {alert.id}, SID: {sms_sid}")
            else:
                NotificationLog.objects.create(
                    alert=alert, notification_type='sms', recipient=profile.phone_number, status='failed'
                )
                logger.error(f"Failed to send SMS to user {profile.user.username} at {profile.phone_number} for alert {alert.id}")

    logger.info(f"Dispatched notifications for alert {alert.id} to {len(recipients_notified)} unique barangay contacts.")