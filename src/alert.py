"""
src/alert.py
Alert system for unknown face detections.
Logs to DB and optionally sends email/webhook notifications.
"""

import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime


class AlertSystem:
    def __init__(self, db):
        self.db = db
        self.smtp_host = os.getenv('ALERT_SMTP_HOST', '')
        self.smtp_user = os.getenv('ALERT_SMTP_USER', '')
        self.smtp_pass = os.getenv('ALERT_SMTP_PASS', '')
        self.alert_to  = os.getenv('ALERT_EMAIL_TO', '')

    def trigger(self, image_path: str, embedding: list):
        """Called when an unknown face is detected."""
        self.db.log_alert(image_path, embedding)
        if self.smtp_host and self.alert_to:
            self._send_email(image_path)

    def _send_email(self, image_path: str):
        try:
            msg = MIMEText(
                f"VisionGuard Alert\n\n"
                f"Unknown face detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Image saved: {image_path}"
            )
            msg['Subject'] = 'VisionGuard: Unknown Face Detected'
            msg['From']    = self.smtp_user
            msg['To']      = self.alert_to
            with smtplib.SMTP_SSL(self.smtp_host, 465) as s:
                s.login(self.smtp_user, self.smtp_pass)
                s.send_message(msg)
        except Exception as e:
            print(f"[AlertSystem] Email failed: {e}")
