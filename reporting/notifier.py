#!/usr/bin/env python3
"""
HTML Email Notifier for Arsenal Intelligence Brief

This module handles sending HTML email notifications with support for:
- SMTP delivery (Gmail and other providers)
- Inline images (base64-encoded charts)
- File attachments
- Multipart messages (HTML + plain text fallback)

Environment Variables:
    SMTP_HOST - SMTP server hostname (default: smtp.gmail.com)
    SMTP_PORT - SMTP server port (default: 587)
    SMTP_USER - SMTP username/email
    SMTP_PASSWORD - SMTP password or app-specific password
    EMAIL_RECIPIENTS - Comma-separated list of recipient emails

Task: arsenalScript-vqp.42 - Implement HTML email sender
"""

import base64
import logging
import os
import re
import smtplib
import ssl
from dataclasses import dataclass, field
from email import encoders
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class EmailConfig:
    """Configuration for email delivery."""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    use_tls: bool = True
    use_ssl: bool = False
    timeout: int = 30

    # Sender info
    sender_name: str = "Arsenal Intelligence Brief"
    sender_email: Optional[str] = None  # Defaults to smtp_user if not set

    # Recipient info
    recipients: List[str] = field(default_factory=list)
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> 'EmailConfig':
        """Create EmailConfig from environment variables."""
        recipients = []
        env_recipients = os.environ.get("EMAIL_RECIPIENTS", "")
        if env_recipients:
            recipients = [e.strip() for e in env_recipients.split(",") if e.strip()]

        return cls(
            smtp_host=os.environ.get("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(os.environ.get("SMTP_PORT", "587")),
            smtp_user=os.environ.get("SMTP_USER"),
            smtp_password=os.environ.get("SMTP_PASSWORD"),
            use_tls=os.environ.get("SMTP_USE_TLS", "true").lower() == "true",
            use_ssl=os.environ.get("SMTP_USE_SSL", "false").lower() == "true",
            sender_name=os.environ.get("SMTP_SENDER_NAME", "Arsenal Intelligence Brief"),
            sender_email=os.environ.get("SMTP_SENDER_EMAIL"),
            recipients=recipients,
        )

    def is_valid(self) -> Tuple[bool, str]:
        """Check if configuration is valid for sending emails."""
        if not self.smtp_user:
            return False, "SMTP_USER not configured"
        if not self.smtp_password:
            return False, "SMTP_PASSWORD not configured"
        if not self.recipients:
            return False, "No recipients configured"
        return True, "Configuration valid"


@dataclass
class EmailAttachment:
    """Represents an email attachment."""
    filename: str
    content: bytes
    content_type: str = "application/octet-stream"
    content_id: Optional[str] = None  # For inline images (CID)

    @classmethod
    def from_file(cls, filepath: Union[str, Path], content_type: Optional[str] = None) -> 'EmailAttachment':
        """Create attachment from a file path."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Attachment file not found: {filepath}")

        with open(path, 'rb') as f:
            content = f.read()

        # Guess content type from extension if not provided
        if content_type is None:
            ext = path.suffix.lower()
            content_types = {
                '.pdf': 'application/pdf',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.html': 'text/html',
                '.txt': 'text/plain',
                '.json': 'application/json',
                '.csv': 'text/csv',
            }
            content_type = content_types.get(ext, 'application/octet-stream')

        return cls(
            filename=path.name,
            content=content,
            content_type=content_type,
        )

    @classmethod
    def from_base64(
        cls,
        base64_data: str,
        filename: str,
        content_type: str = "image/png"
    ) -> 'EmailAttachment':
        """Create attachment from base64-encoded data."""
        content = base64.b64decode(base64_data)
        return cls(
            filename=filename,
            content=content,
            content_type=content_type,
        )


@dataclass
class EmailMessage:
    """Represents an email message to be sent."""
    subject: str
    html_content: str
    plain_content: Optional[str] = None  # Fallback for non-HTML clients
    attachments: List[EmailAttachment] = field(default_factory=list)
    inline_images: Dict[str, EmailAttachment] = field(default_factory=dict)  # CID -> attachment

    # Override recipients for this specific message
    recipients: Optional[List[str]] = None
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None
    reply_to: Optional[str] = None

    def add_attachment(self, attachment: EmailAttachment) -> None:
        """Add an attachment to the message."""
        self.attachments.append(attachment)

    def add_attachment_from_file(self, filepath: Union[str, Path]) -> None:
        """Add an attachment from a file path."""
        self.attachments.append(EmailAttachment.from_file(filepath))

    def add_inline_image(self, content_id: str, attachment: EmailAttachment) -> None:
        """Add an inline image with a content ID for embedding in HTML."""
        attachment.content_id = content_id
        self.inline_images[content_id] = attachment

    def add_inline_image_from_base64(
        self,
        content_id: str,
        base64_data: str,
        filename: str = "image.png",
        content_type: str = "image/png"
    ) -> None:
        """Add an inline image from base64 data."""
        attachment = EmailAttachment.from_base64(base64_data, filename, content_type)
        self.add_inline_image(content_id, attachment)


class HTMLEmailSender:
    """
    Sends HTML emails via SMTP with support for attachments and inline images.

    Usage:
        config = EmailConfig.from_env()
        sender = HTMLEmailSender(config)

        # Create message
        message = EmailMessage(
            subject="Arsenal Intelligence Brief - Chelsea",
            html_content=html_report,
            plain_content="View this email in an HTML-capable client"
        )

        # Add attachment
        message.add_attachment_from_file("report.pdf")

        # Send
        success, error = sender.send(message)
    """

    def __init__(self, config: Optional[EmailConfig] = None):
        """
        Initialize the email sender.

        Args:
            config: EmailConfig instance. If None, loads from environment.
        """
        self.config = config or EmailConfig.from_env()
        self.logger = logging.getLogger(f"{__name__}.HTMLEmailSender")

    def _create_mime_message(self, message: EmailMessage) -> MIMEMultipart:
        """Create a MIME message from an EmailMessage."""
        # Determine recipients
        recipients = message.recipients or self.config.recipients
        cc = message.cc or self.config.cc
        bcc = message.bcc or self.config.bcc

        # Create the root message
        if message.inline_images:
            # Use 'related' for inline images + 'alternative' for text/html
            root = MIMEMultipart('related')
            alt = MIMEMultipart('alternative')
            root.attach(alt)
        else:
            # Simple alternative for text/html
            root = MIMEMultipart('alternative')
            alt = root

        # Set headers
        sender_email = self.config.sender_email or self.config.smtp_user
        root['Subject'] = message.subject
        root['From'] = f"{self.config.sender_name} <{sender_email}>"
        root['To'] = ', '.join(recipients)

        if cc:
            root['Cc'] = ', '.join(cc)

        if message.reply_to:
            root['Reply-To'] = message.reply_to

        # Add plain text version (fallback)
        plain_content = message.plain_content
        if not plain_content:
            # Generate plain text from HTML by stripping tags
            plain_content = self._html_to_plain(message.html_content)

        alt.attach(MIMEText(plain_content, 'plain', 'utf-8'))

        # Add HTML version
        alt.attach(MIMEText(message.html_content, 'html', 'utf-8'))

        # Add inline images
        for cid, attachment in message.inline_images.items():
            # Extract subtype from content_type (e.g., 'image/png' -> 'png')
            subtype = attachment.content_type.split('/')[-1] if '/' in attachment.content_type else 'png'
            img = MIMEImage(attachment.content, _subtype=subtype)
            img.add_header('Content-ID', f'<{cid}>')
            img.add_header('Content-Disposition', 'inline', filename=attachment.filename)
            root.attach(img)

        # Add regular attachments
        for attachment in message.attachments:
            if attachment.content_type.startswith('image/'):
                subtype = attachment.content_type.split('/')[-1]
                part = MIMEImage(attachment.content, _subtype=subtype)
            elif attachment.content_type.startswith('text/'):
                part = MIMEText(attachment.content.decode('utf-8', errors='replace'))
            else:
                part = MIMEBase(*attachment.content_type.split('/', 1))
                part.set_payload(attachment.content)
                encoders.encode_base64(part)

            part.add_header(
                'Content-Disposition',
                'attachment',
                filename=attachment.filename
            )
            root.attach(part)

        return root

    def _html_to_plain(self, html: str) -> str:
        """Convert HTML to plain text for fallback."""
        # Remove style and script blocks
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)

        # Replace common block elements with newlines
        html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</?p[^>]*>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</?div[^>]*>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</?h[1-6][^>]*>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</?tr[^>]*>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</?li[^>]*>', '\n- ', html, flags=re.IGNORECASE)

        # Remove remaining HTML tags
        html = re.sub(r'<[^>]+>', '', html)

        # Decode HTML entities
        html = html.replace('&nbsp;', ' ')
        html = html.replace('&amp;', '&')
        html = html.replace('&lt;', '<')
        html = html.replace('&gt;', '>')
        html = html.replace('&quot;', '"')

        # Clean up whitespace
        lines = [line.strip() for line in html.split('\n')]
        lines = [line for line in lines if line]

        return '\n\n'.join(lines)

    def _connect(self) -> smtplib.SMTP:
        """Establish SMTP connection."""
        if self.config.use_ssl:
            # SSL connection on port 465
            context = ssl.create_default_context()
            server = smtplib.SMTP_SSL(
                self.config.smtp_host,
                self.config.smtp_port,
                context=context,
                timeout=self.config.timeout
            )
        else:
            # Standard connection with STARTTLS
            server = smtplib.SMTP(
                self.config.smtp_host,
                self.config.smtp_port,
                timeout=self.config.timeout
            )
            if self.config.use_tls:
                server.starttls()

        # Login
        server.login(self.config.smtp_user, self.config.smtp_password)

        return server

    def send(self, message: EmailMessage) -> Tuple[bool, Optional[str]]:
        """
        Send an email message.

        Args:
            message: EmailMessage to send

        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        # Validate configuration
        is_valid, error = self.config.is_valid()
        if not is_valid:
            self.logger.error(f"Invalid email configuration: {error}")
            return False, error

        # Determine all recipients
        recipients = message.recipients or self.config.recipients
        cc = message.cc or self.config.cc
        bcc = message.bcc or self.config.bcc
        all_recipients = list(set(recipients + cc + bcc))

        if not all_recipients:
            error = "No recipients specified"
            self.logger.error(error)
            return False, error

        try:
            # Create MIME message
            mime_message = self._create_mime_message(message)

            # Connect and send
            self.logger.info(f"Connecting to {self.config.smtp_host}:{self.config.smtp_port}")

            with self._connect() as server:
                sender_email = self.config.sender_email or self.config.smtp_user
                self.logger.info(f"Sending email to {len(all_recipients)} recipient(s)")
                server.sendmail(sender_email, all_recipients, mime_message.as_string())

            self.logger.info(f"Email sent successfully: {message.subject}")
            return True, None

        except smtplib.SMTPAuthenticationError as e:
            error = f"SMTP authentication failed: {e}"
            self.logger.error(error)
            return False, error

        except smtplib.SMTPRecipientsRefused as e:
            error = f"All recipients were refused: {e}"
            self.logger.error(error)
            return False, error

        except smtplib.SMTPSenderRefused as e:
            error = f"Sender address refused: {e}"
            self.logger.error(error)
            return False, error

        except smtplib.SMTPDataError as e:
            error = f"SMTP data error (message may be too large): {e}"
            self.logger.error(error)
            return False, error

        except smtplib.SMTPException as e:
            error = f"SMTP error: {e}"
            self.logger.error(error)
            return False, error

        except TimeoutError as e:
            error = f"Connection timeout: {e}"
            self.logger.error(error)
            return False, error

        except Exception as e:
            error = f"Unexpected error sending email: {e}"
            self.logger.exception(error)
            return False, error

    def send_report(
        self,
        html_content: str,
        subject: str,
        match_id: str,
        opponent: str,
        match_date: str,
        attachments: Optional[List[Union[str, Path, EmailAttachment]]] = None,
        recipients: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Send an intelligence brief report email.

        Convenience method for sending Arsenal Intelligence Brief reports.

        Args:
            html_content: The HTML report content
            subject: Email subject line
            match_id: Match identifier
            opponent: Opposing team name
            match_date: Match date string
            attachments: Optional list of attachment file paths or EmailAttachment objects
            recipients: Optional override for recipients

        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        # Create plain text fallback
        plain_content = f"""
Arsenal Intelligence Brief

Match: Arsenal vs {opponent}
Date: {match_date}
Match ID: {match_id}

This email contains an HTML report. Please view in an HTML-capable email client
for the best experience.

If you cannot view HTML emails, please contact us for a plain text version.

---
Arsenal Intelligence Brief
Powered by Arsenal Pulse
        """

        # Create message
        message = EmailMessage(
            subject=subject,
            html_content=html_content,
            plain_content=plain_content,
            recipients=recipients,
        )

        # Add attachments
        if attachments:
            for att in attachments:
                if isinstance(att, EmailAttachment):
                    message.add_attachment(att)
                else:
                    message.add_attachment_from_file(att)

        return self.send(message)

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the SMTP connection without sending an email.

        Returns:
            Tuple of (success: bool, message: str)
        """
        is_valid, error = self.config.is_valid()
        if not is_valid:
            return False, f"Invalid configuration: {error}"

        try:
            with self._connect() as server:
                # NOOP command to verify connection
                server.noop()

            return True, f"Successfully connected to {self.config.smtp_host}:{self.config.smtp_port}"

        except smtplib.SMTPAuthenticationError as e:
            return False, f"Authentication failed: {e}"

        except smtplib.SMTPException as e:
            return False, f"SMTP error: {e}"

        except Exception as e:
            return False, f"Connection failed: {e}"


class IntelligenceBriefNotifier:
    """
    High-level notifier specifically for Arsenal Intelligence Briefs.

    Integrates with the ReportBuilder to generate and send reports.

    Usage:
        from reporting.report_builder import ReportBuilder, IntelligenceBrief
        from reporting.notifier import IntelligenceBriefNotifier

        # Build report
        builder = ReportBuilder()
        report = builder.build_report(...)

        # Send notification
        notifier = IntelligenceBriefNotifier()
        success = notifier.send_report(report)
    """

    def __init__(self, config: Optional[EmailConfig] = None):
        """Initialize the notifier with optional email configuration."""
        self.config = config or EmailConfig.from_env()
        self.sender = HTMLEmailSender(self.config)
        self.logger = logging.getLogger(f"{__name__}.IntelligenceBriefNotifier")

    def send_report(
        self,
        report: Any,  # IntelligenceBrief from report_builder
        recipients: Optional[List[str]] = None,
        include_charts_as_attachments: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Send an IntelligenceBrief report via email.

        Args:
            report: IntelligenceBrief object from ReportBuilder
            recipients: Optional list of email recipients (overrides config)
            include_charts_as_attachments: Whether to attach charts as separate files

        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        try:
            # Generate HTML content
            html_content = report.to_html()

            # Build subject line
            subject = f"Arsenal Intelligence Brief - {report.home_team} vs {report.away_team}"
            if report.match_date:
                subject += f" ({report.match_date[:10]})"

            # Create plain text fallback
            plain_content = f"""
Arsenal Intelligence Brief

Match: {report.home_team} vs {report.away_team}
Competition: {report.competition}
Date: {report.match_date or 'TBD'}
Match ID: {report.match_id}

Data Completeness: {report.data_completeness.completeness_score:.0f}%

This email contains an HTML report with charts and detailed analysis.
Please view in an HTML-capable email client for the best experience.

Generated: {report.generated_at}

---
Arsenal Intelligence Brief
Powered by Arsenal Pulse
            """

            # Create message
            message = EmailMessage(
                subject=subject,
                html_content=html_content,
                plain_content=plain_content,
                recipients=recipients,
            )

            # Optionally attach charts as separate files
            if include_charts_as_attachments and report.charts:
                for chart_name, base64_data in report.charts.items():
                    attachment = EmailAttachment.from_base64(
                        base64_data,
                        filename=f"{chart_name}.png",
                        content_type="image/png"
                    )
                    message.add_attachment(attachment)

            # Send
            success, error = self.sender.send(message)

            if success:
                self.logger.info(f"Report sent for match {report.match_id}")
            else:
                self.logger.error(f"Failed to send report: {error}")

            return success, error

        except Exception as e:
            error = f"Failed to send report: {e}"
            self.logger.exception(error)
            return False, error

    def send_html_file(
        self,
        html_path: Union[str, Path],
        subject: str,
        recipients: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Send an HTML file as an email.

        Args:
            html_path: Path to the HTML file
            subject: Email subject line
            recipients: Optional list of email recipients

        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        path = Path(html_path)
        if not path.exists():
            return False, f"HTML file not found: {html_path}"

        with open(path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        message = EmailMessage(
            subject=subject,
            html_content=html_content,
            recipients=recipients,
        )

        return self.sender.send(message)


# ==========================================================================
# DEMO / TEST
# ==========================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Arsenal Intelligence Brief - Email Notifier Demo")
    print("=" * 60)

    # Load configuration from environment
    config = EmailConfig.from_env()

    print("\nConfiguration:")
    print(f"  SMTP Host: {config.smtp_host}")
    print(f"  SMTP Port: {config.smtp_port}")
    print(f"  SMTP User: {'[SET]' if config.smtp_user else '[NOT SET]'}")
    print(f"  SMTP Password: {'[SET]' if config.smtp_password else '[NOT SET]'}")
    print(f"  Recipients: {config.recipients if config.recipients else '[NOT SET]'}")

    # Check configuration validity
    is_valid, message = config.is_valid()
    print(f"\nConfiguration Valid: {is_valid}")
    if not is_valid:
        print(f"  Error: {message}")
        print("\nTo test email sending, set the following environment variables:")
        print("  SMTP_USER=your_email@gmail.com")
        print("  SMTP_PASSWORD=your_app_password")
        print("  EMAIL_RECIPIENTS=recipient@example.com")
        sys.exit(1)

    # Test connection
    sender = HTMLEmailSender(config)
    print("\n--- Testing SMTP Connection ---")
    success, msg = sender.test_connection()
    print(f"Connection Test: {'SUCCESS' if success else 'FAILED'}")
    print(f"  {msg}")

    if not success:
        sys.exit(1)

    # Optionally send a test email
    if len(sys.argv) > 1 and sys.argv[1] == "--send-test":
        print("\n--- Sending Test Email ---")

        test_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; }
                .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
                h1 { color: #EF0107; }
                .footer { margin-top: 20px; padding-top: 20px; border-top: 1px solid #eee; color: #666; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Arsenal Intelligence Brief Test</h1>
                <p>This is a test email from the Arsenal Intelligence Brief notification system.</p>
                <p>If you're receiving this email, the SMTP configuration is working correctly.</p>
                <div class="footer">
                    <p>Arsenal Intelligence Brief | Test Email</p>
                </div>
            </div>
        </body>
        </html>
        """

        message = EmailMessage(
            subject="[TEST] Arsenal Intelligence Brief - Email System Test",
            html_content=test_html,
        )

        success, error = sender.send(message)

        if success:
            print("Test email sent successfully!")
        else:
            print(f"Failed to send test email: {error}")
    else:
        print("\nTo send a test email, run: python notifier.py --send-test")

    print("\n" + "=" * 60)
    print("Demo complete.")
