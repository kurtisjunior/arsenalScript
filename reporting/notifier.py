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


class EmailHTMLOptimizer:
    """
    Optimizes HTML content for email client compatibility.

    Email clients have varying levels of CSS support. This class processes HTML to:
    - Inline critical CSS styles directly on elements
    - Add fallback text for images
    - Ensure tables render properly across clients
    - Add explicit dimensions to images
    - Convert CSS-based styling to inline attributes

    Task: arsenalScript-vqp.44 - Add inline charts and tables to notifications

    Usage:
        optimizer = EmailHTMLOptimizer()
        optimized_html = optimizer.optimize(html_content)
    """

    # Arsenal color scheme for inline styles
    COLORS = {
        'red': '#EF0107',
        'navy': '#063672',
        'gold': '#9C824A',
        'white': '#FFFFFF',
        'light_gray': '#f5f5f5',
        'dark_gray': '#333333',
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
    }

    # Common inline styles for email clients
    TABLE_STYLES = (
        'border-collapse: collapse; '
        'width: 100%; '
        'margin: 15px 0; '
        'font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;'
    )

    TH_STYLES = (
        'background-color: #063672; '
        'color: white; '
        'padding: 12px 10px; '
        'text-align: left; '
        'font-weight: 600; '
        'font-size: 13px; '
        'border: 1px solid #063672;'
    )

    TD_STYLES = (
        'padding: 10px; '
        'border: 1px solid #e9ecef; '
        'font-size: 14px; '
        'color: #333333;'
    )

    IMG_STYLES = (
        'max-width: 100%; '
        'height: auto; '
        'display: block; '
        'margin: 10px auto;'
    )

    def __init__(self, max_image_width: int = 600):
        """
        Initialize the optimizer.

        Args:
            max_image_width: Maximum width for images in pixels
        """
        self.max_image_width = max_image_width
        self.logger = logging.getLogger(f"{__name__}.EmailHTMLOptimizer")

    def optimize(self, html: str) -> str:
        """
        Optimize HTML for email client compatibility.

        Args:
            html: Original HTML content

        Returns:
            Optimized HTML with inline styles
        """
        if not html:
            return html

        # Apply optimizations in order
        html = self._add_table_styles(html)
        html = self._add_image_attributes(html)
        html = self._add_chart_fallbacks(html)
        html = self._ensure_doctype(html)

        return html

    def _add_table_styles(self, html: str) -> str:
        """Add inline styles to tables for email compatibility."""
        # Add styles to <table> tags that don't have inline styles
        html = re.sub(
            r'<table([^>]*?)(?<!style=")>',
            lambda m: f'<table{m.group(1)} style="{self.TABLE_STYLES}">',
            html,
            flags=re.IGNORECASE
        )

        # Add styles to <th> tags
        html = re.sub(
            r'<th([^>]*?)(?<!style=")>',
            lambda m: f'<th{m.group(1)} style="{self.TH_STYLES}">',
            html,
            flags=re.IGNORECASE
        )

        # Add styles to <td> tags that don't have inline styles
        html = re.sub(
            r'<td([^>]*?)(?<!style=")>',
            lambda m: f'<td{m.group(1)} style="{self.TD_STYLES}">',
            html,
            flags=re.IGNORECASE
        )

        return html

    def _add_image_attributes(self, html: str) -> str:
        """Add width, height, and style attributes to images."""
        def process_img(match):
            tag = match.group(0)

            # Add style if not present
            if 'style=' not in tag.lower():
                tag = tag.replace('>', f' style="{self.IMG_STYLES}">')

            # Add width attribute if not present
            if 'width=' not in tag.lower():
                tag = tag.replace('>', f' width="{self.max_image_width}">')

            # Ensure border="0" for email clients
            if 'border=' not in tag.lower():
                tag = tag.replace('>', ' border="0">')

            return tag

        return re.sub(r'<img[^>]+>', process_img, html, flags=re.IGNORECASE)

    def _add_chart_fallbacks(self, html: str) -> str:
        """Add fallback text for chart images."""
        # Pattern to find chart images with base64 data
        pattern = r'(<img[^>]*src="data:image/[^"]*base64,[^"]*"[^>]*alt="([^"]*)"[^>]*>)'

        def add_fallback(match):
            img_tag = match.group(1)
            alt_text = match.group(2) or "Chart"

            # Wrap in a div with noscript fallback
            return f'''<div class="chart-wrapper">
                {img_tag}
                <!--[if !mso]><!-->
                <noscript>
                    <p style="text-align: center; color: #666; font-style: italic;">[{alt_text}]</p>
                </noscript>
                <!--<![endif]-->
            </div>'''

        return re.sub(pattern, add_fallback, html, flags=re.IGNORECASE)

    def _ensure_doctype(self, html: str) -> str:
        """Ensure proper DOCTYPE and html attributes for email."""
        if not html.strip().lower().startswith('<!doctype'):
            html = '<!DOCTYPE html>\n' + html

        # Add xmlns for better Outlook support
        if 'xmlns=' not in html:
            html = html.replace(
                '<html',
                '<html xmlns="http://www.w3.org/1999/xhtml"',
                1
            )

        return html

    def create_inline_chart_html(
        self,
        base64_data: str,
        alt_text: str,
        title: Optional[str] = None,
        width: Optional[int] = None
    ) -> str:
        """
        Create email-compatible HTML for an inline chart.

        Args:
            base64_data: Base64-encoded image data
            alt_text: Alt text for the image
            title: Optional title below the chart
            width: Optional width (defaults to max_image_width)

        Returns:
            HTML string for the chart
        """
        img_width = width or self.max_image_width

        html = f'''
        <table role="presentation" style="width: 100%; border: none; margin: 20px 0;">
            <tr>
                <td style="text-align: center; padding: 0;">
                    <img src="data:image/png;base64,{base64_data}"
                         alt="{alt_text}"
                         width="{img_width}"
                         style="{self.IMG_STYLES}"
                         border="0" />
        '''

        if title:
            html += f'''
                    <p style="text-align: center; color: #666; font-size: 12px; margin-top: 8px; font-style: italic;">
                        {title}
                    </p>
            '''

        html += '''
                </td>
            </tr>
        </table>
        '''

        return html

    def create_data_table_html(
        self,
        headers: List[str],
        rows: List[List[str]],
        title: Optional[str] = None,
        highlight_column: Optional[int] = None
    ) -> str:
        """
        Create an email-compatible HTML table.

        Args:
            headers: List of header strings
            rows: List of row data (each row is a list of cell strings)
            title: Optional table title
            highlight_column: Column index to highlight (for best value, etc.)

        Returns:
            HTML string for the table
        """
        html = ''

        if title:
            html += f'''
            <p style="font-weight: 600; color: #063672; margin-bottom: 10px; font-size: 14px;">
                {title}
            </p>
            '''

        html += f'<table style="{self.TABLE_STYLES}">'

        # Headers
        html += '<tr>'
        for header in headers:
            html += f'<th style="{self.TH_STYLES}">{header}</th>'
        html += '</tr>'

        # Rows
        for i, row in enumerate(rows):
            bg_color = '#f8f9fa' if i % 2 == 1 else '#ffffff'
            html += '<tr>'
            for j, cell in enumerate(row):
                cell_style = self.TD_STYLES + f' background-color: {bg_color};'
                if highlight_column is not None and j == highlight_column:
                    cell_style += ' color: #28a745; font-weight: 700;'
                html += f'<td style="{cell_style}">{cell}</td>'
            html += '</tr>'

        html += '</table>'

        return html

    def create_value_badge_html(self, value: float, label: str) -> str:
        """
        Create an email-compatible value badge (for EV%, edge, etc.).

        Args:
            value: Numeric value
            label: Label for the badge

        Returns:
            HTML string for the badge
        """
        if value > 5:
            bg_color = '#28a745'
            text_color = '#ffffff'
        elif value > 0:
            bg_color = '#ffc107'
            text_color = '#333333'
        else:
            bg_color = '#dc3545'
            text_color = '#ffffff'

        return f'''
        <span style="
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            background-color: {bg_color};
            color: {text_color};
        ">
            {label}: {value:+.1f}%
        </span>
        '''


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

    def __init__(self, config: Optional[EmailConfig] = None, optimize_html: bool = True):
        """
        Initialize the notifier with optional email configuration.

        Args:
            config: EmailConfig instance. If None, loads from environment.
            optimize_html: Whether to optimize HTML for email client compatibility
        """
        self.config = config or EmailConfig.from_env()
        self.sender = HTMLEmailSender(self.config)
        self.optimize_html = optimize_html
        self.optimizer = EmailHTMLOptimizer() if optimize_html else None
        self.logger = logging.getLogger(f"{__name__}.IntelligenceBriefNotifier")

    def send_report(
        self,
        report: Any,  # IntelligenceBrief from report_builder
        recipients: Optional[List[str]] = None,
        include_charts_as_attachments: bool = False,
        optimize_for_email: Optional[bool] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Send an IntelligenceBrief report via email.

        Args:
            report: IntelligenceBrief object from ReportBuilder
            recipients: Optional list of email recipients (overrides config)
            include_charts_as_attachments: Whether to attach charts as separate files
            optimize_for_email: Override HTML optimization (None uses class default)

        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        try:
            # Generate HTML content
            html_content = report.to_html()

            # Optimize HTML for email clients
            should_optimize = optimize_for_email if optimize_for_email is not None else self.optimize_html
            if should_optimize and self.optimizer:
                html_content = self.optimizer.optimize(html_content)
                self.logger.info("HTML content optimized for email clients")

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
