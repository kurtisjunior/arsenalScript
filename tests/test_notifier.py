#!/usr/bin/env python3
"""
Unit Tests for HTML Email Notifier

Tests for reporting/notifier.py covering:
- EmailConfig creation and validation
- EmailAttachment handling (files and base64)
- EmailMessage creation and formatting
- HTMLEmailSender SMTP operations (mocked)
- IntelligenceBriefNotifier integration
- HTML to plain text conversion
- Error handling for SMTP failures

Task: arsenalScript-vqp.51 - Test HTML email sender
"""

import base64
import os
import pytest
import smtplib
import tempfile
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch, PropertyMock
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from reporting.notifier import (
    EmailConfig,
    EmailAttachment,
    EmailMessage,
    HTMLEmailSender,
    IntelligenceBriefNotifier,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_env_vars():
    """Set up mock environment variables for email configuration."""
    env_vars = {
        "SMTP_HOST": "smtp.test.com",
        "SMTP_PORT": "587",
        "SMTP_USER": "test@example.com",
        "SMTP_PASSWORD": "test_password",
        "EMAIL_RECIPIENTS": "recipient1@example.com,recipient2@example.com",
        "SMTP_SENDER_NAME": "Test Sender",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def valid_email_config():
    """Create a valid email configuration for testing."""
    return EmailConfig(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        smtp_user="sender@gmail.com",
        smtp_password="app_password_123",
        use_tls=True,
        sender_name="Arsenal Intelligence Brief",
        sender_email="sender@gmail.com",
        recipients=["recipient@example.com"],
    )


@pytest.fixture
def sample_html_content():
    """Sample HTML content for email tests."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Email</title>
        <style>body { font-family: Arial; }</style>
    </head>
    <body>
        <h1>Arsenal Intelligence Brief</h1>
        <p>This is a <strong>test</strong> email with <em>formatting</em>.</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
        <table>
            <tr><td>Cell 1</td><td>Cell 2</td></tr>
        </table>
        <br>
        <p>Footer text &amp; special chars &lt;test&gt;</p>
    </body>
    </html>
    """


@pytest.fixture
def mock_intelligence_brief():
    """Create a mock IntelligenceBrief object for testing."""
    mock_brief = MagicMock()
    mock_brief.match_id = "20260120_ARS_CHE"
    mock_brief.home_team = "Arsenal"
    mock_brief.away_team = "Chelsea"
    mock_brief.match_date = "2026-01-20T15:00:00Z"
    mock_brief.competition = "Premier League"
    mock_brief.generated_at = "2026-01-19T10:00:00Z"
    mock_brief.charts = {
        "odds_comparison": base64.b64encode(b"fake_png_data").decode('utf-8'),
    }
    mock_brief.data_completeness = MagicMock()
    mock_brief.data_completeness.completeness_score = 80.0
    mock_brief.to_html.return_value = "<html><body>Test Report</body></html>"
    return mock_brief


@pytest.fixture
def temp_attachment_file():
    """Create a temporary file for attachment testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is test file content for attachment testing.")
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# =============================================================================
# TEST: EmailConfig
# =============================================================================

class TestEmailConfig:
    """Tests for EmailConfig dataclass."""

    def test_default_values(self):
        """Test EmailConfig has sensible defaults."""
        config = EmailConfig()
        assert config.smtp_host == "smtp.gmail.com"
        assert config.smtp_port == 587
        assert config.use_tls is True
        assert config.use_ssl is False
        assert config.timeout == 30
        assert config.sender_name == "Arsenal Intelligence Brief"
        assert config.recipients == []

    def test_from_env(self, mock_env_vars):
        """Test EmailConfig.from_env() loads from environment."""
        config = EmailConfig.from_env()

        assert config.smtp_host == "smtp.test.com"
        assert config.smtp_port == 587
        assert config.smtp_user == "test@example.com"
        assert config.smtp_password == "test_password"
        assert config.recipients == ["recipient1@example.com", "recipient2@example.com"]
        assert config.sender_name == "Test Sender"

    def test_from_env_missing_vars(self):
        """Test EmailConfig.from_env() handles missing env vars."""
        with patch.dict(os.environ, {}, clear=True):
            config = EmailConfig.from_env()

            assert config.smtp_host == "smtp.gmail.com"  # default
            assert config.smtp_user is None
            assert config.smtp_password is None
            assert config.recipients == []

    def test_is_valid_with_complete_config(self, valid_email_config):
        """Test is_valid() returns True for complete configuration."""
        is_valid, message = valid_email_config.is_valid()
        assert is_valid is True
        assert message == "Configuration valid"

    def test_is_valid_missing_smtp_user(self, valid_email_config):
        """Test is_valid() returns False when SMTP user is missing."""
        valid_email_config.smtp_user = None
        is_valid, message = valid_email_config.is_valid()
        assert is_valid is False
        assert "SMTP_USER" in message

    def test_is_valid_missing_smtp_password(self, valid_email_config):
        """Test is_valid() returns False when SMTP password is missing."""
        valid_email_config.smtp_password = None
        is_valid, message = valid_email_config.is_valid()
        assert is_valid is False
        assert "SMTP_PASSWORD" in message

    def test_is_valid_missing_recipients(self, valid_email_config):
        """Test is_valid() returns False when no recipients configured."""
        valid_email_config.recipients = []
        is_valid, message = valid_email_config.is_valid()
        assert is_valid is False
        assert "recipients" in message.lower()

    def test_recipients_parsing_with_whitespace(self):
        """Test that recipients are parsed and trimmed correctly."""
        with patch.dict(os.environ, {"EMAIL_RECIPIENTS": " a@b.com , c@d.com , e@f.com "}, clear=False):
            config = EmailConfig.from_env()
            assert config.recipients == ["a@b.com", "c@d.com", "e@f.com"]


# =============================================================================
# TEST: EmailAttachment
# =============================================================================

class TestEmailAttachment:
    """Tests for EmailAttachment dataclass."""

    def test_from_file_txt(self, temp_attachment_file):
        """Test creating attachment from a text file."""
        attachment = EmailAttachment.from_file(temp_attachment_file)

        assert attachment.filename == Path(temp_attachment_file).name
        assert attachment.content_type == "text/plain"
        assert b"test file content" in attachment.content

    def test_from_file_png_content_type(self):
        """Test content type detection for PNG files."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(b'\x89PNG\r\n\x1a\n')  # PNG magic bytes
            temp_path = f.name

        try:
            attachment = EmailAttachment.from_file(temp_path)
            assert attachment.content_type == "image/png"
        finally:
            os.unlink(temp_path)

    def test_from_file_custom_content_type(self, temp_attachment_file):
        """Test overriding content type detection."""
        attachment = EmailAttachment.from_file(
            temp_attachment_file,
            content_type="application/custom"
        )
        assert attachment.content_type == "application/custom"

    def test_from_file_not_found(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            EmailAttachment.from_file("/nonexistent/path/file.txt")

    def test_from_base64_valid(self):
        """Test creating attachment from base64 data."""
        original_data = b"Hello, World!"
        base64_data = base64.b64encode(original_data).decode('utf-8')

        attachment = EmailAttachment.from_base64(
            base64_data,
            filename="test.txt",
            content_type="text/plain"
        )

        assert attachment.filename == "test.txt"
        assert attachment.content == original_data
        assert attachment.content_type == "text/plain"

    def test_from_base64_image(self):
        """Test creating image attachment from base64."""
        # Fake PNG data
        png_data = b'\x89PNG\r\n\x1a\nFAKEPNGDATA'
        base64_data = base64.b64encode(png_data).decode('utf-8')

        attachment = EmailAttachment.from_base64(
            base64_data,
            filename="chart.png",
            content_type="image/png"
        )

        assert attachment.filename == "chart.png"
        assert attachment.content == png_data
        assert attachment.content_type == "image/png"

    def test_content_type_detection_various_extensions(self):
        """Test content type detection for various file extensions."""
        extensions_and_types = {
            '.pdf': 'application/pdf',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.html': 'text/html',
            '.json': 'application/json',
            '.csv': 'text/csv',
            '.xyz': 'application/octet-stream',  # unknown extension
        }

        for ext, expected_type in extensions_and_types.items():
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(b"test content")
                temp_path = f.name

            try:
                attachment = EmailAttachment.from_file(temp_path)
                assert attachment.content_type == expected_type, f"Failed for {ext}"
            finally:
                os.unlink(temp_path)


# =============================================================================
# TEST: EmailMessage
# =============================================================================

class TestEmailMessage:
    """Tests for EmailMessage dataclass."""

    def test_basic_creation(self, sample_html_content):
        """Test basic EmailMessage creation."""
        message = EmailMessage(
            subject="Test Subject",
            html_content=sample_html_content,
        )

        assert message.subject == "Test Subject"
        assert "Arsenal Intelligence Brief" in message.html_content
        assert message.plain_content is None
        assert message.attachments == []
        assert message.inline_images == {}

    def test_with_plain_content(self, sample_html_content):
        """Test EmailMessage with explicit plain text content."""
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
            plain_content="Plain text version of the email",
        )

        assert message.plain_content == "Plain text version of the email"

    def test_add_attachment(self, sample_html_content):
        """Test adding attachment to message."""
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        attachment = EmailAttachment(
            filename="test.txt",
            content=b"Test content",
            content_type="text/plain"
        )
        message.add_attachment(attachment)

        assert len(message.attachments) == 1
        assert message.attachments[0].filename == "test.txt"

    def test_add_attachment_from_file(self, sample_html_content, temp_attachment_file):
        """Test adding attachment from file path."""
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        message.add_attachment_from_file(temp_attachment_file)

        assert len(message.attachments) == 1
        assert message.attachments[0].content_type == "text/plain"

    def test_add_inline_image(self, sample_html_content):
        """Test adding inline image with Content-ID."""
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        attachment = EmailAttachment(
            filename="logo.png",
            content=b"PNG_DATA",
            content_type="image/png"
        )
        message.add_inline_image("logo_cid", attachment)

        assert "logo_cid" in message.inline_images
        assert message.inline_images["logo_cid"].content_id == "logo_cid"

    def test_add_inline_image_from_base64(self, sample_html_content):
        """Test adding inline image from base64 data."""
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        png_data = base64.b64encode(b"FAKE_PNG").decode('utf-8')
        message.add_inline_image_from_base64(
            content_id="chart_cid",
            base64_data=png_data,
            filename="chart.png"
        )

        assert "chart_cid" in message.inline_images
        assert message.inline_images["chart_cid"].filename == "chart.png"

    def test_override_recipients(self, sample_html_content):
        """Test message-specific recipient override."""
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
            recipients=["override@example.com"],
            cc=["cc@example.com"],
            bcc=["bcc@example.com"],
        )

        assert message.recipients == ["override@example.com"]
        assert message.cc == ["cc@example.com"]
        assert message.bcc == ["bcc@example.com"]


# =============================================================================
# TEST: HTMLEmailSender
# =============================================================================

class TestHTMLEmailSender:
    """Tests for HTMLEmailSender class."""

    def test_init_with_config(self, valid_email_config):
        """Test sender initialization with config."""
        sender = HTMLEmailSender(valid_email_config)
        assert sender.config == valid_email_config

    def test_init_from_env(self, mock_env_vars):
        """Test sender initialization from environment."""
        sender = HTMLEmailSender()
        assert sender.config.smtp_host == "smtp.test.com"
        assert sender.config.smtp_user == "test@example.com"

    def test_html_to_plain_basic(self, valid_email_config):
        """Test HTML to plain text conversion."""
        sender = HTMLEmailSender(valid_email_config)

        html = """
        <html>
        <body>
            <h1>Title</h1>
            <p>Paragraph with <strong>bold</strong> text.</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </body>
        </html>
        """

        plain = sender._html_to_plain(html)

        assert "Title" in plain
        assert "Paragraph with" in plain
        assert "bold" in plain
        assert "Item 1" in plain
        assert "<" not in plain  # No HTML tags
        assert ">" not in plain

    def test_html_to_plain_removes_style_script(self, valid_email_config):
        """Test that style and script blocks are removed."""
        sender = HTMLEmailSender(valid_email_config)

        html = """
        <html>
        <head>
            <style>body { color: red; }</style>
            <script>alert('test');</script>
        </head>
        <body>
            <p>Content</p>
        </body>
        </html>
        """

        plain = sender._html_to_plain(html)

        assert "color: red" not in plain
        assert "alert" not in plain
        assert "Content" in plain

    def test_html_to_plain_decodes_entities(self, valid_email_config):
        """Test HTML entity decoding."""
        sender = HTMLEmailSender(valid_email_config)

        html = "<p>Test &amp; &lt;special&gt; &quot;chars&quot; &nbsp;</p>"
        plain = sender._html_to_plain(html)

        assert "&" in plain
        assert "<special>" in plain
        assert '"chars"' in plain

    def test_create_mime_message_basic(self, valid_email_config, sample_html_content):
        """Test MIME message creation."""
        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test Subject",
            html_content=sample_html_content,
        )

        mime_msg = sender._create_mime_message(message)

        assert mime_msg['Subject'] == "Test Subject"
        assert "sender@gmail.com" in mime_msg['From']
        assert "recipient@example.com" in mime_msg['To']

    def test_create_mime_message_with_cc(self, valid_email_config, sample_html_content):
        """Test MIME message with CC recipients."""
        valid_email_config.cc = ["cc@example.com"]
        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        mime_msg = sender._create_mime_message(message)

        assert "cc@example.com" in mime_msg['Cc']

    def test_create_mime_message_with_reply_to(self, valid_email_config, sample_html_content):
        """Test MIME message with Reply-To header."""
        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
            reply_to="reply@example.com",
        )

        mime_msg = sender._create_mime_message(message)

        assert mime_msg['Reply-To'] == "reply@example.com"

    @patch('reporting.notifier.smtplib.SMTP')
    def test_send_success(self, mock_smtp_class, valid_email_config, sample_html_content):
        """Test successful email sending."""
        # Setup mock - the SMTP class itself is used as context manager
        mock_smtp_instance = MagicMock()
        mock_smtp_class.return_value = mock_smtp_instance
        mock_smtp_instance.__enter__ = Mock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__ = Mock(return_value=False)

        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        success, error = sender.send(message)

        assert success is True
        assert error is None
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with("sender@gmail.com", "app_password_123")
        mock_smtp_instance.sendmail.assert_called_once()

    @patch('reporting.notifier.smtplib.SMTP')
    def test_send_authentication_error(self, mock_smtp_class, valid_email_config, sample_html_content):
        """Test handling of SMTP authentication error."""
        # Setup mock to raise authentication error
        mock_smtp_instance = MagicMock()
        mock_smtp_instance.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Authentication failed")
        mock_smtp_class.return_value = mock_smtp_instance
        mock_smtp_instance.__enter__ = Mock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__ = Mock(return_value=False)

        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        success, error = sender.send(message)

        assert success is False
        assert "authentication" in error.lower()

    @patch('reporting.notifier.smtplib.SMTP')
    def test_send_recipients_refused(self, mock_smtp_class, valid_email_config, sample_html_content):
        """Test handling of recipients refused error."""
        mock_smtp = MagicMock()
        mock_smtp.sendmail.side_effect = smtplib.SMTPRecipientsRefused(
            {"recipient@example.com": (550, b"User unknown")}
        )
        mock_smtp_class.return_value.__enter__ = Mock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = Mock(return_value=False)

        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        success, error = sender.send(message)

        assert success is False
        assert "refused" in error.lower()

    def test_send_invalid_config(self, sample_html_content):
        """Test send fails with invalid configuration."""
        config = EmailConfig()  # Missing user/password
        sender = HTMLEmailSender(config)
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        success, error = sender.send(message)

        assert success is False
        assert "SMTP_USER" in error

    def test_send_no_recipients(self, valid_email_config, sample_html_content):
        """Test send fails with no recipients."""
        valid_email_config.recipients = []
        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        success, error = sender.send(message)

        assert success is False
        assert "recipient" in error.lower()

    @patch('reporting.notifier.smtplib.SMTP')
    def test_send_with_attachments(self, mock_smtp_class, valid_email_config, sample_html_content, temp_attachment_file):
        """Test sending email with attachments."""
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__ = Mock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = Mock(return_value=False)

        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test with Attachment",
            html_content=sample_html_content,
        )
        message.add_attachment_from_file(temp_attachment_file)

        success, error = sender.send(message)

        assert success is True
        # Verify sendmail was called with content containing attachment
        call_args = mock_smtp.sendmail.call_args
        email_content = call_args[0][2]  # Third argument is the message content
        assert "Content-Disposition: attachment" in email_content

    @patch('reporting.notifier.smtplib.SMTP')
    def test_test_connection_success(self, mock_smtp_class, valid_email_config):
        """Test connection testing success."""
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__ = Mock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = Mock(return_value=False)

        sender = HTMLEmailSender(valid_email_config)
        success, message = sender.test_connection()

        assert success is True
        assert "Successfully connected" in message

    @patch('reporting.notifier.smtplib.SMTP')
    def test_test_connection_failure(self, mock_smtp_class, valid_email_config):
        """Test connection testing failure."""
        mock_smtp_class.side_effect = smtplib.SMTPConnectError(421, b"Service unavailable")

        sender = HTMLEmailSender(valid_email_config)
        success, message = sender.test_connection()

        assert success is False
        assert "Connection failed" in message or "SMTP error" in message

    @patch('reporting.notifier.smtplib.SMTP_SSL')
    def test_ssl_connection(self, mock_smtp_ssl_class, valid_email_config, sample_html_content):
        """Test SSL connection mode."""
        valid_email_config.use_ssl = True
        valid_email_config.use_tls = False
        valid_email_config.smtp_port = 465

        mock_smtp = MagicMock()
        mock_smtp_ssl_class.return_value.__enter__ = Mock(return_value=mock_smtp)
        mock_smtp_ssl_class.return_value.__exit__ = Mock(return_value=False)

        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        success, error = sender.send(message)

        assert success is True
        mock_smtp_ssl_class.assert_called_once()


# =============================================================================
# TEST: IntelligenceBriefNotifier
# =============================================================================

class TestIntelligenceBriefNotifier:
    """Tests for IntelligenceBriefNotifier class."""

    def test_init(self, valid_email_config):
        """Test notifier initialization."""
        notifier = IntelligenceBriefNotifier(valid_email_config)
        assert notifier.config == valid_email_config
        assert notifier.sender is not None

    @patch('reporting.notifier.smtplib.SMTP')
    def test_send_report_success(self, mock_smtp_class, valid_email_config, mock_intelligence_brief):
        """Test sending intelligence brief report."""
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__ = Mock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = Mock(return_value=False)

        notifier = IntelligenceBriefNotifier(valid_email_config)
        success, error = notifier.send_report(mock_intelligence_brief)

        assert success is True
        assert error is None
        mock_smtp.sendmail.assert_called_once()

        # Check subject includes team names
        call_args = mock_smtp.sendmail.call_args
        email_content = call_args[0][2]
        assert "Arsenal" in email_content
        assert "Chelsea" in email_content

    @patch('reporting.notifier.smtplib.SMTP')
    def test_send_report_with_charts_as_attachments(self, mock_smtp_class, valid_email_config, mock_intelligence_brief):
        """Test sending report with charts as separate attachments."""
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__ = Mock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = Mock(return_value=False)

        notifier = IntelligenceBriefNotifier(valid_email_config)
        success, error = notifier.send_report(
            mock_intelligence_brief,
            include_charts_as_attachments=True
        )

        assert success is True
        call_args = mock_smtp.sendmail.call_args
        email_content = call_args[0][2]
        assert "odds_comparison.png" in email_content

    @patch('reporting.notifier.smtplib.SMTP')
    def test_send_report_custom_recipients(self, mock_smtp_class, valid_email_config, mock_intelligence_brief):
        """Test sending report to custom recipients."""
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__ = Mock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = Mock(return_value=False)

        notifier = IntelligenceBriefNotifier(valid_email_config)
        success, error = notifier.send_report(
            mock_intelligence_brief,
            recipients=["custom@example.com"]
        )

        assert success is True
        call_args = mock_smtp.sendmail.call_args
        recipients = call_args[0][1]
        assert "custom@example.com" in recipients

    def test_send_report_invalid_config(self, mock_intelligence_brief):
        """Test send_report fails with invalid config."""
        config = EmailConfig()  # Invalid - no credentials
        notifier = IntelligenceBriefNotifier(config)

        success, error = notifier.send_report(mock_intelligence_brief)

        assert success is False
        assert error is not None

    @patch('reporting.notifier.smtplib.SMTP')
    def test_send_html_file(self, mock_smtp_class, valid_email_config):
        """Test sending HTML file."""
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__ = Mock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = Mock(return_value=False)

        # Create temp HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write("<html><body><h1>Test Report</h1></body></html>")
            temp_path = f.name

        try:
            notifier = IntelligenceBriefNotifier(valid_email_config)
            success, error = notifier.send_html_file(
                temp_path,
                subject="Test Report"
            )

            assert success is True
            call_args = mock_smtp.sendmail.call_args
            email_content = call_args[0][2]
            assert "Test Report" in email_content
        finally:
            os.unlink(temp_path)

    def test_send_html_file_not_found(self, valid_email_config):
        """Test send_html_file with non-existent file."""
        notifier = IntelligenceBriefNotifier(valid_email_config)
        success, error = notifier.send_html_file(
            "/nonexistent/path.html",
            subject="Test"
        )

        assert success is False
        assert "not found" in error.lower()


# =============================================================================
# TEST: Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling scenarios."""

    @patch('reporting.notifier.smtplib.SMTP')
    def test_timeout_error(self, mock_smtp_class, valid_email_config, sample_html_content):
        """Test handling of connection timeout."""
        mock_smtp_class.side_effect = TimeoutError("Connection timed out")

        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        success, error = sender.send(message)

        assert success is False
        assert "timeout" in error.lower()

    @patch('reporting.notifier.smtplib.SMTP')
    def test_smtp_data_error(self, mock_smtp_class, valid_email_config, sample_html_content):
        """Test handling of SMTP data error (message too large)."""
        mock_smtp = MagicMock()
        mock_smtp.sendmail.side_effect = smtplib.SMTPDataError(552, b"Message too large")
        mock_smtp_class.return_value.__enter__ = Mock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = Mock(return_value=False)

        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        success, error = sender.send(message)

        assert success is False
        assert "data error" in error.lower() or "too large" in error.lower()

    @patch('reporting.notifier.smtplib.SMTP')
    def test_sender_refused_error(self, mock_smtp_class, valid_email_config, sample_html_content):
        """Test handling of sender refused error."""
        mock_smtp = MagicMock()
        mock_smtp.sendmail.side_effect = smtplib.SMTPSenderRefused(
            550, b"Sender not authorized", "sender@gmail.com"
        )
        mock_smtp_class.return_value.__enter__ = Mock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = Mock(return_value=False)

        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        success, error = sender.send(message)

        assert success is False
        assert "sender" in error.lower() and "refused" in error.lower()

    @patch('reporting.notifier.smtplib.SMTP')
    def test_generic_smtp_exception(self, mock_smtp_class, valid_email_config, sample_html_content):
        """Test handling of generic SMTP exception."""
        mock_smtp = MagicMock()
        mock_smtp.sendmail.side_effect = smtplib.SMTPException("Unknown SMTP error")
        mock_smtp_class.return_value.__enter__ = Mock(return_value=mock_smtp)
        mock_smtp_class.return_value.__exit__ = Mock(return_value=False)

        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        success, error = sender.send(message)

        assert success is False
        assert "SMTP error" in error

    def test_send_report_exception_handling(self, valid_email_config):
        """Test that send_report handles exceptions gracefully."""
        mock_brief = MagicMock()
        mock_brief.to_html.side_effect = Exception("Template rendering failed")

        notifier = IntelligenceBriefNotifier(valid_email_config)
        success, error = notifier.send_report(mock_brief)

        assert success is False
        assert "Template rendering failed" in error or "Failed to send" in error


# =============================================================================
# TEST: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_html_content(self, valid_email_config):
        """Test handling of empty HTML content."""
        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test",
            html_content="",
        )

        mime_msg = sender._create_mime_message(message)
        assert mime_msg is not None

    def test_very_long_subject(self, valid_email_config, sample_html_content):
        """Test handling of very long subject line."""
        sender = HTMLEmailSender(valid_email_config)
        long_subject = "A" * 1000

        message = EmailMessage(
            subject=long_subject,
            html_content=sample_html_content,
        )

        mime_msg = sender._create_mime_message(message)
        assert long_subject in mime_msg['Subject']

    def test_unicode_in_content(self, valid_email_config):
        """Test handling of Unicode characters."""
        sender = HTMLEmailSender(valid_email_config)

        html = """
        <html><body>
        <p>Unicode test: 你好世界 🌍 مرحبا العالم αβγδ</p>
        </body></html>
        """

        message = EmailMessage(
            subject="Unicode Test: ★ Special ★",
            html_content=html,
        )

        mime_msg = sender._create_mime_message(message)
        assert mime_msg is not None

    def test_special_characters_in_email(self, valid_email_config, sample_html_content):
        """Test handling of special characters in email addresses."""
        valid_email_config.recipients = ["user+tag@example.com", "user.name@sub.domain.com"]

        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        mime_msg = sender._create_mime_message(message)
        assert "user+tag@example.com" in mime_msg['To']

    def test_multiple_attachments(self, valid_email_config, sample_html_content):
        """Test message with multiple attachments."""
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        for i in range(5):
            message.add_attachment(EmailAttachment(
                filename=f"file_{i}.txt",
                content=f"Content {i}".encode(),
                content_type="text/plain"
            ))

        assert len(message.attachments) == 5

    def test_combined_attachments_and_inline_images(self, valid_email_config, sample_html_content):
        """Test message with both attachments and inline images."""
        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        # Add regular attachment
        message.add_attachment(EmailAttachment(
            filename="document.pdf",
            content=b"PDF_CONTENT",
            content_type="application/pdf"
        ))

        # Add inline image
        message.add_inline_image("logo", EmailAttachment(
            filename="logo.png",
            content=b"PNG_CONTENT",
            content_type="image/png"
        ))

        mime_msg = sender._create_mime_message(message)
        msg_string = mime_msg.as_string()

        assert "document.pdf" in msg_string
        assert "logo.png" in msg_string

    def test_bcc_not_in_headers(self, valid_email_config, sample_html_content):
        """Test that BCC recipients are not visible in headers."""
        valid_email_config.bcc = ["secret@example.com"]

        sender = HTMLEmailSender(valid_email_config)
        message = EmailMessage(
            subject="Test",
            html_content=sample_html_content,
        )

        mime_msg = sender._create_mime_message(message)

        # BCC should not appear in any header
        assert "secret@example.com" not in mime_msg.as_string().split("\n\n")[0]
