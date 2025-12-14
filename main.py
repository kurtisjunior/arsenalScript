#!/usr/bin/env python3
"""Arsenal Match Notifier - Main Orchestrator"""

import os
import logging
from datetime import datetime, timedelta
import pytz
import requests
import apprise

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
NOTIFY_URL = os.getenv("NOTIFY_URL")
USER_TIMEZONE = os.getenv("USER_TIMEZONE", "America/New_York")
ARSENAL_TEAM_ID = 57

# Channel mapping
CHANNELS = {
    "Premier League": "NBC, USA Network, or Peacock (check NBCSports.com)",
    "UEFA Champions League": "Paramount+ or CBS Sports",
    "FA Cup": "ESPN+",
    "EFL Cup": "ESPN+",
}


def get_next_fixture():
    """Fetch Arsenal's next match from football-data.org"""
    url = f"https://api.football-data.org/v4/teams/{ARSENAL_TEAM_ID}/matches"
    headers = {"X-Auth-Token": API_KEY}
    params = {"status": "SCHEDULED", "limit": 1}

    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()

    matches = response.json().get("matches", [])
    return matches[0] if matches else None


def should_notify_today(match_utc_date):
    """Check if we should send notification today (18 hours before match)"""
    user_tz = pytz.timezone(USER_TIMEZONE)
    now_local = datetime.now(user_tz)
    match_local = datetime.fromisoformat(match_utc_date.replace('Z', '+00:00')).astimezone(user_tz)

    notify_time = match_local - timedelta(hours=18)
    time_diff = abs((now_local - notify_time).total_seconds())

    return time_diff < 3600  # Within 1 hour window


def format_notification(match):
    """Format the notification message"""
    home = match["homeTeam"]["name"]
    away = match["awayTeam"]["name"]
    competition = match["competition"]["name"]
    match_time = datetime.fromisoformat(match["utcDate"].replace('Z', '+00:00'))

    user_tz = pytz.timezone(USER_TIMEZONE)
    local_time = match_time.astimezone(user_tz)

    is_home = "Arsenal" in home
    opponent = away if is_home else home
    location = "HOME" if is_home else "AWAY"
    channel = CHANNELS.get(competition, "Check local listings")

    return f"""🔴⚪ Arsenal Match Tomorrow!

Arsenal vs {opponent}
{competition} - {location} game

📅 {local_time.strftime('%A, %B %d, %Y')}
🕐 {local_time.strftime('%I:%M %p %Z')}
📺 {channel}

COYG! 🔴⚪"""


def send_notification(message):
    """Send notification via Apprise"""
    apobj = apprise.Apprise()
    apobj.add(NOTIFY_URL)
    success = apobj.notify(title="Arsenal Match Tomorrow!", body=message)
    return success


def main():
    """Main execution"""
    logging.info("Arsenal Match Notifier - Started")

    try:
        # 1. Get next fixture
        match = get_next_fixture()
        if not match:
            logging.info("No upcoming fixtures found")
            return

        logging.info(f"Next match: {match['homeTeam']['name']} vs {match['awayTeam']['name']}")

        # 2. Check if we should notify today
        if not should_notify_today(match["utcDate"]):
            logging.info("Not time to notify yet")
            return

        # 3. Format and send notification
        message = format_notification(match)
        logging.info("Sending notification...")

        if send_notification(message):
            logging.info("✓ Notification sent successfully!")
        else:
            logging.error("✗ Failed to send notification")

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        raise

    finally:
        logging.info("Arsenal Match Notifier - Completed")


if __name__ == "__main__":
    main()
