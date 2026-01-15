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
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY") or ""
NOTIFY_URL = os.getenv("NOTIFY_URL") or ""
USER_TIMEZONE = os.getenv("USER_TIMEZONE") or "America/New_York"
NOTIFY_LOCAL_TIME = os.getenv("NOTIFY_LOCAL_TIME") or "18:00"  # 6pm local, night before
try:
    NOTIFY_WINDOW_MINUTES = int(os.getenv("NOTIFY_WINDOW_MINUTES") or "240")  # allow for GH schedule drift/DST
except ValueError:
    raise ValueError("NOTIFY_WINDOW_MINUTES must be an integer number of minutes (e.g. 240)")
FORCE_NOTIFY = (os.getenv("FORCE_NOTIFY") or "").strip().lower() in {"1", "true", "yes", "y", "on"}
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
    """Check if we should send a notification now.

    The GitHub Action runs on a fixed daily schedule, so we notify at a fixed local
    time on the day before the match (default 18:00). This avoids missing matches
    with different kick-off times.
    """
    user_tz = pytz.timezone(USER_TIMEZONE)
    now_local = datetime.now(user_tz)
    match_local = datetime.fromisoformat(match_utc_date.replace('Z', '+00:00')).astimezone(user_tz)

    # Only notify on the day before the match (in the user's timezone)
    if match_local.date() != (now_local.date() + timedelta(days=1)):
        return False

    try:
        hour_str, minute_str = NOTIFY_LOCAL_TIME.split(":")
        notify_hour = int(hour_str)
        notify_minute = int(minute_str)
    except Exception:
        raise ValueError("NOTIFY_LOCAL_TIME must be in HH:MM 24-hour format (e.g. 18:00)")

    notify_time = now_local.replace(hour=notify_hour, minute=notify_minute, second=0, microsecond=0)
    time_diff = abs((now_local - notify_time).total_seconds())

    return time_diff <= (NOTIFY_WINDOW_MINUTES * 60)


def format_notification(match):
    """Format the notification message"""
    home = match["homeTeam"]["name"]
    away = match["awayTeam"]["name"]
    competition = match["competition"]["name"]
    match_time = datetime.fromisoformat(match["utcDate"].replace('Z', '+00:00'))

    user_tz = pytz.timezone(USER_TIMEZONE)
    local_time = match_time.astimezone(user_tz)
    days_until = (local_time.date() - datetime.now(user_tz).date()).days

    if days_until == 1:
        headline = "🔴⚪ Arsenal Match Tomorrow!"
    elif days_until == 0:
        headline = "🔴⚪ Arsenal Match Today!"
    elif days_until > 1:
        headline = f"🔴⚪ Arsenal Match in {days_until} days!"
    else:
        headline = "🔴⚪ Arsenal Match Update!"

    is_home = "Arsenal" in home
    opponent = away if is_home else home
    location = "HOME" if is_home else "AWAY"
    channel = CHANNELS.get(competition, "Check local listings")

    return f"""{headline}

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
        if not API_KEY:
            raise RuntimeError("Missing FOOTBALL_DATA_API_KEY environment variable")
        if not NOTIFY_URL:
            raise RuntimeError("Missing NOTIFY_URL environment variable")

        # 1. Get next fixture
        match = get_next_fixture()
        if not match:
            logging.info("No upcoming fixtures found")
            return

        logging.info(f"Next match: {match['homeTeam']['name']} vs {match['awayTeam']['name']}")

        # 2. Check if we should notify today
        if not FORCE_NOTIFY and not should_notify_today(match["utcDate"]):
            logging.info("Not time to notify yet")
            return
        if FORCE_NOTIFY:
            logging.info("FORCE_NOTIFY enabled; sending notification now")

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
