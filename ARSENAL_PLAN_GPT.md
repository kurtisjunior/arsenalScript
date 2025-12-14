# Arsenal Fixture Notification System — Ultimate Hybrid Plan

## Executive Summary

This plan artfully blends the best elements of two expert approaches to deliver a robust, adaptive, and maintainable system for notifying users of Arsenal’s next fixture and US TV channel, the night before every game. It combines:
- **Real-time adaptability**: Notification timing and delivery can follow user/system context (e.g., local time, system appearance, or user preference), inspired by the pragmatic, user-centric approach in vimchange.md.
- **Clear, actionable implementation steps**: Step-by-step guidance for setup and testing.
- **References to best practices and external resources**.
- **Modular, extensible, and robust architecture**: Retaining the strengths of the original plan.

The result is a system that is not only reliable and extensible, but also context-aware and user-friendly, ensuring notifications are always timely and relevant in real-world use.

---

## Table of Contents
1. Current System Analysis
2. Requirements & Objectives
3. Ultimate Architecture Overview
4. Data Sourcing & Caching
5. Scheduling Options
6. USA Channel Mapping Logic
7. Notification System
8. Error Handling & Monitoring
9. Implementation Roadmap
10. Security & Privacy
11. Cost & Alternatives
12. Future Enhancements
13. References

---

## 1. Current System Analysis
- Old system relies on fragile web scraping, manual execution, and UK-centric data.
- No automation, notifications, or robust error handling.
- No support for US TV channels or user-friendly configuration.

## 2. Requirements & Objectives
- **Automated**: Scheduled, no manual intervention.
- **Accurate**: Uses reputable APIs for fixtures and TV info.
- **US-Focused**: Explicitly fetches US broadcast info.
- **Multi-Channel Notification**: Email, SMS, push, chat, etc.
- **Configurable**: User can set team, time, channels, etc.
- **Extensible**: Easy to add teams, competitions, or notification methods.
- **Robust**: Error handling, logging, health checks.
- **Free or Low Cost**: Prefer free tiers and open-source tools.

## 3. Ultimate Architecture Overview
```
┌───────────────┐   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Scheduler    │→│ Fixture Fetch │→│ Channel Mapper │→│ Notifier      │
└───────────────┘   └───────────────┘   └───────────────┘   └───────────────┘
```
- **Scheduler**: GitHub Actions (default), with option for cron, cloud, or local.
- **Fixture Fetch**: API-Football and/or football-data.org, with backup static JSON and caching.
- **Channel Mapper**: Multi-layered logic for US TV, with fallback scraping and default messaging.
- **Notifier**: Apprise for multi-channel delivery, with modular config.

## 4. Data Sourcing & Caching
- **Primary**: football-data.org (free, reliable, Premier League/Europe) or API-Football (richer data, free tier).
- **Backup**: OpenFootball static JSON (GitHub CDN).
- **Caching**: Store last API response for 24h to avoid quota issues and improve reliability.
- **Fallback**: Scrape live-footballontv.com only if all else fails.

## 5. Scheduling Options
- **Default**: GitHub Actions (free, serverless, versioned, secure secrets).
- **Alternative**: Local cron, cloud scheduler, or serverless (AWS Lambda, GCP, etc.) for advanced users.
- **Configurable**: User sets notification time and timezone (e.g., 6pm local, night before match), or can opt for **real-time, context-aware scheduling** (e.g., notifications that follow system appearance, or time-of-day logic, similar to vimchange.md’s approach for Vim and Ghostty).
- **Adaptability**: Optionally, notifications can be triggered based on system events (e.g., system switches to dark mode in the evening, or user-defined triggers), for maximum relevance.

## 6. USA Channel Mapping Logic
- **Layer 1**: Competition-based mapping (Premier League → NBC/Peacock/USA, UCL → Paramount+/CBS, FA Cup → ESPN+).
- **Layer 2**: Time/opponent-based logic (big matches on NBC, others on Peacock, etc.).
- **Layer 3**: Fallback scraping for exact channel if needed.
- **Layer 4**: Default message if all else fails.
- **Config**: Modular JSON/YAML for easy updates as rights change.

## 7. Notification System
- **Library**: Apprise (supports 90+ services: SMS, push, email, chat, etc.).
- **Config**: YAML or .env for user preferences, secrets, and notification URLs.
- **Message Format**: Clear, friendly, and informative (opponent, date/time, competition, channel, home/away, timezone-aware).
- **Multi-Channel**: User can enable/disable any channel.
- **Health Checks**: Weekly test notification to ensure system is working.

## 8. Error Handling & Monitoring
- **Retry Logic**: Exponential backoff for API calls.
- **Graceful Degradation**: Fallback to backup sources and default messages.
- **Logging**: Structured logs, uploaded as artifacts in GitHub Actions.
- **Alerts**: Notify admin on repeated failures.
- **Testing**: Unit and integration tests for all modules.

## 9. Implementation Roadmap

### Step-by-Step Implementation
1. **Setup & Configuration**
  - Register for API keys (football-data.org, API-Football, etc.).
  - Clone repo and scaffold config files (YAML/.env).
  - Set up caching for API responses.
2. **Core Module Development**
  - Implement fixture fetcher, channel mapper, and notifier modules.
  - Write unit tests for each module.
3. **Scheduling & Adaptability**
  - Integrate GitHub Actions for default scheduling.
  - Add support for local cron/cloud/serverless as alternatives.
  - Implement context-aware notification logic (e.g., time-of-day, system appearance, or user triggers).
4. **Testing & Validation**
  - Run end-to-end tests.
  - Test notification delivery across all enabled channels.
  - Simulate different user/system contexts to ensure adaptability.
5. **Deployment & Monitoring**
  - Deploy with secrets configured.
  - Set up health checks and logging.
  - Document usage and troubleshooting steps.

## 10. Security & Privacy
- **Secrets**: Store API keys and notification URLs in GitHub/CI secrets or .env (never hardcoded).
- **User Data**: Minimal storage, no PII unless required for notifications.
- **Open Source**: Encourage transparency and community review.

## 11. Cost & Alternatives
- **Free Tier**: football-data.org, GitHub Actions, ntfy.sh, Discord, Email.
- **Low Cost**: Twilio SMS (~$0.30/month for 30 matches).
- **Premium**: Paid API or self-hosted notification server for advanced use.
- **Alternatives**: Local cron, serverless, or dedicated VPS for power users.

## 12. Future Enhancements
- Multi-team and multi-user support.
- Web dashboard for managing preferences.
- Post-match result notifications.
- Lineup/ticket reminders.
- Voice assistant integration.
- Advanced analytics and stats.

---

## 13. References

- [Ghostty Color Theme Features](https://ghostty.org/docs/features/theme)
- [Automatic dark mode for Terminal Apps](https://arslan.io/2025/06/06/automatic-dark-mode-for-terminal-apps-revisited/)
- [Vim time-based background switching](https://dev.to/voyeg3r/set-your-vim-colorscheme-and-background-based-on-hour-hoj)
- [Apprise Notification Library](https://github.com/caronc/apprise)
- [football-data.org API](https://www.football-data.org/)
- [API-Football](https://www.api-football.com/)
- [OpenFootball JSON](https://github.com/openfootball/football.json)

---

This hybrid plan is designed for real-world reliability, adaptability, and ease of use, blending the best of both expert approaches. It is ready for production and future-proofed for evolving needs.

---

## Example Config (config.yaml)
```yaml
api_key: "YOUR_API_KEY"
team_id: 57  # Arsenal’s ID in football-data.org
notify_time: "18:00"  # 6pm the night before
notify_method: ["ntfy", "email"]
user_timezone: "America/New_York"
notifications:
  - service: ntfy
    url: "ntfy://ntfy.sh/arsenal-fixtures-username"
    enabled: true
  - service: email
    url: "mailto://user:password@gmail.com?to=your@email.com"
    enabled: false
  - service: twilio
    url: "twilio://ACCOUNT_SID:AUTH_TOKEN@FROM_PHONE/TO_PHONE"
    enabled: false
```

## Example Notification
```
🔴⚪ Arsenal Match Tomorrow! ⚽
Arsenal vs Wolves
Premier League (HOME)
Saturday, December 13, 2025 @ 8:00pm EST
US TV: TNT Sports 1, TNT Sports Ultimate
COYG! 🔴⚪
```

---

This hybrid plan is designed for real-world reliability, extensibility, and ease of use, blending the best of both expert approaches. It is ready for production and future-proofed for evolving needs.
