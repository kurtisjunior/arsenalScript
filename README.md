# Arsenal Match Notifier 🔴⚪

Automated push notifications for Arsenal matches the night before kickoff, with USA TV channel information.

## Features

- 🔔 **Push notifications** to your phone via ntfy.sh
- 📅 **Automatic scheduling** - runs daily, notifies 18 hours before matches
- 📺 **USA TV channels** - NBC, Peacock, ESPN+, Paramount+, etc.
- ⚽ **All competitions** - Premier League, Champions League, FA Cup, EFL Cup
- 🆓 **100% FREE** - No monthly costs
- 🤖 **Zero maintenance** - Runs automatically via GitHub Actions

## Quick Start (20 minutes)

### 1. Get Football API Key (2 minutes)

1. Visit [football-data.org](https://www.football-data.org/client/register)
2. Register for a free account
3. Copy your API key

### 2. Setup ntfy.sh Notifications (2 minutes)

1. Install the [ntfy app](https://ntfy.sh/) on your phone (iOS/Android)
2. Open the app and subscribe to a topic (e.g., `arsenal-john-2025`)
3. Remember your topic name for later

### 3. Fork and Clone This Repository (3 minutes)

```bash
# Fork this repo on GitHub first, then:
git clone https://github.com/YOUR-USERNAME/arsenalScript.git
cd arsenalScript
```

### 4. Add GitHub Secrets (5 minutes)

1. Go to your forked repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** and add:

**Secret 1:**
- Name: `FOOTBALL_DATA_API_KEY`
- Value: Your API key from step 1

**Secret 2:**
- Name: `NOTIFY_URL`
- Value: `ntfy://ntfy.sh/arsenal-YOUR-TOPIC-NAME`
  - Replace `YOUR-TOPIC-NAME` with your topic from step 2
  - Example: `ntfy://ntfy.sh/arsenal-john-2025`

### 5. Enable GitHub Actions (2 minutes)

1. Go to the **Actions** tab in your repository
2. Click "I understand my workflows, go ahead and enable them"
3. The workflow will now run automatically every day at 6 PM EST

### 6. Test It! (3 minutes)

1. Go to **Actions** tab → **Arsenal Match Notifier**
2. Click **Run workflow** → **Run workflow**
3. Wait ~30 seconds
4. Check your phone for a notification!

**Done!** You'll now get push notifications 18 hours before every Arsenal match.

## Local Testing (Optional)

If you want to test the script locally before deploying:

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### Configure .env

Edit `.env` and add your credentials:

```bash
FOOTBALL_DATA_API_KEY=your_api_key_here
NOTIFY_URL=ntfy://ntfy.sh/your-topic-name
USER_TIMEZONE=America/New_York
```

### Run

```bash
python main.py
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  GitHub Actions (runs daily at 6 PM EST)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  main.py fetches next Arsenal match from API               │
│  (football-data.org - Arsenal team ID: 57)                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Check if match is ~18 hours away                          │
│  If yes: format notification with USA TV channel           │
│  If no: exit quietly                                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Send push notification via ntfy.sh                        │
│  Message includes: opponent, time, competition, TV channel │
└─────────────────────────────────────────────────────────────┘
```

## Notification Format

```
🔴⚪ Arsenal Match Tomorrow!

Arsenal vs Manchester City
Premier League - HOME game

📅 Sunday, December 14, 2025
🕐 11:30 AM EST
📺 NBC, USA Network, or Peacock (check NBCSports.com)

COYG! 🔴⚪
```

## USA TV Channel Mapping

| Competition | Broadcasters |
|-------------|--------------|
| Premier League | NBC, USA Network, Peacock |
| Champions League | Paramount+, CBS Sports |
| FA Cup | ESPN+, ESPN, ABC (final) |
| EFL Cup | ESPN+ |

## Configuration

### Change Notification Time

Edit `.github/workflows/notifier.yml`:

```yaml
schedule:
  # Current: 11 PM UTC = 6 PM EST
  - cron: '0 23 * * *'

  # For 7 PM EST: '0 0 * * *'
  # For 8 PM EST: '0 1 * * *'
```

### Change Hours Before Match

Edit `main.py` line 50:

```python
notify_time = match_local - timedelta(hours=18)  # Change 18 to your preference
```

### Change Timezone

Update GitHub secret `USER_TIMEZONE` or edit workflow file:

```yaml
env:
  USER_TIMEZONE: 'America/Los_Angeles'  # Or America/Chicago, etc.
```

## Troubleshooting

### Not receiving notifications?

1. **Check GitHub Actions logs**
   - Go to Actions tab → Latest workflow run → View logs
   - Look for errors or "Not time to notify yet" message

2. **Verify ntfy.sh topic**
   - Make sure you're subscribed to the correct topic in the app
   - Topic name in app must match `NOTIFY_URL` secret

3. **Check API key**
   - Verify `FOOTBALL_DATA_API_KEY` is set correctly in GitHub secrets
   - Test API key: `curl -H "X-Auth-Token: YOUR_KEY" https://api.football-data.org/v4/teams/57/matches`

4. **Test manually**
   - Run workflow manually from Actions tab
   - Check phone immediately after run completes

### API rate limits?

The free tier allows 10 calls/minute. Running once per day uses ~30 calls/month, well within limits.

### Want SMS instead of push?

Replace ntfy.sh with Twilio:

1. Sign up for [Twilio](https://www.twilio.com/)
2. Update `NOTIFY_URL` secret:
   ```
   twilio://ACCOUNT_SID:AUTH_TOKEN@FROM_NUMBER/TO_NUMBER
   ```

Note: Twilio costs ~$0.01/message (~$9/year for 30 matches).

## Project Structure

```
arsenalScript/
├── main.py                    # Main notification script
├── requirements.txt           # Python dependencies
├── .github/
│   └── workflows/
│       └── notifier.yml      # GitHub Actions workflow
├── .env.example              # Environment template
├── .gitignore                # Protect secrets
├── README.md                 # This file
├── ars.py                    # OLD: Web scraping version (deprecated)
└── test_ars.py               # OLD: Tests for legacy version
```

## Migration from Old Version

The old `ars.py` used web scraping, which was:
- ❌ Fragile (broke when website HTML changed)
- ❌ UK-focused (showed UK channels, not USA)
- ❌ Manual (had to run yourself)
- ❌ No notifications (console output only)

The new `main.py` uses a reliable API and:
- ✅ Stable (uses official football API)
- ✅ USA-focused (shows NBC, Peacock, ESPN+, etc.)
- ✅ Automated (runs daily via GitHub Actions)
- ✅ Push notifications (to your phone)

## Cost Breakdown

| Component | Service | Cost |
|-----------|---------|------|
| Deployment | GitHub Actions (public repo) | **$0** |
| API | football-data.org (free tier) | **$0** |
| Notifications | ntfy.sh | **$0** |
| **Total** | | **$0/month** |

## Contributing

Found a bug or want to improve something?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - Feel free to use and modify!

## Support

- For API issues: [football-data.org support](https://www.football-data.org/)
- For ntfy.sh help: [ntfy.sh documentation](https://ntfy.sh/)
- For GitHub Actions: [GitHub Actions docs](https://docs.github.com/en/actions)

---

**COYG!** 🔴⚪
