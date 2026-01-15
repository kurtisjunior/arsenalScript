# Arsenal Intelligence Brief - Agent Handoff Documentation

**Project Status:** Infrastructure initialized, 49 Beads tasks created
**Ready for:** Multi-agent implementation starting tomorrow
**Epic ID:** `arsenalScript-vqp`
**Last Updated:** 2026-01-06

---

## 🎯 Project Overview

Transform the Arsenal Match Notifier into a comprehensive **Intelligence Brief** system that provides:

- **Odds Aggregation** - Compare 5+ bookmakers, identify best value
- **Injury/Lineup Intelligence** - Reddit + Twitter scraping for team news
- **ML Predictions** - Machine learning win probability with confidence
- **NLP Sentiment Analysis** - News and fan sentiment analysis
- **Value Betting Opportunities** - Expected value calculations

**Development Strategy:** Multi-agent collaboration
- **Agent 1:** Data Collection Specialist
- **Agent 2:** Analysis & Reporting Specialist

---

## 📋 Quick Start Checklist

Before you start coding tomorrow, verify these are complete:

- [x] Beads installed and initialized
- [x] 49 tasks created in Beads (1 epic + 48 tasks)
- [ ] MCP Agent Mail installed and running
- [ ] Python dependencies installed
- [ ] Directory structure created
- [ ] Agent mailboxes configured
- [ ] Git working tree clean

---

## 🚀 Step-by-Step Setup Instructions

### Step 1: Install Beads (Task Management)

**Status:** ✅ COMPLETE

Beads is already installed. Verify it's working:

```bash
cd /Users/kurtis/tinker/arsenalScript

# Check beads is installed
bd --version

# List all tasks
bd list

# View task hierarchy
bd ready

# View dependency graph (if beads graph is available)
bd graph
```

---

### Step 2: Install MCP Agent Mail (Local Server)

**Status:** ⏳ TODO

MCP Agent Mail provides local message passing between agents.

**Install via one-liner:**
```bash
curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/mcp_agent_mail/main/scripts/install.sh?$(date +%s)" | bash -s -- --yes
```

**Or via NPM:**
```bash
npx @anthropic/mcp-agent-mail
```

**Start the server:**
```bash
# Start on localhost:8765
uv run python -m mcp_agent_mail.cli serve-http --host 127.0.0.1 --port 8765
```

**Verify it's running:**
```bash
# Check the server is up
curl http://localhost:8765/health

# Or open in browser
open http://localhost:8765
```

**Expected:** You should see the MCP Agent Mail web UI

---

### Step 3: Create Project Directory Structure

**Status:** ⏳ TODO
**Beads Task:** `arsenalScript-vqp.3`

```bash
cd /Users/kurtis/tinker/arsenalScript

# Create all directories
mkdir -p data_collection
mkdir -p analysis
mkdir -p reporting/templates
mkdir -p data/{odds,lineups,news,schemas}
mkdir -p models
mkdir -p tests
mkdir -p logs

# Create Python __init__.py files
touch data_collection/__init__.py
touch analysis/__init__.py
touch reporting/__init__.py

# Verify structure
tree -L 2 -I 'venv|__pycache__|*.pyc|node_modules'
```

**Expected directory structure:**
```
arsenalScript/
├── .beads/                    # Beads task database ✅
├── data/                      # Runtime data storage
│   ├── odds/
│   ├── lineups/
│   ├── news/
│   └── schemas/
├── data_collection/           # Agent 1's domain
│   ├── __init__.py
│   ├── odds_fetcher.py       (to be created)
│   ├── lineup_scraper.py     (to be created)
│   └── news_scraper.py       (to be created)
├── analysis/                  # Agent 2's domain
│   ├── __init__.py
│   ├── ml_predictor.py       (to be created)
│   ├── sentiment_analyzer.py (to be created)
│   └── odds_analyzer.py      (to be created)
├── reporting/                 # Collaborative
│   ├── __init__.py
│   ├── report_builder.py     (to be created)
│   ├── notifier.py           (to be created)
│   └── templates/
│       └── intelligence_brief.html (to be created)
├── models/                    # ML model storage
├── tests/
├── logs/
├── main.py                    # Existing notifier ✅
├── orchestrator.py            # NEW: Main pipeline (to be created)
└── requirements.txt           # Update needed
```

---

### Step 4: Update Requirements.txt

**Status:** ⏳ TODO
**Beads Task:** `arsenalScript-vqp.4`

```bash
cd /Users/kurtis/tinker/arsenalScript

# Append new dependencies
cat >> requirements.txt << 'EOF'

# Data Collection (Agent 1)
beautifulsoup4==4.12.3
praw==7.7.1                    # Reddit API
lxml==5.1.0                    # XML/HTML parsing

# Analysis (Agent 2)
pandas==2.2.0
scikit-learn==1.4.0
numpy==1.26.3

# Reporting
jinja2==3.1.3

# Development
pytest==7.4.4
EOF
```

**Install dependencies:**
```bash
# Activate venv if you have one
source venv/bin/activate  # or: python3 -m venv venv && source venv/bin/activate

# Install
pip install -r requirements.txt

# Verify
pip list | grep -E 'pandas|scikit|praw|beautiful'
```

---

### Step 5: Update .gitignore

**Status:** ⏳ TODO
**Beads Task:** `arsenalScript-vqp.5`

```bash
cat >> .gitignore << 'EOF'

# Multi-agent infrastructure
.beads/cache/
data/
models/*.pkl
logs/

# Agent mail
mailbox/
EOF
```

---

### Step 6: Configure MCP Agent Mail in Claude Code (Optional)

If you're using Claude Code, add to MCP settings:

**File:** `~/.config/claude-code/mcp_config.json`

```json
{
  "mcpServers": {
    "agent-mail": {
      "command": "npx",
      "args": ["@anthropic/mcp-agent-mail"]
    }
  }
}
```

---

### Step 7: Create Agent Mailboxes

**In the MCP Agent Mail web UI** (`http://localhost:8765`):

1. Create mailbox: **agent1-data-collection**
2. Create mailbox: **agent2-analysis**

Or use CLI/API if available (check MCP Agent Mail docs).

---

### Step 8: View All Beads Tasks

```bash
# List all tasks
bd list

# Filter by agent
bd list --assignee agent1
bd list --assignee agent2

# View tasks ready to work on (no blockers)
bd ready

# View specific task details
bd show arsenalScript-vqp.6  # Example: Research betting APIs
```

---

## 👥 Agent Role Assignments

### Agent 1: Data Collection Specialist

**Responsibilities:**
- Fetch data from external APIs and web sources
- Normalize and store in JSON schemas
- Handle rate limiting and errors

**Tools & Libraries:**
- requests, BeautifulSoup, praw (Reddit), lxml
- The Odds API, Reddit API, web scraping

**Assigned Tasks (by label):**
```bash
bd list --labels agent1
```

**Key Tasks:**
- Odds fetching: `arsenalScript-vqp.6` through `arsenalScript-vqp.10`
- Lineup scraping: `arsenalScript-vqp.11` through `arsenalScript-vqp.16`
- News scraping: `arsenalScript-vqp.17` through `arsenalScript-vqp.21`

**Output:** Normalized JSON files in `data/{odds,lineups,news}/`

---

### Agent 2: Analysis & Reporting Specialist

**Responsibilities:**
- Analyze data with ML and NLP
- Generate predictions and insights
- Build intelligence brief reports

**Tools & Libraries:**
- pandas, scikit-learn, numpy (for ML)
- jinja2 (for HTML templates)

**Assigned Tasks (by label):**
```bash
bd list --labels agent2
```

**Key Tasks:**
- ML predictor: `arsenalScript-vqp.22` through `arsenalScript-vqp.27`
- NLP sentiment: `arsenalScript-vqp.28` through `arsenalScript-vqp.32`
- Value analyzer: `arsenalScript-vqp.33` through `arsenalScript-vqp.36`

**Dependencies:** Blocked until Agent 1 completes data schemas

---

## 🔄 Multi-Agent Workflow

### Daily Agent Routine

**Agent 1 (Data Collection):**
```bash
# 1. Check what's ready to work on
bd ready --assignee agent1

# 2. Start a task
bd start arsenalScript-vqp.6  # Example: Research betting APIs

# 3. Do the work...
# Create files, write code, test

# 4. When done, mark complete
bd complete arsenalScript-vqp.6

# 5. Commit to git
git add data_collection/odds_fetcher.py
git commit -m "Implement odds fetcher with rate limiting"
git push

# 6. Notify Agent 2 via MCP Agent Mail
# (Send message that data schema is ready)

# 7. Get next task
bd ready --assignee agent1
```

**Agent 2 (Analysis):**
```bash
# 1. Check what's ready (may be blocked initially)
bd ready --assignee agent2

# 2. Wait for Agent 1 to complete data schemas
# Check messages in MCP Agent Mail for notifications

# 3. Once unblocked, start analysis tasks
bd start arsenalScript-vqp.22  # Collect historical match data

# 4. Complete workflow same as Agent 1
```

---

### Communication Protocol

**Use MCP Agent Mail for:**
- Agent 1 completes data schema → Email Agent 2
- Agent 2 needs clarification → Email Agent 1
- Either agent encounters blocker → Email project lead
- Daily status reports

**Example message format:**
```
From: agent1-data-collection
To: agent2-analysis
Subject: Odds data schema ready

The odds data schema is complete and documented at:
data/schemas/odds.json

You can now start implementing analysis/odds_analyzer.py

Data format includes:
- Bookmaker odds for all markets
- Best value identification
- Timestamp and match_id

Sample data available at: data/odds/arsenal-vs-man-city-2025-12-14.json
```

---

## 📊 Beads Task Breakdown

### Phase 1: Infrastructure (5 tasks) - Priority P0

All agents should complete these first:

1. `arsenalScript-vqp.1` - Install Beads ✅
2. `arsenalScript-vqp.2` - Install MCP Agent Mail ⏳
3. `arsenalScript-vqp.3` - Create directories ⏳
4. `arsenalScript-vqp.4` - Update requirements.txt ⏳
5. `arsenalScript-vqp.5` - Update .gitignore ⏳

---

### Phase 2: Data Collection (16 tasks) - Agent 1

**Odds Module (5 tasks):**
- `vqp.6` - Research betting APIs
- `vqp.7` - Implement odds_fetcher.py
- `vqp.8` - Normalize odds formats
- `vqp.9` - Cross-bookmaker comparison
- `vqp.10` - **Create odds schema** (blocks Agent 2)

**Lineup Module (6 tasks):**
- `vqp.11` - Reddit API setup
- `vqp.12` - Scrape r/Gunners
- `vqp.13` - Twitter scraping
- `vqp.14` - Parse injury keywords
- `vqp.15` - Extract lineups
- `vqp.16` - **Create lineup schema** (blocks Agent 2)

**News Module (5 tasks):**
- `vqp.17` - Scrape Arsenal.com
- `vqp.18` - Scrape BBC Sport
- `vqp.19` - Extract quotes
- `vqp.20` - Store full text
- `vqp.21` - **Create news schema** (blocks Agent 2)

---

### Phase 3: Analysis (15 tasks) - Agent 2

**ML Prediction (6 tasks):**
- `vqp.22` - Collect historical data
- `vqp.23` - Feature engineering
- `vqp.24` - Train model
- `vqp.25` - Validate model
- `vqp.26` - Generate predictions
- `vqp.27` - Save model

**NLP Sentiment (5 tasks):**
- `vqp.28` - Setup transformers
- `vqp.29` - Analyze news sentiment
- `vqp.30` - Analyze Reddit sentiment
- `vqp.31` - Extract themes
- `vqp.32` - Generate summary

**Value Analyzer (4 tasks):**
- `vqp.33` - Convert to probabilities
- `vqp.34` - Compare with ML
- `vqp.35` - Calculate expected value
- `vqp.36` - Flag opportunities

---

### Phase 4: Report Generation (9 tasks) - Collaborative

**Report Building (5 tasks):**
- `vqp.37` - Design structure
- `vqp.38` - Create HTML template
- `vqp.39` - Aggregate data
- `vqp.40` - Generate charts
- `vqp.41` - Format for email

**Notification (4 tasks):**
- `vqp.42` - HTML email sender
- `vqp.43` - Optional: Telegram bot
- `vqp.44` - Inline charts
- `vqp.45` - Test deliverability

---

### Phase 5: Integration (4 tasks) - Collaborative

- `vqp.46` - Integration tests
- `vqp.47` - Create orchestrator.py
- `vqp.48` - Update GitHub Actions
- `vqp.49` - Documentation

---

## 🔗 Critical Dependencies

**Agent 2 is blocked until Agent 1 completes:**

1. **Odds schema** (`vqp.10`) - Required for `vqp.33` (odds analyzer)
2. **Lineup schema** (`vqp.16`) - Required for ML features
3. **News schema** (`vqp.21`) - Required for `vqp.29` (sentiment)

**To unblock Agent 2:**
```bash
# Agent 1 must complete these three schema tasks first
bd complete arsenalScript-vqp.10  # Odds schema
bd complete arsenalScript-vqp.16  # Lineup schema
bd complete arsenalScript-vqp.21  # News schema

# Then notify Agent 2 via MCP Agent Mail
```

---

## 🧪 Testing the Setup

Run these commands to verify everything is ready:

```bash
# 1. Beads is working
bd list | head
# Should show: arsenalScript-vqp (epic) + 49 tasks

# 2. MCP Agent Mail is running
curl http://localhost:8765/health
# Should return: 200 OK or similar

# 3. Directory structure exists
ls -la data_collection/ analysis/ reporting/
# Should show: __init__.py in each

# 4. Python dependencies installed
python -c "import pandas, sklearn, praw, bs4; print('✅ All deps installed')"
# Should print: ✅ All deps installed

# 5. Git status is clean
git status
# Should show: nothing to commit (except .beads/)
```

---

## 📚 Reference Documentation

### Beads Commands Cheat Sheet

```bash
# List all tasks
bd list

# Show ready tasks (no blockers)
bd ready

# Show tasks for specific agent
bd ready --assignee agent1
bd ready --assignee agent2

# Start working on a task
bd start arsenalScript-vqp.6

# Mark task complete
bd complete arsenalScript-vqp.6

# Show task details
bd show arsenalScript-vqp.6

# Update task
bd update arsenalScript-vqp.6 --description "New description"

# Add dependency
bd update arsenalScript-vqp.34 --deps blocks:arsenalScript-vqp.10

# Search tasks
bd list --labels agent1
bd list --labels ml
bd list --labels schema
```

---

### Data Schemas (To Be Created)

**Odds Schema** (`data/schemas/odds.json`):
```json
{
  "match_id": "string",
  "timestamp": "ISO8601",
  "bookmakers": [
    {
      "name": "string",
      "odds": {
        "arsenal_win": "decimal",
        "draw": "decimal",
        "opponent_win": "decimal"
      }
    }
  ],
  "best_value": {
    "arsenal_win": {"bookmaker": "string", "odds": "decimal"}
  }
}
```

**Lineup Schema** (`data/schemas/lineups.json`):
```json
{
  "match_id": "string",
  "timestamp": "ISO8601",
  "injuries": [
    {
      "player": "string",
      "status": "out|doubt|available",
      "source": "string",
      "confidence": "low|medium|high",
      "details": "string"
    }
  ],
  "rumored_lineup": {
    "formation": "string",
    "players": ["array of names"]
  }
}
```

**News Schema** (`data/schemas/news.json`):
```json
{
  "match_id": "string",
  "timestamp": "ISO8601",
  "articles": [
    {
      "title": "string",
      "source": "string",
      "url": "string",
      "publish_date": "ISO8601",
      "full_text": "string",
      "quotes": [
        {
          "speaker": "string",
          "text": "string"
        }
      ]
    }
  ]
}
```

---

## 🎯 Success Metrics

**Week 1 Goals:**
- [ ] Infrastructure complete (all 5 Phase 1 tasks)
- [ ] Agent 1 completes all data schemas
- [ ] First odds data fetched and stored

**Week 2 Goals:**
- [ ] Agent 2 ML model trained
- [ ] Agent 2 sentiment analysis working
- [ ] Data flowing through full pipeline

**Week 3 Goals:**
- [ ] First intelligence brief generated
- [ ] HTML email template complete
- [ ] Integration tests passing

**Week 4 Goals:**
- [ ] GitHub Actions automated
- [ ] Documentation complete
- [ ] System running end-to-end

**Final Deliverable:**
Automated intelligence brief sent 24h before each Arsenal match with:
- Best odds across 5+ bookmakers ✅
- Injury/lineup intelligence ✅
- ML match prediction ✅
- News sentiment analysis ✅
- Value betting opportunities ✅

---

## 🚨 Troubleshooting

### Beads Issues

**Problem:** `bd: command not found`
```bash
# Reinstall Beads
npm install -g @beads/bd

# Verify installation
which bd
bd --version
```

**Problem:** Tasks not showing up
```bash
# Re-sync database
bd sync

# Check database
ls -la .beads/
```

---

### MCP Agent Mail Issues

**Problem:** Server won't start
```bash
# Check if port 8765 is in use
lsof -i :8765

# Try different port
uv run python -m mcp_agent_mail.cli serve-http --host 127.0.0.1 --port 8766
```

**Problem:** Can't access web UI
```bash
# Check server is running
curl http://localhost:8765

# Check firewall settings
```

---

### Python Dependency Issues

**Problem:** Import errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check virtual environment is activated
which python
# Should be: /path/to/venv/bin/python
```

---

## 🔐 Security & API Keys

**Required API Keys (get before starting):**

1. **Football Data API** (existing)
   - Already configured in `FOOTBALL_DATA_API_KEY`
   - Used by: existing `main.py`

2. **The Odds API** (new - free tier)
   - Sign up: https://the-odds-api.com/
   - Free tier: 500 requests/month
   - Add to secrets: `ODDS_API_KEY`

3. **Reddit API** (new - free)
   - Create app: https://www.reddit.com/prefs/apps
   - Get: client_id, client_secret
   - Add to secrets: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`

4. **Twitter API** (optional - or use nitter scraping)
   - If using tweepy: Get API keys from Twitter Developer Portal
   - Alternative: Use nitter.net for scraping (no API key needed)

**Storing Secrets:**

**Local development:**
```bash
# Create .env file (already in .gitignore)
cat > .env << 'EOF'
FOOTBALL_DATA_API_KEY=your_existing_key
ODDS_API_KEY=your_new_key
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret
EOF
```

**GitHub Actions:**
Add to: Repository → Settings → Secrets and variables → Actions

---

## 📞 Communication Channels

**For tomorrow's agent(s):**

1. **Check Beads first:**
   ```bash
   bd ready --assignee agent1  # or agent2
   ```

2. **Check MCP Agent Mail:**
   - Web UI: http://localhost:8765
   - Look for messages from other agents

3. **Check this file:**
   - `AGENT_HANDOFF.md` (you're reading it!)

4. **Check the plan:**
   - `/Users/kurtis/.claude/plans/purring-stargazing-naur.md`

---

## 🎬 Getting Started Tomorrow

**For Agent 1 (Data Collection):**
```bash
# 1. Complete infrastructure tasks
bd start arsenalScript-vqp.2  # Install MCP Agent Mail
bd start arsenalScript-vqp.3  # Create directories
bd start arsenalScript-vqp.4  # Update requirements
bd start arsenalScript-vqp.5  # Update .gitignore

# 2. Start data collection
bd start arsenalScript-vqp.6  # Research betting APIs
# Continue with odds module...
```

**For Agent 2 (Analysis):**
```bash
# 1. Help with infrastructure if needed
bd ready --assignee agent2

# 2. Prepare while waiting for Agent 1
# - Research ML libraries
# - Design model architecture
# - Plan feature engineering

# 3. Monitor for schema completion
# Check MCP Agent Mail for notifications from Agent 1

# 4. Start analysis when unblocked
bd start arsenalScript-vqp.22  # Collect historical data
```

---

## ✅ Pre-Flight Checklist

Before starting implementation tomorrow:

**Infrastructure:**
- [ ] Beads installed: `bd --version` works
- [ ] MCP Agent Mail running: `curl localhost:8765` returns OK
- [ ] Directories created: `ls data_collection/ analysis/ reporting/`
- [ ] Dependencies installed: `pip list | grep pandas`
- [ ] .gitignore updated
- [ ] Git working tree clean

**API Access:**
- [ ] Football Data API key ready
- [ ] The Odds API key obtained
- [ ] Reddit API credentials created
- [ ] All secrets added to .env file

**Agent Coordination:**
- [ ] Agent mailboxes created in MCP Agent Mail
- [ ] Both agents understand their roles
- [ ] Communication protocol clear

**Documentation:**
- [ ] Read this file (AGENT_HANDOFF.md)
- [ ] Review implementation plan
- [ ] Understand data schemas

---

## 🎉 You're Ready!

All Beads tasks are created and waiting for you. Tomorrow, just run:

```bash
cd /Users/kurtis/tinker/arsenalScript
bd ready --assignee agent1  # (or agent2)
```

And start building! 🚀

**Good luck and COYG!** 🔴⚪

---

**Epic:** `arsenalScript-vqp`
**Total Tasks:** 49 (1 epic + 48 tasks)
**Status:** Ready for multi-agent implementation
**Next Action:** Complete Phase 1 infrastructure tasks
