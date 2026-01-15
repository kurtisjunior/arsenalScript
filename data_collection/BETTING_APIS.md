# Betting APIs Research - Arsenal Intelligence Brief

## Recommended APIs for Integration

### 1. The Odds API (RECOMMENDED - Primary)
**Website**: https://the-odds-api.com/

**Why Recommended**:
- Free tier: 500 requests/month (sufficient for daily checks)
- Covers 40+ bookmakers including US sportsbooks
- Premier League coverage excellent
- Clean JSON response format
- No authentication complexity (just API key)

**Rate Limits**:
- Free tier: 500 requests/month
- Paid plans: Start at $15/month for 10,000 requests

**Markets Available**:
- h2h (1X2 match result)
- spreads (Asian handicap)
- totals (over/under goals)

**Sample Endpoint**:
```
GET https://api.the-odds-api.com/v4/sports/soccer_epl/odds/
?apiKey=YOUR_KEY
&regions=us,uk
&markets=h2h,totals
```

**Integration Notes**:
- Returns odds in decimal format by default
- Can request American or fractional via `oddsFormat` param
- Bookmaker keys are consistent across requests

---

### 2. Betfair Exchange API (Optional - Advanced)
**Website**: https://developer.betfair.com/

**Why Consider**:
- Exchange odds (often better value)
- Real-time data
- Historical odds data available

**Limitations**:
- Requires Betfair account with positive balance
- Complex authentication (certificate-based)
- UK/EU focused, limited US availability

**Rate Limits**:
- 20 requests/second for most endpoints
- Some endpoints have stricter limits

**Best For**:
- Detecting sharp money movements
- Exchange vs bookmaker arbitrage analysis

---

### 3. OddsJam API (Premium Alternative)
**Website**: https://oddsjam.com/api

**Why Consider**:
- US market focused
- Real-time odds updates
- Positive EV bet detection built-in

**Limitations**:
- No free tier
- Starts at $99/month
- Overkill for this project

**Best For**:
- Serious bettors needing fastest updates
- Projects requiring live odds streaming

---

### 4. Football-Data.org (Already Used - Fixture Data Only)
**Website**: https://www.football-data.org/

**Current Use**: Fixture information
**Odds Available**: Limited/None
**Note**: Keep using for fixtures, supplement with The Odds API for betting data

---

## API Selection for Arsenal Intelligence Brief

### Primary API: The Odds API
- **Reason**: Best free tier, excellent Premier League coverage
- **Implementation**: `data_collection/odds_fetcher.py`
- **Frequency**: Once daily (to conserve quota)
- **Markets**: h2h, totals

### Secondary API: None required initially
- The free tier of The Odds API is sufficient
- Consider Betfair for future enhancement if exchange odds desired

---

## Implementation Recommendations

### Rate Limiting Strategy
```python
# Recommended rate limiting for free tier (500 req/month)
MAX_DAILY_REQUESTS = 15  # ~450/month, leaves buffer
REQUEST_INTERVAL_SECONDS = 5  # Between consecutive calls
CACHE_DURATION_HOURS = 6  # Re-use cached data within window
```

### Error Handling
- Handle 401 (invalid API key)
- Handle 429 (rate limit exceeded)
- Handle 404 (event not found)
- Cache last successful response as fallback

### Data Storage
- Save to `data/odds/{match_id}.json`
- Follow schema in `data/schemas/odds.json`
- Include fetch timestamp for cache validation

---

## Environment Variables Required

```bash
# Add to .env file
THE_ODDS_API_KEY=your_api_key_here
```

## API Registration Links

1. **The Odds API**: https://the-odds-api.com/ (sign up for free tier)
2. **Betfair**: https://developer.betfair.com/ (requires Betfair account)
3. **OddsJam**: https://oddsjam.com/api (paid only)

---

## Next Steps for Implementation (Task vqp.7)

1. Create `data_collection/odds_fetcher.py`
2. Implement `TheOddsAPIClient` class
3. Add rate limiting with token bucket algorithm
4. Add caching layer to minimize API calls
5. Integrate with `OddsData` class from `odds_data.py`

---

*Research completed: 2026-01-15*
*Agent: Agent 1 (DataCollector)*
