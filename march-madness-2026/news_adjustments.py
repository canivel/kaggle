#!/usr/bin/env python3
"""
News-based adjustments for 2026 March Madness predictions.
Incorporates injuries, suspensions, momentum, and betting odds.

These factors are NOT captured by historical statistics and can significantly
move win probabilities.
"""

import numpy as np

# ============================================================
# INJURY / SUSPENSION ADJUSTMENTS (March 18, 2026)
# Negative = team is WEAKER than stats suggest
# Values represent estimated Brier-equivalent probability shift
# ============================================================

# TeamID -> adjustment_factor (multiply win probability by this)
# < 1.0 = team weakened, > 1.0 = team strengthened
MEN_ADJUSTMENTS = {
    # === MAJOR INJURIES / SUSPENSIONS ===
    1104: 0.82,  # Alabama (#4) - Aden Holloway suspended (2nd-leading scorer, felony arrest)
    1181: 0.92,  # Duke (#1) - Caleb Foster broken foot (key player, but deep roster)
    1314: 0.85,  # North Carolina (#6) - Caleb Wilson season-ending thumb surgery
    1140: 0.85,  # BYU (#6) - Richie Saunders season-ending knee injury
    1211: 0.88,  # Gonzaga (#3) - Braden Huff out since Jan (17.8ppg, 5.6rpg)
    1417: 0.90,  # UCLA (#7) - Multiple minor injuries (knee strain, calf strain)

    # === NEGATIVE MOMENTUM ===
    1242: 0.92,  # Kansas (#4) - Lost 5 of last 9 games

    # === POSITIVE MOMENTUM ===
    1345: 1.08,  # Purdue (#2) - Won Big Ten Tournament, moved up to #2 seed
    1112: 1.05,  # Arizona (#1) - Won Big 12 Tournament, considered healthiest favorite
    1116: 1.05,  # Arkansas (#4) - Won SEC Tournament
    1385: 1.05,  # St John's (#5) - Won Big East Tournament over UConn
    1219: 1.10,  # High Point (#12) - 30-4, haven't lost in 2 months, 90ppg offense
    1320: 1.05,  # Northern Iowa (#12) - Won MVC Tournament, elite defense

    # === CINDERELLA / UNDERDOG SIGNALS ===
    1465: 1.08,  # Cal Baptist (#13) - First tournament, Daniels Jr. averaging 23.2ppg, 32ppg last 3
    1218: 1.05,  # Hawaii (#13) - Won Big West, 24-8, 7-foot center

    # === EXPERT CONSENSUS: SAFEST PICK ===
    # Arizona considered "safest bet" by experts due to health + balance
}

# Championship odds (implied probability from American odds)
# Source: Vegas odds March 2026
CHAMPIONSHIP_ODDS = {
    # Men's - updated from multiple sportsbooks March 18, 2026
    1181: 0.23,   # Duke +330 (32-2, #1 overall)
    1276: 0.21,   # Michigan +370 (31-3)
    1112: 0.19,   # Arizona +425 (32-2, healthiest)
    1196: 0.12,   # Florida +750 (26-7)
    1222: 0.09,   # Houston +1000 (28-6)
    1163: 0.055,  # UConn +1700 (29-5)
    1228: 0.05,   # Illinois +1900 (24-8, #1 offense)
    1235: 0.043,  # Iowa St +2200 (27-7)
    1345: 0.028,  # Purdue +3500 (27-8, won BTT)
    1277: 0.024,  # Michigan St +4000 (25-7)
    1116: 0.020,  # Arkansas +5000 (26-8, won SEC)
    1242: 0.020,  # Kansas +5000 (23-10, cold streak)
    1385: 0.020,  # St John's +5000 (28-6)
    1211: 0.017,  # Gonzaga +5500 (30-3, Huff injured)
    1438: 0.015,  # Virginia +6000 (29-5)
    1435: 0.014,  # Vanderbilt +6500 (26-8)
    1458: 0.013,  # Wisconsin +7000 (24-10)
    1397: 0.010,  # Tennessee +9000 (22-11)
    1304: 0.010,  # Nebraska +10000 (26-6)
    1246: 0.010,  # Kentucky +10000 (21-13)
    1104: 0.010,  # Alabama +10000 (23-9, Holloway OUT)
    1314: 0.009,  # North Carolina +11000 (24-8, Wilson OUT)
    1403: 0.009,  # Texas Tech +11000 (22-10)
    1257: 0.008,  # Louisville +12000 (23-10)
    1140: 0.007,  # BYU +13000 (23-11, Saunders OUT)
    1417: 0.007,  # UCLA +13000 (23-11, injuries)
}

# KenPom-inspired power ratings (relative to average D1 team)
# Higher = stronger. Average D1 team = 0.0
# Derived from expert analysis + KenPom efficiency
POWER_RATINGS = {
    # KenPom-derived power ratings (March 18, 2026)
    # Scale: ~39 = elite, ~10 = tournament floor, ~0 = average D1
    # Adjusted for injuries/news
    1181: 38.9,  # Duke (32-2, KenPom #1, #4 off #2 def)
    1112: 37.7,  # Arizona (32-2, KenPom #2, healthiest #1 seed)
    1276: 37.6,  # Michigan (31-3, KenPom #3, #1 defense)
    1196: 32.0,  # Florida (26-7)
    1222: 31.0,  # Houston (28-6)
    1163: 29.0,  # UConn (29-5)
    1235: 28.0,  # Iowa St (27-7)
    1228: 27.5,  # Illinois (24-8, #1 offense)
    1345: 27.0,  # Purdue (27-8, won BTT)
    1277: 26.0,  # Michigan St (25-7)
    1438: 25.5,  # Virginia (29-5)
    1211: 24.0,  # Gonzaga (30-3, but Huff out -> downgraded from ~27)
    1116: 24.0,  # Arkansas (26-8, won SEC)
    1304: 23.5,  # Nebraska (26-6)
    1435: 23.0,  # Vanderbilt (26-8)
    1385: 23.0,  # St John's (28-6, won Big East)
    1458: 22.5,  # Wisconsin (24-10)
    1403: 22.0,  # Texas Tech (22-10)
    1104: 21.5,  # Alabama (23-9, downgraded: Holloway suspended)
    1397: 21.5,  # Tennessee (22-11)
    1246: 21.0,  # Kentucky (21-13)
    1242: 21.0,  # Kansas (23-10, cold: lost 5 of last 9)
    1257: 20.5,  # Louisville (23-10)
    1314: 19.5,  # North Carolina (24-8, downgraded: Wilson out)
    1155: 20.0,  # Clemson (24-10)
    1417: 19.5,  # UCLA (23-11, minor injuries)
    1274: 19.5,  # Miami FL (25-8)
    1140: 18.0,  # BYU (23-11, downgraded: Saunders out)
    1437: 19.0,  # Villanova (24-8)
    1326: 18.5,  # Ohio St (21-12)
    1208: 18.5,  # Georgia (22-10)
    1234: 18.0,  # Iowa (21-12)
    1401: 18.0,  # Texas A&M (21-11)
    1395: 17.5,  # TCU (22-11)
    1429: 17.5,  # Utah St (28-6)
    1388: 17.0,  # St Mary's CA (27-5)
    1281: 16.5,  # Missouri (20-12)
    1387: 16.5,  # St Louis (28-5)
    1365: 16.0,  # Santa Clara (26-8)
    1416: 15.5,  # UCF (21-11)
    1433: 15.0,  # VCU (27-7)
    1378: 14.5,  # South Florida (25-8)
    1270: 14.5,  # McNeese St (28-5)
    1219: 14.5,  # High Point (30-4, 90ppg, 2mo unbeaten)
    1320: 14.0,  # Northern Iowa (23-12, elite defense)
    1103: 13.5,  # Akron (29-5)
    1275: 13.5,  # Miami OH (31-1!)
    1465: 13.5,  # Cal Baptist (25-8, Daniels 32ppg last 3)
    1218: 13.0,  # Hawaii (24-8)
    1301: 12.5,  # NC State (20-13)
    1400: 12.5,  # Texas (18-14)
    1374: 12.5,  # SMU (20-13)
    1220: 12.0,  # Hofstra (24-10)
    1407: 11.5,  # Troy (22-11)
    1335: 11.5,  # Penn (18-11)
    1295: 10.5,  # N Dakota St (27-7)
    1460: 10.0,  # Wright St (23-11)
    1244: 10.0,  # Kennesaw (21-13)
    1398: 9.5,   # Tennessee St (23-9)
    1202: 9.5,   # Furman (22-12)
    1225: 9.0,   # Idaho (21-14)
    1474: 8.5,   # Queens NC (21-13)
    1373: 8.0,   # Siena (23-11)
    1254: 7.5,   # LIU Brooklyn (24-10)
    1224: 7.0,   # Howard (23-10)
    1420: 7.0,   # UMBC (24-8)
    1250: 6.5,   # Lehigh (18-16)
    1341: 5.0,   # Prairie View (18-17)
}


def get_adjustment(team_id):
    """Get the news-based adjustment factor for a team. 1.0 = no adjustment."""
    return MEN_ADJUSTMENTS.get(team_id, 1.0)


def get_power_rating(team_id):
    """Get estimated power rating. Returns None if unknown."""
    return POWER_RATINGS.get(team_id, None)


def adjust_prediction(pred, team1_id, team2_id):
    """
    Adjust a model prediction based on news factors.
    pred = P(team1 wins) from the model.
    Returns adjusted probability.
    """
    adj1 = get_adjustment(team1_id)
    adj2 = get_adjustment(team2_id)

    if adj1 == 1.0 and adj2 == 1.0:
        return pred

    # Convert to log-odds, apply adjustments, convert back
    # Adjustment shifts log-odds proportionally
    eps = 1e-7
    pred = np.clip(pred, eps, 1 - eps)
    log_odds = np.log(pred / (1 - pred))

    # Adjustment factor in log-odds space
    # adj < 1 means team is weaker -> reduce their log-odds contribution
    shift = np.log(adj1) - np.log(adj2)
    adjusted_log_odds = log_odds + shift

    adjusted_pred = 1 / (1 + np.exp(-adjusted_log_odds))
    return np.clip(adjusted_pred, 0.01, 0.99)


def odds_based_prediction(team1_id, team2_id):
    """
    Generate a prediction purely from power ratings / championship odds.
    Uses Bradley-Terry model: P(A beats B) = rating_A / (rating_A + rating_B)
    Returns None if we don't have ratings for both teams.
    """
    r1 = POWER_RATINGS.get(team1_id)
    r2 = POWER_RATINGS.get(team2_id)

    if r1 is not None and r2 is not None:
        # Convert power ratings to win probability using logistic function
        # Calibrated so 10-point difference ≈ 75% win probability
        diff = r1 - r2
        return 1 / (1 + np.exp(-diff / 6.0))

    return None


def blend_with_odds(model_pred, team1_id, team2_id, model_weight=0.70, odds_weight=0.30):
    """
    Blend model prediction with odds-based prediction.
    If no odds available, return adjusted model prediction.
    """
    # First apply injury/news adjustments to model prediction
    adjusted = adjust_prediction(model_pred, team1_id, team2_id)

    # Then blend with odds-based prediction
    odds_pred = odds_based_prediction(team1_id, team2_id)

    if odds_pred is not None:
        blended = model_weight * adjusted + odds_weight * odds_pred
    else:
        blended = adjusted

    return np.clip(blended, 0.01, 0.99)
