import numpy as np
import pandas as pd
from pathlib import Path


class FeatureBuilder:

    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self._load_data()

        self._team_season_stats = None
        self._elo_ratings = None
        self._seed_features = None
        self._massey_features = None
        self._coach_features = None
        self._h2h_games = None

    # ------------------------------------------------------------------ IO
    def _read(self, filename, **kwargs):
        return pd.read_csv(self.data_dir / filename, **kwargs)

    def _load_data(self):
        self.m_teams = self._read('MTeams.csv')
        self.w_teams = self._read('WTeams.csv')

        self.m_seasons = self._read('MSeasons.csv')
        self.w_seasons = self._read('WSeasons.csv')

        self.m_tourney_seeds = self._read('MNCAATourneySeeds.csv')
        self.w_tourney_seeds = self._read('WNCAATourneySeeds.csv')

        self.m_reg_compact = self._read('MRegularSeasonCompactResults.csv')
        self.w_reg_compact = self._read('WRegularSeasonCompactResults.csv')

        self.m_reg_detailed = self._read('MRegularSeasonDetailedResults.csv')
        self.w_reg_detailed = self._read('WRegularSeasonDetailedResults.csv')

        self.m_tourney_compact = self._read('MNCAATourneyCompactResults.csv')
        self.w_tourney_compact = self._read('WNCAATourneyCompactResults.csv')

        self.m_tourney_detailed = self._read('MNCAATourneyDetailedResults.csv')
        self.w_tourney_detailed = self._read('WNCAATourneyDetailedResults.csv')

        self.m_coaches = self._read('MTeamCoaches.csv')

        self.conferences = self._read('Conferences.csv')
        self.m_team_conf = self._read('MTeamConferences.csv')
        self.w_team_conf = self._read('WTeamConferences.csv')

    def _combine_mw(self, m_df, w_df):
        m = m_df.copy()
        w = w_df.copy()
        m['Gender'] = 'M'
        w['Gender'] = 'W'
        return pd.concat([m, w], ignore_index=True)

    def _get_h2h_games(self):
        if self._h2h_games is None:
            compact_all = self._combine_mw(self.m_reg_compact, self.w_reg_compact)
            tourney_all = self._combine_mw(self.m_tourney_compact, self.w_tourney_compact)
            self._h2h_games = pd.concat([compact_all, tourney_all], ignore_index=True)
        return self._h2h_games

    # -------------------------------------------------------- team season stats
    def build_team_season_stats(self):
        if self._team_season_stats is not None:
            return self._team_season_stats

        compact = self._combine_mw(self.m_reg_compact, self.w_reg_compact)
        detailed = self._combine_mw(self.m_reg_detailed, self.w_reg_detailed)

        stats = self._compute_compact_stats(compact)
        detail_stats = self._compute_detailed_stats(detailed)
        sos = self._compute_sos(compact)
        conf_strength = self._compute_conference_strength(compact)
        momentum = self._compute_momentum(compact)

        result = stats.merge(detail_stats, on=['Season', 'TeamID', 'Gender'], how='left')
        result = result.merge(sos, on=['Season', 'TeamID', 'Gender'], how='left')
        result = result.merge(conf_strength, on=['Season', 'TeamID', 'Gender'], how='left')
        result = result.merge(momentum, on=['Season', 'TeamID', 'Gender'], how='left')

        self._team_season_stats = result
        return result

    def _compute_compact_stats(self, compact):
        win_rows = compact[['Season', 'WTeamID', 'WScore', 'LScore', 'WLoc', 'Gender']].copy()
        win_rows.columns = ['Season', 'TeamID', 'Score', 'OppScore', 'Loc', 'Gender']
        win_rows['Win'] = 1
        win_rows['IsHome'] = (win_rows['Loc'] == 'H').astype(int)
        win_rows['IsAway'] = (win_rows['Loc'] == 'A').astype(int)
        win_rows['IsNeutral'] = (win_rows['Loc'] == 'N').astype(int)

        loss_rows = compact[['Season', 'LTeamID', 'LScore', 'WScore', 'WLoc', 'Gender']].copy()
        loss_rows.columns = ['Season', 'TeamID', 'Score', 'OppScore', 'WLoc', 'Gender']
        loss_rows['Win'] = 0
        loss_rows['IsHome'] = (loss_rows['WLoc'] == 'A').astype(int)
        loss_rows['IsAway'] = (loss_rows['WLoc'] == 'H').astype(int)
        loss_rows['IsNeutral'] = (loss_rows['WLoc'] == 'N').astype(int)
        loss_rows.drop(columns='WLoc', inplace=True)

        games = pd.concat([win_rows, loss_rows], ignore_index=True)
        games['Margin'] = games['Score'] - games['OppScore']

        grp = games.groupby(['Season', 'TeamID', 'Gender'])

        stats = pd.DataFrame({
            'games_played': grp['Win'].count(),
            'win_rate': grp['Win'].mean(),
            'avg_score': grp['Score'].mean(),
            'avg_allowed': grp['OppScore'].mean(),
            'avg_margin': grp['Margin'].mean(),
        })

        for loc_name, col in [('home', 'IsHome'), ('away', 'IsAway'), ('neutral', 'IsNeutral')]:
            loc_games = games[games[col] == 1]
            loc_wr = loc_games.groupby(['Season', 'TeamID', 'Gender'])['Win'].mean()
            stats[f'{loc_name}_win_rate'] = loc_wr

        stats = stats.reset_index()
        stats[['home_win_rate', 'away_win_rate', 'neutral_win_rate']] = \
            stats[['home_win_rate', 'away_win_rate', 'neutral_win_rate']].fillna(0.5)

        return stats

    def _compute_detailed_stats(self, detailed):
        w_cols = {
            'Season': 'Season', 'WTeamID': 'TeamID', 'Gender': 'Gender',
            'WFGM': 'FGM', 'WFGA': 'FGA', 'WFGM3': 'FGM3', 'WFGA3': 'FGA3',
            'WFTM': 'FTM', 'WFTA': 'FTA', 'WOR': 'OR', 'WDR': 'DR',
            'WAst': 'Ast', 'WTO': 'TO', 'WStl': 'Stl', 'WBlk': 'Blk', 'WPF': 'PF',
            'LFGM': 'OppFGM', 'LFGA': 'OppFGA', 'LFGM3': 'OppFGM3', 'LFGA3': 'OppFGA3',
            'LFTM': 'OppFTM', 'LFTA': 'OppFTA',
        }
        l_cols = {
            'Season': 'Season', 'LTeamID': 'TeamID', 'Gender': 'Gender',
            'LFGM': 'FGM', 'LFGA': 'FGA', 'LFGM3': 'FGM3', 'LFGA3': 'FGA3',
            'LFTM': 'FTM', 'LFTA': 'FTA', 'LOR': 'OR', 'LDR': 'DR',
            'LAst': 'Ast', 'LTO': 'TO', 'LStl': 'Stl', 'LBlk': 'Blk', 'LPF': 'PF',
            'WFGM': 'OppFGM', 'WFGA': 'OppFGA', 'WFGM3': 'OppFGM3', 'WFGA3': 'OppFGA3',
            'WFTM': 'OppFTM', 'WFTA': 'OppFTA',
        }

        w_df = detailed[list(w_cols.keys())].rename(columns=w_cols)
        l_df = detailed[list(l_cols.keys())].rename(columns=l_cols)
        games = pd.concat([w_df, l_df], ignore_index=True)

        grp = games.groupby(['Season', 'TeamID', 'Gender'])
        sums = grp.sum()
        counts = grp.size()

        stats = pd.DataFrame(index=sums.index)
        stats['fg_pct'] = sums['FGM'] / sums['FGA']
        stats['fg3_pct'] = sums['FGM3'] / sums['FGA3']
        stats['ft_pct'] = sums['FTM'] / sums['FTA']

        for col in ['OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']:
            stats[f'avg_{col.lower()}'] = sums[col] / counts

        stats['avg_rebounds'] = (sums['OR'] + sums['DR']) / counts

        stats['opp_fg_pct'] = sums['OppFGM'] / sums['OppFGA']
        stats['opp_fg3_pct'] = sums['OppFGM3'] / sums['OppFGA3']
        stats['opp_ft_pct'] = sums['OppFTM'] / sums['OppFTA']

        stats = stats.replace([np.inf, -np.inf], np.nan).reset_index()
        return stats

    def _compute_sos(self, compact):
        win_rows = compact[['Season', 'WTeamID', 'Gender']].copy()
        win_rows.columns = ['Season', 'TeamID', 'Gender']
        win_rows['Win'] = 1
        loss_rows = compact[['Season', 'LTeamID', 'Gender']].copy()
        loss_rows.columns = ['Season', 'TeamID', 'Gender']
        loss_rows['Win'] = 0
        all_games = pd.concat([win_rows, loss_rows], ignore_index=True)
        team_wr = all_games.groupby(['Season', 'TeamID', 'Gender'])['Win'].mean().reset_index()
        team_wr.columns = ['Season', 'TeamID', 'Gender', 'OppWinRate']

        opp_from_wins = compact[['Season', 'WTeamID', 'LTeamID', 'Gender']].copy()
        opp_from_wins.columns = ['Season', 'TeamID', 'OppID', 'Gender']
        opp_from_losses = compact[['Season', 'LTeamID', 'WTeamID', 'Gender']].copy()
        opp_from_losses.columns = ['Season', 'TeamID', 'OppID', 'Gender']
        opponents = pd.concat([opp_from_wins, opp_from_losses], ignore_index=True)

        opponents = opponents.merge(
            team_wr, left_on=['Season', 'OppID', 'Gender'],
            right_on=['Season', 'TeamID', 'Gender'], suffixes=('', '_opp')
        )

        sos = opponents.groupby(['Season', 'TeamID', 'Gender'])['OppWinRate'].mean().reset_index()
        sos.columns = ['Season', 'TeamID', 'Gender', 'sos']
        return sos

    def _compute_conference_strength(self, compact):
        m_conf = self.m_team_conf.copy()
        m_conf['Gender'] = 'M'
        w_conf = self.w_team_conf.copy()
        w_conf['Gender'] = 'W'
        team_conf = pd.concat([m_conf, w_conf], ignore_index=True)

        win_rows = compact[['Season', 'WTeamID', 'Gender']].copy()
        win_rows.columns = ['Season', 'TeamID', 'Gender']
        win_rows['Win'] = 1
        loss_rows = compact[['Season', 'LTeamID', 'Gender']].copy()
        loss_rows.columns = ['Season', 'TeamID', 'Gender']
        loss_rows['Win'] = 0
        all_games = pd.concat([win_rows, loss_rows], ignore_index=True)
        team_wr = all_games.groupby(['Season', 'TeamID', 'Gender'])['Win'].mean().reset_index()
        team_wr.columns = ['Season', 'TeamID', 'Gender', 'WinRate']

        merged = team_wr.merge(team_conf, on=['Season', 'TeamID', 'Gender'], how='left')
        conf_str = merged.groupby(['Season', 'ConfAbbrev', 'Gender'])['WinRate'].mean().reset_index()
        conf_str.columns = ['Season', 'ConfAbbrev', 'Gender', 'conf_strength']

        team_with_conf = merged[['Season', 'TeamID', 'Gender', 'ConfAbbrev']].drop_duplicates()
        result = team_with_conf.merge(conf_str, on=['Season', 'ConfAbbrev', 'Gender'], how='left')
        result = result[['Season', 'TeamID', 'Gender', 'conf_strength']].drop_duplicates()
        return result

    def _compute_momentum(self, compact):
        max_days = compact.groupby(['Season', 'Gender'])['DayNum'].max().reset_index()
        max_days.columns = ['Season', 'Gender', 'MaxDay']
        compact_m = compact.merge(max_days, on=['Season', 'Gender'])
        recent = compact_m[compact_m['DayNum'] > compact_m['MaxDay'] - 14].copy()

        win_rows = recent[['Season', 'WTeamID', 'WScore', 'LScore', 'Gender']].copy()
        win_rows.columns = ['Season', 'TeamID', 'Score', 'OppScore', 'Gender']
        win_rows['Win'] = 1

        loss_rows = recent[['Season', 'LTeamID', 'LScore', 'WScore', 'Gender']].copy()
        loss_rows.columns = ['Season', 'TeamID', 'Score', 'OppScore', 'Gender']
        loss_rows['Win'] = 0

        games = pd.concat([win_rows, loss_rows], ignore_index=True)
        games['Margin'] = games['Score'] - games['OppScore']

        grp = games.groupby(['Season', 'TeamID', 'Gender'])
        momentum = pd.DataFrame({
            'momentum_win_rate': grp['Win'].mean(),
            'momentum_avg_margin': grp['Margin'].mean(),
            'momentum_games': grp['Win'].count(),
        }).reset_index()

        return momentum

    # --------------------------------------------------------------- Elo ratings
    def build_elo_ratings(self):
        if self._elo_ratings is not None:
            return self._elo_ratings

        compact = self._combine_mw(self.m_reg_compact, self.w_reg_compact)
        tourney = self._combine_mw(self.m_tourney_compact, self.w_tourney_compact)

        all_games = pd.concat([compact, tourney], ignore_index=True)
        all_games = all_games.sort_values(['Gender', 'Season', 'DayNum']).reset_index(drop=True)

        K = 32
        HCA = 100

        results = []

        for gender in ['M', 'W']:
            elo = {}
            g_games = all_games[all_games['Gender'] == gender]
            seasons = sorted(g_games['Season'].unique())

            for season in seasons:
                for tid in elo:
                    elo[tid] = 0.75 * elo[tid] + 0.25 * 1500

                sg = g_games[g_games['Season'] == season].sort_values('DayNum')
                w_ids = sg['WTeamID'].values
                l_ids = sg['LTeamID'].values
                locs = sg['WLoc'].values
                days = sg['DayNum'].values

                reg_elo_snapshot = {}
                passed_reg = False

                for i in range(len(w_ids)):
                    if not passed_reg and days[i] > 132:
                        passed_reg = True
                        reg_elo_snapshot = dict(elo)

                    wt, lt = int(w_ids[i]), int(l_ids[i])
                    w_e = elo.get(wt, 1500.0)
                    l_e = elo.get(lt, 1500.0)

                    loc = locs[i]
                    hca_w = HCA if loc == 'H' else (-HCA if loc == 'A' else 0)

                    w_adj = w_e + hca_w
                    l_adj = l_e - hca_w

                    w_exp = 1.0 / (1.0 + 10.0 ** ((l_adj - w_adj) / 400.0))
                    elo[wt] = w_e + K * (1.0 - w_exp)
                    elo[lt] = l_e + K * (w_exp - 1.0)

                if not passed_reg:
                    reg_elo_snapshot = dict(elo)

                season_teams = set(int(t) for t in w_ids.tolist() + l_ids.tolist())
                for tid in season_teams:
                    results.append({
                        'Season': season,
                        'TeamID': tid,
                        'Gender': gender,
                        'elo_end_season': elo.get(tid, 1500.0),
                        'elo_reg_season': reg_elo_snapshot.get(tid, elo.get(tid, 1500.0)),
                    })

        self._elo_ratings = pd.DataFrame(results)
        return self._elo_ratings

    # ----------------------------------------------------------- seed features
    def build_seed_features(self):
        if self._seed_features is not None:
            return self._seed_features

        m_seeds = self.m_tourney_seeds.copy()
        m_seeds['Gender'] = 'M'
        w_seeds = self.w_tourney_seeds.copy()
        w_seeds['Gender'] = 'W'
        seeds = pd.concat([m_seeds, w_seeds], ignore_index=True)

        seeds['seed_region'] = seeds['Seed'].str[0]
        seeds['seed_num'] = seeds['Seed'].str[1:3].astype(int)

        self._seed_features = seeds[['Season', 'TeamID', 'Gender', 'seed_region', 'seed_num']]
        return self._seed_features

    # --------------------------------------------------------- massey features
    def build_massey_features(self):
        if self._massey_features is not None:
            return self._massey_features

        dtypes = {
            'Season': 'int16',
            'RankingDayNum': 'int16',
            'SystemName': 'category',
            'TeamID': 'int32',
            'OrdinalRank': 'int16',
        }

        massey = pd.read_csv(
            self.data_dir / 'MMasseyOrdinals.csv',
            dtype=dtypes,
            usecols=['Season', 'RankingDayNum', 'SystemName', 'TeamID', 'OrdinalRank'],
        )

        max_day = massey.groupby('Season')['RankingDayNum'].max().reset_index()
        max_day.columns = ['Season', 'MaxDay']
        massey = massey.merge(max_day, on='Season')
        final = massey[massey['RankingDayNum'] == massey['MaxDay']].copy()
        final.drop(columns=['MaxDay', 'RankingDayNum'], inplace=True)

        grp = final.groupby(['Season', 'TeamID'])['OrdinalRank']
        agg = pd.DataFrame({
            'massey_mean_rank': grp.mean(),
            'massey_median_rank': grp.median(),
            'massey_min_rank': grp.min(),
            'massey_max_rank': grp.max(),
            'massey_std_rank': grp.std(),
        }).reset_index()

        specific_systems = {'POM': 'massey_pom', 'SAG': 'massey_sag',
                            'MOR': 'massey_mor', 'DOL': 'massey_dol'}
        for sys_name, col_name in specific_systems.items():
            sys_df = final[final['SystemName'] == sys_name][['Season', 'TeamID', 'OrdinalRank']].copy()
            sys_df.columns = ['Season', 'TeamID', col_name]
            agg = agg.merge(sys_df, on=['Season', 'TeamID'], how='left')

        agg['Gender'] = 'M'
        self._massey_features = agg
        return self._massey_features

    # --------------------------------------------------------- coach features
    def build_coach_features(self):
        if self._coach_features is not None:
            return self._coach_features

        coaches = self.m_coaches.copy()

        # Keep only the coach active at end of season (largest LastDayNum per season/team)
        active_coaches = coaches.sort_values('LastDayNum', ascending=False) \
                                .drop_duplicates(subset=['Season', 'TeamID'], keep='first')

        # Coach experience: count of distinct prior seasons for this coach
        coach_first_season = coaches.groupby('CoachName')['Season'].min().reset_index()
        coach_first_season.columns = ['CoachName', 'FirstSeason']
        active_coaches = active_coaches.merge(coach_first_season, on='CoachName', how='left')
        active_coaches['coach_experience'] = active_coaches['Season'] - active_coaches['FirstSeason']

        # Tourney wins and appearances (cumulative prior to current season)
        tourney = self.m_tourney_compact.copy()

        # Match tourney games to coaches via season/team with day range check
        tourney_w = tourney.merge(
            coaches, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID']
        )
        tourney_w = tourney_w[
            (tourney_w['DayNum'] >= tourney_w['FirstDayNum']) &
            (tourney_w['DayNum'] <= tourney_w['LastDayNum'])
        ]

        tourney_l = tourney.merge(
            coaches, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID']
        )
        tourney_l = tourney_l[
            (tourney_l['DayNum'] >= tourney_l['FirstDayNum']) &
            (tourney_l['DayNum'] <= tourney_l['LastDayNum'])
        ]

        # Wins per coach per season
        wins_by_cs = tourney_w.groupby(['CoachName', 'Season']).size().reset_index(name='wins')
        # Appearances: unique seasons in tourney
        apps_w = tourney_w[['CoachName', 'Season']].drop_duplicates()
        apps_l = tourney_l[['CoachName', 'Season']].drop_duplicates()
        all_apps = pd.concat([apps_w, apps_l]).drop_duplicates()

        # Build cumulative stats per coach up to each season
        all_coach_names = active_coaches['CoachName'].unique()

        cum_records = []
        for coach_name in all_coach_names:
            c_wins = wins_by_cs[wins_by_cs['CoachName'] == coach_name].sort_values('Season')
            c_apps = all_apps[all_apps['CoachName'] == coach_name]['Season'].sort_values().values

            wins_dict = dict(zip(c_wins['Season'], c_wins['wins']))
            all_seasons_with_data = sorted(set(c_wins['Season'].tolist() + list(c_apps)))

            running_wins = 0
            running_apps = 0
            # Store cumulative totals BEFORE each season
            prior_totals = {}

            for s in all_seasons_with_data:
                prior_totals[s] = (running_wins, running_apps)
                running_wins += wins_dict.get(s, 0)
                if s in c_apps:
                    running_apps += 1

            # For seasons after all data, carry forward
            prior_totals['_final'] = (running_wins, running_apps)

            coach_rows = active_coaches[active_coaches['CoachName'] == coach_name]
            for _, row in coach_rows.iterrows():
                season = row['Season']
                # Find cumulative before this season
                applicable = {k: v for k, v in prior_totals.items()
                              if isinstance(k, (int, float)) and k <= season}
                if applicable:
                    _, (w, a) = max(applicable.items(), key=lambda x: x[0])
                else:
                    w, a = 0, 0
                cum_records.append({
                    'Season': season, 'TeamID': row['TeamID'],
                    'CoachName': coach_name,
                    'coach_tourney_wins': w, 'coach_tourney_apps': a
                })

        cum_df = pd.DataFrame(cum_records) if cum_records else pd.DataFrame(
            columns=['Season', 'TeamID', 'CoachName', 'coach_tourney_wins', 'coach_tourney_apps']
        )

        result = active_coaches[['Season', 'TeamID', 'CoachName', 'coach_experience']].merge(
            cum_df[['Season', 'TeamID', 'CoachName', 'coach_tourney_wins', 'coach_tourney_apps']],
            on=['Season', 'TeamID', 'CoachName'], how='left'
        )
        result['coach_tourney_wins'] = result['coach_tourney_wins'].fillna(0).astype(int)
        result['coach_tourney_apps'] = result['coach_tourney_apps'].fillna(0).astype(int)
        result['Gender'] = 'M'
        result = result[['Season', 'TeamID', 'Gender', 'coach_experience',
                         'coach_tourney_wins', 'coach_tourney_apps']]

        self._coach_features = result
        return self._coach_features

    # ----------------------------------------------------- matchup features
    def _get_feature_columns(self):
        stats = self.build_team_season_stats()
        stat_cols = [c for c in stats.columns if c not in ['Season', 'TeamID', 'Gender']]

        feature_names = [f'{c}_diff' for c in stat_cols]
        feature_names += ['elo_end_season_diff', 'elo_reg_season_diff']
        feature_names += ['seed_num_diff']
        feature_names += [f'{c}_diff' for c in [
            'massey_mean_rank', 'massey_median_rank', 'massey_min_rank',
            'massey_max_rank', 'massey_std_rank',
            'massey_pom', 'massey_sag', 'massey_mor', 'massey_dol'
        ]]
        feature_names += [f'{c}_diff' for c in [
            'coach_experience', 'coach_tourney_wins', 'coach_tourney_apps'
        ]]
        feature_names += ['h2h_win_rate', 'h2h_games']
        return feature_names, stat_cols

    def _ensure_all_features_built(self):
        self.build_team_season_stats()
        self.build_elo_ratings()
        self.build_seed_features()
        self.build_massey_features()
        self.build_coach_features()

    def _build_lookups(self):
        stats_lk = {}
        for _, row in self._team_season_stats.iterrows():
            stats_lk[(row['Season'], row['TeamID'], row['Gender'])] = row

        elo_lk = {}
        for _, row in self._elo_ratings.iterrows():
            elo_lk[(row['Season'], row['TeamID'], row['Gender'])] = row

        seed_lk = {}
        for _, row in self._seed_features.iterrows():
            seed_lk[(row['Season'], row['TeamID'], row['Gender'])] = row

        massey_lk = {}
        for _, row in self._massey_features.iterrows():
            massey_lk[(row['Season'], row['TeamID'], row['Gender'])] = row

        coach_lk = {}
        for _, row in self._coach_features.iterrows():
            coach_lk[(row['Season'], row['TeamID'], row['Gender'])] = row

        return stats_lk, elo_lk, seed_lk, massey_lk, coach_lk

    def _compute_matchup_vector(self, season, team1, team2, gender,
                                stat_cols, stats_lk, elo_lk, seed_lk,
                                massey_lk, coach_lk, h2h_games):
        features = []
        k1 = (season, team1, gender)
        k2 = (season, team2, gender)

        massey_cols = ['massey_mean_rank', 'massey_median_rank', 'massey_min_rank',
                       'massey_max_rank', 'massey_std_rank',
                       'massey_pom', 'massey_sag', 'massey_mor', 'massey_dol']
        coach_cols = ['coach_experience', 'coach_tourney_wins', 'coach_tourney_apps']

        # Stat diffs
        s1 = stats_lk.get(k1)
        s2 = stats_lk.get(k2)
        for col in stat_cols:
            v1 = s1[col] if s1 is not None and pd.notna(s1.get(col, np.nan)) else np.nan
            v2 = s2[col] if s2 is not None and pd.notna(s2.get(col, np.nan)) else np.nan
            features.append(v1 - v2 if pd.notna(v1) and pd.notna(v2) else np.nan)

        # Elo diffs
        e1 = elo_lk.get(k1)
        e2 = elo_lk.get(k2)
        for col in ['elo_end_season', 'elo_reg_season']:
            v1 = e1[col] if e1 is not None else 1500.0
            v2 = e2[col] if e2 is not None else 1500.0
            features.append(v1 - v2)

        # Seed diff
        sd1 = seed_lk.get(k1)
        sd2 = seed_lk.get(k2)
        sn1 = sd1['seed_num'] if sd1 is not None else np.nan
        sn2 = sd2['seed_num'] if sd2 is not None else np.nan
        features.append(sn1 - sn2 if pd.notna(sn1) and pd.notna(sn2) else np.nan)

        # Massey diffs (men only)
        if gender == 'M':
            m1 = massey_lk.get(k1)
            m2 = massey_lk.get(k2)
            for col in massey_cols:
                v1 = m1[col] if m1 is not None and pd.notna(m1.get(col, np.nan)) else np.nan
                v2 = m2[col] if m2 is not None and pd.notna(m2.get(col, np.nan)) else np.nan
                features.append(v1 - v2 if pd.notna(v1) and pd.notna(v2) else np.nan)
        else:
            features.extend([np.nan] * len(massey_cols))

        # Coach diffs (men only)
        if gender == 'M':
            c1 = coach_lk.get(k1)
            c2 = coach_lk.get(k2)
            for col in coach_cols:
                v1 = float(c1[col]) if c1 is not None and pd.notna(c1.get(col, np.nan)) else np.nan
                v2 = float(c2[col]) if c2 is not None and pd.notna(c2.get(col, np.nan)) else np.nan
                features.append(v1 - v2 if pd.notna(v1) and pd.notna(v2) else np.nan)
        else:
            features.extend([np.nan] * len(coach_cols))

        # Head-to-head (last 5 seasons, excluding current)
        recent = h2h_games[
            (h2h_games['Gender'] == gender) &
            (h2h_games['Season'] >= season - 5) &
            (h2h_games['Season'] < season)
        ]
        h2h_t1 = len(recent[(recent['WTeamID'] == team1) & (recent['LTeamID'] == team2)])
        h2h_t2 = len(recent[(recent['WTeamID'] == team2) & (recent['LTeamID'] == team1)])
        total = h2h_t1 + h2h_t2
        features.append(h2h_t1 / total if total > 0 else 0.5)
        features.append(total)

        return features

    def build_matchup_features(self, season, team1_id, team2_id, gender='M'):
        self._ensure_all_features_built()
        stats_lk, elo_lk, seed_lk, massey_lk, coach_lk = self._build_lookups()

        feature_names, stat_cols = self._get_feature_columns()
        h2h_games = self._get_h2h_games()

        vec = self._compute_matchup_vector(
            season, team1_id, team2_id, gender,
            stat_cols, stats_lk, elo_lk, seed_lk, massey_lk, coach_lk, h2h_games
        )
        return dict(zip(feature_names, vec))

    # -------------------------------------------------- training data builder
    def build_training_data(self, seasons):
        m_tourney = self.m_tourney_compact[self.m_tourney_compact['Season'].isin(seasons)].copy()
        m_tourney['Gender'] = 'M'
        w_tourney = self.w_tourney_compact[self.w_tourney_compact['Season'].isin(seasons)].copy()
        w_tourney['Gender'] = 'W'
        tourney = pd.concat([m_tourney, w_tourney], ignore_index=True)

        self._ensure_all_features_built()
        stats_lk, elo_lk, seed_lk, massey_lk, coach_lk = self._build_lookups()
        feature_names, stat_cols = self._get_feature_columns()
        h2h_games = self._get_h2h_games()

        all_features = []
        targets = []
        season_labels = []

        for _, game in tourney.iterrows():
            gender = game['Gender']
            season = game['Season']
            w_id = game['WTeamID']
            l_id = game['LTeamID']

            team1 = min(w_id, l_id)
            team2 = max(w_id, l_id)
            y_val = 1 if w_id == team1 else 0

            vec = self._compute_matchup_vector(
                season, team1, team2, gender,
                stat_cols, stats_lk, elo_lk, seed_lk, massey_lk, coach_lk, h2h_games
            )
            all_features.append(vec)
            targets.append(y_val)
            season_labels.append(season)

        X = pd.DataFrame(all_features, columns=feature_names)
        y = pd.Series(targets, name='target')
        seasons_out = pd.Series(season_labels, name='season')
        return X, y, seasons_out

    # ------------------------------------------------- submission features
    def build_submission_features(self, submission_file):
        sub = pd.read_csv(submission_file)

        parts = sub['ID'].str.split('_', expand=True).astype(int)
        sub['Season'] = parts[0]
        sub['Team1'] = parts[1]
        sub['Team2'] = parts[2]
        sub['Gender'] = np.where(sub['Team1'] < 3000, 'M', 'W')

        self._ensure_all_features_built()
        feature_names, stat_cols = self._get_feature_columns()

        # Build DataFrames indexed by (Season, TeamID, Gender) for vectorized merge
        stats = self._team_season_stats.copy()
        elo = self._elo_ratings.copy()
        seeds = self._seed_features.copy()
        massey = self._massey_features.copy() if self._massey_features is not None else pd.DataFrame()
        coach = self._coach_features.copy() if self._coach_features is not None else pd.DataFrame()

        # Merge stats for team1 and team2
        def merge_team_data(df, team_col, suffix, data, cols, key_cols=['Season', 'TeamID', 'Gender']):
            rename = {c: f'{c}_{suffix}' for c in cols}
            merged_cols = key_cols + cols
            available = [c for c in merged_cols if c in data.columns]
            merge_on = {k: team_col if k == 'TeamID' else k for k in key_cols}
            tmp = data[available].copy()
            for k, v in merge_on.items():
                if k != v:
                    tmp = tmp.rename(columns={k: v})
            rename_back = {f: f'{c}_{suffix}' for c, f in zip(cols, cols)}
            return df.merge(tmp, on=list(merge_on.values()), how='left', suffixes=('', f'_dup_{suffix}'))

        # Simpler vectorized approach: convert lookups to DataFrames and merge
        result_cols = []

        # Stats diffs
        for col in stat_cols:
            sub = sub.copy() if col == stat_cols[0] else sub
            sub = sub.merge(
                stats[['Season', 'TeamID', 'Gender', col]].rename(columns={'TeamID': 'Team1', col: f'{col}_1'}),
                on=['Season', 'Team1', 'Gender'], how='left'
            ).merge(
                stats[['Season', 'TeamID', 'Gender', col]].rename(columns={'TeamID': 'Team2', col: f'{col}_2'}),
                on=['Season', 'Team2', 'Gender'], how='left'
            )
            sub[f'{col}_diff'] = sub[f'{col}_1'] - sub[f'{col}_2']
            result_cols.append(f'{col}_diff')
            sub = sub.drop(columns=[f'{col}_1', f'{col}_2'])

        # Elo diffs
        for col in ['elo_end_season', 'elo_reg_season']:
            sub = sub.merge(
                elo[['Season', 'TeamID', 'Gender', col]].rename(columns={'TeamID': 'Team1', col: f'{col}_1'}),
                on=['Season', 'Team1', 'Gender'], how='left'
            ).merge(
                elo[['Season', 'TeamID', 'Gender', col]].rename(columns={'TeamID': 'Team2', col: f'{col}_2'}),
                on=['Season', 'Team2', 'Gender'], how='left'
            )
            sub[f'{col}_1'] = sub[f'{col}_1'].fillna(1500.0)
            sub[f'{col}_2'] = sub[f'{col}_2'].fillna(1500.0)
            sub[f'{col}_diff'] = sub[f'{col}_1'] - sub[f'{col}_2']
            result_cols.append(f'{col}_diff')
            sub = sub.drop(columns=[f'{col}_1', f'{col}_2'])

        # Seed diff
        sub = sub.merge(
            seeds[['Season', 'TeamID', 'Gender', 'seed_num']].rename(columns={'TeamID': 'Team1', 'seed_num': 'seed_1'}),
            on=['Season', 'Team1', 'Gender'], how='left'
        ).merge(
            seeds[['Season', 'TeamID', 'Gender', 'seed_num']].rename(columns={'TeamID': 'Team2', 'seed_num': 'seed_2'}),
            on=['Season', 'Team2', 'Gender'], how='left'
        )
        sub['seed_num_diff'] = sub['seed_1'] - sub['seed_2']
        result_cols.append('seed_num_diff')
        sub = sub.drop(columns=['seed_1', 'seed_2'])

        # Massey diffs (men only)
        massey_cols = ['massey_mean_rank', 'massey_median_rank', 'massey_min_rank',
                       'massey_max_rank', 'massey_std_rank',
                       'massey_pom', 'massey_sag', 'massey_mor', 'massey_dol']
        if len(massey) > 0:
            for col in massey_cols:
                if col in massey.columns:
                    sub = sub.merge(
                        massey[['Season', 'TeamID', 'Gender', col]].rename(columns={'TeamID': 'Team1', col: f'{col}_1'}),
                        on=['Season', 'Team1', 'Gender'], how='left'
                    ).merge(
                        massey[['Season', 'TeamID', 'Gender', col]].rename(columns={'TeamID': 'Team2', col: f'{col}_2'}),
                        on=['Season', 'Team2', 'Gender'], how='left'
                    )
                    sub[f'{col}_diff'] = sub[f'{col}_1'] - sub[f'{col}_2']
                    sub = sub.drop(columns=[f'{col}_1', f'{col}_2'])
                else:
                    sub[f'{col}_diff'] = np.nan
                result_cols.append(f'{col}_diff')
        else:
            for col in massey_cols:
                sub[f'{col}_diff'] = np.nan
                result_cols.append(f'{col}_diff')

        # Coach diffs (men only)
        coach_cols = ['coach_experience', 'coach_tourney_wins', 'coach_tourney_apps']
        if len(coach) > 0:
            for col in coach_cols:
                if col in coach.columns:
                    sub = sub.merge(
                        coach[['Season', 'TeamID', 'Gender', col]].rename(columns={'TeamID': 'Team1', col: f'{col}_1'}),
                        on=['Season', 'Team1', 'Gender'], how='left'
                    ).merge(
                        coach[['Season', 'TeamID', 'Gender', col]].rename(columns={'TeamID': 'Team2', col: f'{col}_2'}),
                        on=['Season', 'Team2', 'Gender'], how='left'
                    )
                    sub[f'{col}_diff'] = sub[f'{col}_1'] - sub[f'{col}_2']
                    sub = sub.drop(columns=[f'{col}_1', f'{col}_2'])
                else:
                    sub[f'{col}_diff'] = np.nan
                result_cols.append(f'{col}_diff')
        else:
            for col in coach_cols:
                sub[f'{col}_diff'] = np.nan
                result_cols.append(f'{col}_diff')

        # H2H: precompute dict lookup for speed
        sub['h2h_win_rate'] = 0.5
        sub['h2h_games'] = 0.0
        result_cols.extend(['h2h_win_rate', 'h2h_games'])

        h2h = self._get_h2h_games()
        if len(h2h) > 0:
            for season in sub['Season'].unique():
                recent = h2h[(h2h['Season'] >= season - 5) & (h2h['Season'] < season)]
                if len(recent) == 0:
                    continue
                # Build dict: (winner, loser) -> count
                h2h_dict = {}
                for _, g in recent.iterrows():
                    k = (g['WTeamID'], g['LTeamID'])
                    h2h_dict[k] = h2h_dict.get(k, 0) + 1

                season_mask = sub['Season'] == season
                t1 = sub.loc[season_mask, 'Team1'].values
                t2 = sub.loc[season_mask, 'Team2'].values
                wr = np.full(len(t1), 0.5)
                gm = np.zeros(len(t1))
                for i in range(len(t1)):
                    w1 = h2h_dict.get((t1[i], t2[i]), 0)
                    w2 = h2h_dict.get((t2[i], t1[i]), 0)
                    total = w1 + w2
                    if total > 0:
                        wr[i] = w1 / total
                        gm[i] = total
                sub.loc[season_mask, 'h2h_win_rate'] = wr
                sub.loc[season_mask, 'h2h_games'] = gm

        X = sub[result_cols].copy()
        X.columns = feature_names
        return X


if __name__ == '__main__':
    fb = FeatureBuilder(data_dir='data')

    print('Building team season stats...')
    stats = fb.build_team_season_stats()
    print(f'  Shape: {stats.shape}')
    print(f'  Columns: {list(stats.columns)}')

    print('\nBuilding Elo ratings...')
    elo = fb.build_elo_ratings()
    print(f'  Shape: {elo.shape}')

    print('\nBuilding seed features...')
    seeds = fb.build_seed_features()
    print(f'  Shape: {seeds.shape}')

    print('\nBuilding Massey features...')
    massey = fb.build_massey_features()
    print(f'  Shape: {massey.shape}')

    print('\nBuilding coach features...')
    coach = fb.build_coach_features()
    print(f'  Shape: {coach.shape}')

    print('\nBuilding training data for 2021-2024...')
    X, y = fb.build_training_data(list(range(2021, 2025)))
    print(f'  X shape: {X.shape}')
    print(f'  y distribution: {y.value_counts().to_dict()}')
    print(f'  Feature names: {list(X.columns)}')
    print(f'  Missing values per column:')
    missing = X.isnull().sum()
    print(missing[missing > 0])
