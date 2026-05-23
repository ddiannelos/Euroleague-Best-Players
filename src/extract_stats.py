import pandas as pd
from euroleague_api.boxscore_data import BoxScoreData
from euroleague_api.team_stats import TeamStats
from euroleague_api.game_stats import GameStats
from euroleague_api.player_stats import PlayerStats
import numpy as np
import os

# --- Formatting Helpers ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

team_mapping = {
    'TEL': 'Maccabi Rapyd Tel Aviv',
    'ASV': 'LDLC ASVEL Villeurbanne',
    'DUB': 'Dubai Basketball',
    'MUN': 'FC Bayern Munich',
    'MIL': 'EA7 Emporio Armani Milan',
    'PAN': 'Panathinaikos AKTOR Athens',
    'PRS': 'Paris Basketball',
    'ZAL': 'Zalgiris Kaunas',
    'MAD': 'Real Madrid',
    'BAS': 'Kosner Baskonia Vitoria-Gasteiz',
    'PAM': 'Valencia Basket',
    'IST': 'Anadolu Efes Istanbul',
    'MCO': 'AS Monaco',
    'ULK': 'Fenerbahce Beko Istanbul',
    'BAR': 'FC Barcelona',
    'OLY': 'Olympiacos Piraeus',
    'PAR': 'Partizan Mozzart Bet Belgrade',
    'RED': 'Crvena Zvezda Meridianbet Belgrade',
    'VIR': 'Virtus Bologna',
    'HTA': 'Hapoel IBI Tel Aviv'
}

class Stats:
    """
    Handles the extraction, caching and preprocessing of Euroleague data.

    This class interacts with the euroleague_api to fetch player valuations,
    defensive points allowed per position, schedules and team PIR. It uses
    local csv caching to minimize API requests and improve performance.
    """
    def __init__(self, season , current_round, last_rounds):
        self.season = season
        self.current_round = current_round
        self.last_rounds = last_rounds

        # Initialize API clints once
        self.box_score_api = BoxScoreData()
        self.game_stats_api = GameStats()
        self.player_stats_api = PlayerStats()
        self.team_stats_api = TeamStats()

        # File Names
        self.player_cache_file = f"library/player_stats_{self.season}.csv"
        self.defense_cache_file = f"library/defense_stats_{self.season}.csv"

        # Load data (Prints will happen inside these methods)
        self.player_valuation = self.get_player_stats()
        self.defense_points = self.defense_per_position()
        self.schedule = self.get_schedule()
        self.team_pir = self.get_team_pir()

    def get_cache_data(self, file_path, fetch_single_round_func):
        filename = os.path.basename(file_path)

        # 1. Load existing data if file exists
        if os.path.exists(file_path):
            print(f"   {Colors.CYAN}📄 Loading cache: {filename}{Colors.ENDC}")
            df_cache = pd.read_csv(file_path)
            existing_rounds = df_cache['Round'].unique().tolist()
        else:
            print(f"   {Colors.WARNING}⚠️  No cache found for {filename}. Starting fresh...{Colors.ENDC}")
            df_cache = pd.DataFrame()
            existing_rounds = []

        # 2. Identify which rounds we need
        rounds_needed = list(range(1, self.current_round + 1))
        missing_rounds = [r for r in rounds_needed if r not in existing_rounds]

        # 3. Fetch missing rounds
        new_data_frames = []
        if missing_rounds:
            print(f"   {Colors.BLUE}📥 Fetching missing rounds: {missing_rounds}{Colors.ENDC}")
            for r in missing_rounds:
                # Simple progress indicator per round
                print(f"      ▶ Fetching round {r}...", end="\r")
                try:
                    df_round = fetch_single_round_func(r)
                    if df_round is not None and not df_round.empty:
                        df_round['Round'] = r
                        new_data_frames.append(df_round)
                        print(f"      {Colors.GREEN}✔ Round {r} fetched.{Colors.ENDC}")
                    else:
                        print(f"      {Colors.WARNING}⚠ Warning: No data found for round {r}{Colors.ENDC}")
                except Exception as e:
                    print(f"      {Colors.FAIL}✖ Error fetching round {r}: {e}{Colors.ENDC}")

            # 4. Append and save
            if new_data_frames:
                new_data = pd.concat(new_data_frames, ignore_index=True)
                # Combine with existing cache
                df_cache = pd.concat([df_cache, new_data], ignore_index=True)
                # Save to csv
                df_cache.to_csv(file_path, index=False)
                print(f"   {Colors.GREEN}💾 Updated cache saved to {filename}{Colors.ENDC}")
        else:
            print(f"   {Colors.GREEN}✔ Cache up to date.{Colors.ENDC}")

        target_rounds = range(self.current_round - self.last_rounds + 1, self.current_round + 1)
        df_filtered = df_cache[df_cache['Round'].isin(target_rounds)].copy()

        return df_filtered

    def fetch_player_round(self, round_number):
        return self.box_score_api.get_player_boxscore_stats_round(season=self.season, round_number=round_number)

    def get_player_stats(self):
        player_stats = self.get_cache_data(self.player_cache_file, self.fetch_player_round)

        if player_stats.empty:
            print(f"   {Colors.FAIL}✖ No player stats available.{Colors.ENDC}")
            return pd.DataFrame()

        # Keep relevant Columns
        player_stats = player_stats[['Player', 'Team', 'Valuation', 'Points']]

        # Find position
        # Suppress prints for these specific API calls if they are noisy, or just leave them
        df_guards = self.player_stats_api.get_player_stats_leaders_single_season(season=self.season, stat_category='Valuation', position='Guards')[['playerName', 'averagePerGame']]
        df_guards['Position'] = 'G'
        df_forwards = self.player_stats_api.get_player_stats_leaders_single_season(season=self.season, stat_category='Valuation', position='Forwards')[['playerName', 'averagePerGame']]
        df_forwards['Position'] = 'F'
        df_centers = self.player_stats_api.get_player_stats_leaders_single_season(season=self.season, stat_category='Valuation', position='Centers')[['playerName', 'averagePerGame']]
        df_centers['Position'] = 'C'
        df_players = pd.concat([df_guards, df_forwards, df_centers], ignore_index=True)

        df_players['averagePerGame'] = df_players['averagePerGame'].round(1)
        df_players.rename(columns={'averagePerGame': 'AverageValuationPerGame'}, inplace=True)

        player_scores = pd.merge(player_stats, df_players, left_on='Player', right_on='playerName')

        # Find the average Valuation
        player_scores = player_scores.groupby(['Player', 'Team', 'Position', 'AverageValuationPerGame'], as_index=False)['Valuation'].mean().reset_index()
        player_scores['Team'] = player_scores['Team'].replace(team_mapping)

        return player_scores

    def fetch_defense_round(self, round_number):
        try:
            # Get list of games for the round
            round_games = self.game_stats_api.get_game_report_round(season=self.season, round_number=round_number)
        except Exception as e:
            print(f"      {Colors.FAIL}✖ Error fetching game report for round {round_number}: {e}{Colors.ENDC}")
            return None

        if round_games.empty:
            return None

        all_boxscores = []

        # Loop through every game in report
        for index, row in round_games.iterrows():
            game_code = row['Gamecode']
            local = row['local.club.name']
            road = row['road.club.name']

            try:
                # Fetch detailed stats
                game_boxscore = self.box_score_api.get_player_boxscore_stats_data(season=self.season, gamecode=game_code)

                if not game_boxscore.empty:
                    game_boxscore['Gamecode'] = game_code
                    game_boxscore['local.club.name'] = local
                    game_boxscore['road.club.name'] = road
                    all_boxscores.append(game_boxscore)
            except Exception:
                # Skip bad games
                pass

        if all_boxscores:
            return pd.concat(all_boxscores, ignore_index=True)

        return None

    def defense_per_position(self):
        # 1. Get cached defense data
        raw_defense_data = self.get_cache_data(self.defense_cache_file, self.fetch_defense_round)

        if raw_defense_data.empty:
            print(f"   {Colors.WARNING}⚠ No defense data available{Colors.ENDC}")
            return pd.DataFrame()

        # 2. Clean & Process
        # final_df = raw_defense_data[['Gamecode', 'Player', 'Points', 'local.club.name', 'road.club.name']]
        final_df = raw_defense_data[['Gamecode', 'Player', 'Valuation', 'local.club.name', 'road.club.name']]

        final_df = final_df[~final_df['Player'].isin(['Team', 'Total'])]

        # Need position from player stats
        if self.player_valuation.empty:
            print(f"   {Colors.FAIL}✖ Cannot compute defense stats because player stats are missing.{Colors.ENDC}")
            return pd.DataFrame()

        df_merged = pd.merge(final_df, self.player_valuation[['Player', 'Position', 'Team']], on=['Player'], how='left')
        df_merged = df_merged.replace(r'^\s*$', np.nan, regex=True).dropna(subset=['Position', 'Team'])

        # Determine Opponent
        df_merged['AgainstTeam'] = np.where(
            df_merged['Team'] == df_merged['local.club.name'],
            df_merged['road.club.name'],
            df_merged['local.club.name']
        )
        # df_merged = df_merged[['Gamecode', 'Player', 'Position', 'Team', 'Points', 'AgainstTeam']]
        df_merged = df_merged[['Gamecode', 'Player', 'Position', 'Team', 'Valuation', 'AgainstTeam']]

        df_merged = df_merged.drop_duplicates(subset=['Gamecode', 'Player'])

        # 3. Calculate Averages
        # Sum points
        # avg_points_df = df_merged.groupby(['Team', 'AgainstTeam', 'Position'])['Points'].sum().reset_index()

        # Average over all games
        # avg_points_df = avg_points_df.groupby(['AgainstTeam', 'Position'])['Points'].mean().reset_index()

        avg_points_df = df_merged.groupby(['Team', 'AgainstTeam', 'Position'])['Valuation'].sum().reset_index()
        avg_points_df = avg_points_df.groupby(['AgainstTeam', 'Position'])['Valuation'].mean().reset_index()

        # avg_points_df.rename(columns={'Points': 'AveragePointsAgainst', 'AgainstTeam': 'Team'}, inplace=True)
        avg_points_df.rename(columns={'Valuation': 'AveragePointsAgainst', 'AgainstTeam': 'Team'}, inplace=True)

        avg_points_df['AveragePointsAgainst'] = avg_points_df['AveragePointsAgainst'].round(1)

        # Pivot the data
        pivoted_df = avg_points_df.pivot(index='Team', columns='Position', values='AveragePointsAgainst').reset_index()
        pivoted_df.columns = ['points_taken_' + col.lower() for col in pivoted_df.columns]
        pivoted_df.rename(columns={'points_taken_team': 'Team'}, inplace=True)

        return pivoted_df

    def get_schedule(self):
        try:
            print(f"   {Colors.BLUE}📅 Fetching schedule (S{self.season} R{self.current_round+1})...{Colors.ENDC}")
            # Fetch schedule data
            schedule = self.team_stats_api.get_gamecodes_round(season=self.season, round_number=self.current_round+1)[['local.club.name', 'road.club.name']]

            # Rename
            schedule.rename(columns={'local.club.name': 'HomeTeam', 'road.club.name': 'AwayTeam'}, inplace=True)

            # Create Home/Away perspectives
            home_perspective = schedule[['HomeTeam', 'AwayTeam']].rename(columns={'HomeTeam': 'Team', 'AwayTeam': 'Opponent'})
            home_perspective['is_home_game'] = True

            away_perspective = schedule[['AwayTeam', 'HomeTeam']].rename(columns={'AwayTeam': 'Team', 'HomeTeam': 'Opponent'})
            away_perspective['is_home_game'] = False

            matchup_lookup = pd.concat([home_perspective, away_perspective])

            return matchup_lookup
        except Exception as e:
            print(f"   {Colors.FAIL}✖ Error fetching schedule: {e}{Colors.ENDC}")
            return pd.DataFrame()

    def get_team_pir(self):
        try:
            print(f"   {Colors.BLUE}💪 Fetching Team PIR data...{Colors.ENDC}")
            # Get team pir
            team_pir = self.team_stats_api.get_team_stats_single_season('traditional', season=self.season, phase_type_code='RS')[['team.name', 'pir']]

            # Rename
            team_pir.rename(columns={'team.name': 'Team', 'pir': 'PIR'}, inplace=True)

            # Correct names from api
            name_corrections = {
                "Baskonia Vitoria-Gasteiz": 'Kosner Baskonia Vitoria-Gasteiz',
                'Hapoel Tel Aviv': 'Hapoel IBI Tel Aviv'
            }

            team_pir['Team'] = team_pir['Team'].replace(name_corrections)

            return team_pir
        except Exception as e:
            print(f"   {Colors.FAIL}✖ Error fetching team pir: {e}{Colors.ENDC}")
            return pd.DataFrame()

    def get_round(self):
        """
        Automatically finds the current round by checking which games are finished.
        Returns: The round number of the LAST finished round (or current if active).
        """
        print(f"   {Colors.BLUE}🔄  Detecting current round...{Colors.ENDC}")
        try:
            # Fetch all games for the current season
            # We use a large range (e.g., 1 to 34) or just fetch round 1 to 40 loop
            # A more efficient way is to check the latest standings or calendar
            
            # Strategy: Check the latest round's data. 
            # If Round 30 has 0 games played, check Round 29.
            for r in range(38, 0, -1): # Start from end of regular season down to 1
                try:
                    games = self.game_stats_api.get_game_report_round(
                        season=self.season, 
                        round_number=r
                    )
                    
                    if not games.empty and 'score_home' in games.columns:
                        # Check if at least one game has a score
                        # (Some APIs return strings like "0" or "NaN" for unplayed)
                        valid_scores = games[games['score_home'].notnull()]
                        
                        # If we find a round with valid scores, this is the current/latest round
                        if not valid_scores.empty:
                            print(f"      {Colors.GREEN}✔ Found active data at Round {r}{Colors.ENDC}")
                            return r
                except:
                    continue
            
            return -1 # Default fallback
            
        except Exception as e:
            print(f"      {Colors.WARNING}⚠ Could not auto-detect round: {e}{Colors.ENDC}")
            return -1 # Fallback
