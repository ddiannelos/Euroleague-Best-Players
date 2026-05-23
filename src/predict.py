import pandas as pd
import time
import argparse
from extract_stats import Stats
from tabulate import tabulate

# --- Terminal Styling ---
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

# --- Helper Functions ---
def print_step(message, icon="⚙️"):
    print(f"{Colors.BLUE}{icon} {message}{Colors.ENDC}")

def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")

def print_header(current_round):
    print(f"\n{Colors.HEADER}{Colors.BOLD}" + "="*50)
    print(f"🏀 EUROLEAGUE FANTASY PREDICTOR | ROUND {current_round + 1}")
    print("="*50 + f"{Colors.ENDC}\n")

def main():
    # --- Set up Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Predict Euroleague Player PID based on matchup and form.")
    parser.add_argument("--season", type=int, default=2025, help="Starting year of the season (e.g. 2025 for 2025-2026)")
    parser.add_argument("--round", type=int, required=True, help="The current round number")
    parser.add_argument("--last_rounds", type=int, default=5, help="Number of previous rounds to calculate recent form")
    args = parser.parse_args()

    print_header(args.round)

    # --- Step 1: Loading data
    print_step("Initializing stats engine...", icon="🚀")
    start_time = time.time()

    stats = Stats(season=args.season, current_round=args.round, last_rounds=args.last_rounds)

    print_success(f"Data loaded successfully ({round(time.time() - start_time, 2)}s)")

    player_stats = stats.player_valuation
    defense_points = stats.defense_points
    schedule = stats.schedule
    team_pir = stats.team_pir

    # --- Step 2: Processing Data ---
    print_step("Calculating league averages...", icon="🧮")

    # Calculate League average per position
    league_average_g = defense_points['points_taken_g'].mean()
    league_average_f = defense_points['points_taken_f'].mean()
    league_average_c = defense_points['points_taken_c'].mean()

    print(f"   {Colors.CYAN}↳ Avg Points Allowed :: G: {league_average_g:.1f} | F: {league_average_f:.1f} | C: {league_average_c:.1f}{Colors.ENDC}")
    print_step("Merging schedule, team strength, and defense stats...", icon="🔗")

    # Merge players with schedule
    df_main = pd.merge(player_stats, schedule, on='Team', how='left')

    # Merge with team pir
    df_main = pd.merge(df_main, team_pir, on='Team', how='left')
    df_main.rename(columns={'PIR': 'Team_PIR'}, inplace=True)

    # Merge with opponent pir
    df_main = pd.merge(df_main, team_pir, left_on='Opponent', right_on='Team', how='left', suffixes=('', '_Opp'))
    df_main.rename(columns={'PIR': 'Opponent_PIR'}, inplace=True)

    # Merge with defense stats
    df_main = pd.merge(df_main, defense_points, left_on='Opponent', right_on='Team', how='left', suffixes=('', '_Def'))
    df_main.drop(columns=['Team_Def'], inplace=True)

    # --- Step 3: Algorithm ---
    print_step("Running projection algorithm...", icon="🧠")

    def calculate_projection(row):
        """Calculates the projected PIR using base value, matchups, home advantage and team strength."""
        # 1. Base Value (Weighted 60/40)
        v_base = (row['Valuation'] * 0.6) + (row['AverageValuationPerGame'] * 0.4)

        # 2. Matchup Coefficient (Defense vs Position)
        if row['Position'] == 'C':
            matchup_mult = row['points_taken_c'] / league_average_c
        elif row['Position'] == 'F':
            matchup_mult = row['points_taken_f'] / league_average_f
        else:
            matchup_mult = row['points_taken_g'] / league_average_g

        # 3. Home advantage
        home_adv = 1.05 if row['is_home_game'] else 0.95

        # 4. Team Strength (Differential on PIR)
        pir_diff = row['Team_PIR'] - row['Opponent_PIR']
        # team_str = 1 + (pir_diff / 200.0)
        team_str = 1 + (max(min(pir_diff, 25), -25) / 300.0)

        # Final Calculation
        projected_pir = v_base * matchup_mult * home_adv * team_str

        return round(projected_pir, 2)

    # Calculate Projected PIR
    df_main['Projected_PIR'] = df_main.apply(calculate_projection, axis=1)

    # --- Step 4: Ranking ---
    print_step("Ranking players by position...", icon="📊")

    # Find best Players for each Position
    def get_best_by_position(df, pos, top_n=20):
        pos_df = df[df['Position'] == pos].copy()
        pos_df = pos_df.sort_values(by='Projected_PIR', ascending=False)
        pos_df['Rank_in_Pos'] = range(1, len(pos_df) + 1)
        return pos_df.head(top_n)

    top_guards = get_best_by_position(df_main, 'G')
    top_forwards = get_best_by_position(df_main, 'F')
    top_centers = get_best_by_position(df_main, 'C')

    output_columns = [
        'Rank_in_Pos',
        'Player',
        'Team',
        'Opponent',
        'AverageValuationPerGame',
        'Valuation',
        'Projected_PIR'
    ]

    positions_data = {
        "--- Top Guards ---": top_guards,
        "--- Top Forwards ---": top_forwards,
        "--- Top Centers ---": top_centers,
    }

    # --- Step 5: Exporting ---
    output_path = f'reports/player_report_round_{args.round+1}.txt'
    print_step(f"Writing report to file...", icon="💾")

    with open(output_path, 'w', encoding='utf-8') as f:
        for title, data in positions_data.items():
            pretty_table = tabulate(
                data[output_columns],
                headers='keys',
                tablefmt='psql',
                showindex=False,
                floatfmt=".1f"
            )

            f.write(f"{title}\n")
            f.write(pretty_table)
            f.write("\n\n")

    # Final Success Message
    print("\n" + "="*50)
    print(f"{Colors.GREEN}{Colors.BOLD}✨  PREDICTION COMPLETE  ✨{Colors.ENDC}")
    print(f"Report location: {Colors.UNDERLINE}{output_path}{Colors.ENDC}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
