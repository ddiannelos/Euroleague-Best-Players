# Euroleague-PIR-Projection 🏀

Python program that extracts the best players per position for each fantasy round (Guards, Forwards, Centers) using the `euroleague_api`.

This tool automatically fetches live data from the Euroleague API, caches it locally for performance, and applies a custom projection algorithm to predict player performance (Projected PIR) for upcoming rounds. It is designed to evaluate players based on their individual statistics, defensive matchups, and team strength.

## 🚀 Features

* **Automated Data Extraction:** Pulls player valuations, team statistics, and schedules directly from the Euroleague API.
* **Smart Caching:** Saves historical round data to a local `library/` directory as CSVs to minimize API calls and speed up execution.
* **Defensive Matchup Analysis:** Calculates how many points/valuation specific teams allow to specific positions (Guards, Forwards, Centers).
* **Custom Projection Algorithm:** Calculates a `Projected_PIR` for every player based on base value, positional matchups, home-court advantage, and team strength differentials.
* **Automated Reporting:** Generates a formatted text report in the `reports/` directory ranking the top 20 players by position for the upcoming round.

## 🛠️ Prerequisites

* Python 3.8+
* The following Python libraries are required:
    * `pandas`
    * `numpy`
    * `euroleague_api`
    * `tabulate`

## ⚙️ How the Algorithm Works

The script projects a player's future Performance Index Rating (PIR) using the following weighted factors:

1.  **Base Value (60/40 Split):** Weighs the player's recent game Valuation (60%) against their season-long Average Valuation (40%) using PIR.
2.  **Matchup Coefficient:** Compares the opponent's defense against a specific position to the league average. (e.g., If a team gives up more points to Centers than the league average, opposing Centers get a boost).
3.  **Home Advantage:** Applies a 5% boost for home games and a 5% penalty for away games.
4.  **Team Strength:** Adjusts the projection based on the PIR differential between the player's team and the opposing team.

## 💻 Usage

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the prediction script:
   ```bash
   python src/predict.py --season 2025 --round 37
   ```
