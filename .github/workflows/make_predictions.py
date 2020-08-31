"""Produce predictions for premier league matches occurring this week."""
import argparse
import datetime
import requests

import bpl
import pandas as pd
import tabulate
from urllib.error import HTTPError


# some teams are inconsistently named between the two data sources
FPL_TO_FOOTBALL_DATA = {
    "Man Utd": "Man United",
    "Sheffield Utd": "Sheffield United",
    "Spurs": "Tottenham"
}


def download_data(now):
    """Download this season and last season's data for training."""
    this_year = int(str(now.year)[2:])

    if now.month >= 8:
        this_season = f"{this_year}{this_year + 1}"
        last_season = f"{this_year -1}{this_year}"
    else:
        this_season = f"{this_year - 1}{this_year}"
        last_season = f"{this_year - 2}{this_year - 1}"

    data_url = lambda season: f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
    this_season_url = data_url(this_season)
    last_season_url = data_url(last_season)

    last_season_results = pd.read_csv(last_season_url)
    try:
        this_season_results = pd.read_csv(this_season_url)
    except HTTPError:
        this_season_results = None

    results = pd.concat((last_season_results, this_season_results))
    results = results[["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"]].rename(
        {
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
        },
        axis=1,
    )

    results.to_csv("training_set.csv", index=False)


def get_fixtures_for_week(now):
    """Use the fantasy football API to get the fixtures that are coming up this week."""
    url = "https://fantasy.premierleague.com/api/fixtures/"
    fixtures = requests.get(url).json()

    this_week_fixtures = []
    for fixture in fixtures:
        kickoff_time = fixture["kickoff_time"]
        if not kickoff_time:
            continue
        elif (
            pd.to_datetime(fixture["kickoff_time"]).date()
            < (now + datetime.timedelta(days=7)).date()
        ) and (pd.to_datetime(fixture["kickoff_time"]).date() >= now.date()):
            this_week_fixtures.append(fixture)

    teams_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    teams = requests.get(teams_url).json()["teams"]

    code_to_team = {team["id"]: FPL_TO_FOOTBALL_DATA.get(team["name"], team["name"]) for team in teams}

    matches = [
        {
            "date": pd.to_datetime(fixture["kickoff_time"]).date(),
            "home_team": code_to_team[fixture["team_h"]],
            "away_team": code_to_team[fixture["team_a"]],
        }
        for fixture in this_week_fixtures
    ]

    if not matches:
        return
    else:
        pd.DataFrame(matches).set_index("date").to_csv("fixtures.csv")


def make_predictions():
    """Make predictions for this gameweek."""

    output_path = f"./pl-predictions/index.md"

    try:
        fixtures = pd.read_csv("fixtures.csv")
    except FileNotFoundError:
        # no matches, so don't make any predictions this week
        return

    # fit the model
    training_set = pd.read_csv("training_set.csv")
    fifa_ratings = pd.read_csv("./.github/workflows/fifa_ratings.csv")
    model = bpl.BPLModel(training_set, X=fifa_ratings)
    model.fit()

    # add missing teams if necessary (first GW of season for promoted teams)
    teams = set(fixtures["home_team"]) | set(fixtures["away_team"])
    model_teams = set(model.team_indices.keys())
    missing_teams = teams - model_teams
    print("Teams missing from model:", missing_teams)

    fifa_ratings = fifa_ratings.copy().set_index("team")
    for team in missing_teams:
        covariates = fifa_ratings.loc[team].values
        model.add_new_team(team, X=covariates)

    # make predictions
    predictions = model.predict_future_matches(fixtures)
    predictions = predictions.rename(
        {
            "pr_home": "Home win",
            "pr_away": "Away win",
            "pr_draw": "Draw",
            "date": "Date",
            "home_team": "Home team",
            "away_team": "Away team",
        },
        axis=1,
    ).round(decimals=2)

    markdown_table = tabulate.tabulate(
        predictions.values, headers=predictions.columns, tablefmt="github"
    )
    print(markdown_table)

    with open(output_path, "r") as f:
        contents = f.readlines()

    new_predictions = (
        f"## Predictions made on {datetime.datetime.now().date()}\n\n"
        + markdown_table
        + "\n\n"
    )

    contents.insert(14, new_predictions)

    with open(output_path, "w") as f:
        f.writelines(contents)

def main(date):
    download_data(date)
    get_fixtures_for_week(date)
    make_predictions()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", dest="date", type=str, default="")
    args = parser.parse_args()

    if not args.date:
        date = datetime.datetime.now()
    else:
        date = pd.to_datetime(args.date)

    main(date)

