# data_loader.py
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def load_mixed_ncaa_data():
    M_regular_results = pd.read_csv("MRegularSeasonDetailedResults.csv")
    M_tourney_results = pd.read_csv("MNCAATourneyDetailedResults.csv")
    M_seeds = pd.read_csv("MNCAATourneySeeds.csv")

    W_regular_results = pd.read_csv("WRegularSeasonDetailedResults.csv")
    W_tourney_results = pd.read_csv("WNCAATourneyDetailedResults.csv")
    W_seeds = pd.read_csv("WNCAATourneySeeds.csv")

    regular_results = pd.concat(
        [M_regular_results, W_regular_results], ignore_index=True
    )
    tourney_results = pd.concat(
        [M_tourney_results, W_tourney_results], ignore_index=True
    )
    seeds = pd.concat([M_seeds, W_seeds], ignore_index=True)
    # season (temporal)
    start_season = 2003
    regular_results = regular_results.loc[regular_results["Season"] >= start_season]
    tourney_results = tourney_results.loc[tourney_results["Season"] >= start_season]
    seeds = seeds.loc[seeds["Season"] >= start_season]

    return regular_results, tourney_results, seeds


if __name__ == "__main__":
    reg, tourney, seeds = load_mixed_ncaa_data()
