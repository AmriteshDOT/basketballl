import pandas as pd
import numpy as np

def update_elo(winner_elo, loser_elo):
    expected_win = expected_result(winner_elo, loser_elo)
    change_in_elo = k_factor * (1 - expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo

def expected_result(elo_a, elo_b):
    return 1.0 / (1 + 10 ** ((elo_b - elo_a) / elo_width))

def compute_elos(regular_data, seeds):
    base_elo = 1000
    elo_width = 400
    k_factor = 100

    elos = []
    for season in sorted(set(seeds["Season"])):
        ss = regular_data.loc[regular_data["Season"] == season]
        ss = ss.loc[ss["win"] == 1].reset_index(drop=True)
        teams = set(ss["T1_TeamID"]) | set(ss["T2_TeamID"])
        elo = dict(zip(teams, [base_elo] * len(teams)))
        for i in range(ss.shape[0]):
            w_team, l_team = ss.loc[i, "T1_TeamID"], ss.loc[i, "T2_TeamID"]
            w_elo, l_elo = elo[w_team], elo[l_team]
            expected_win = 1.0 / (1 + 10 ** ((l_elo - w_elo) / elo_width))
            change_in_elo = k_factor * (1 - expected_win)
            w_elo_new = w_elo + change_in_elo
            l_elo_new = l_elo - change_in_elo
            elo[w_team] = w_elo_new
            elo[l_team] = l_elo_new
        elo_df = pd.DataFrame.from_dict(elo, orient="index").reset_index()
        elo_df = elo_df.rename({"index": "TeamID", 0: "elo"}, axis=1)
        elo_df["Season"] = season
        elos.append(elo_df)
    elos = pd.concat(elos)

    elos_T1 = elos.copy().rename({"TeamID": "T1_TeamID", "elo": "T1_elo"}, axis=1)
    elos_T2 = elos.copy().rename({"TeamID": "T2_TeamID", "elo": "T2_elo"}, axis=1)
    return elos_T1, elos_T2
