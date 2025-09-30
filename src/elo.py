import pandas as pd
import numpy as np

base_elo = 1000
elo_width = 400
k_factor = 100


def update(w_elo, l_elo):
    exp = expected(w_elo, l_elo)
    delta = k_factor * (1 - exp)
    w_elo += delta
    l_elo -= delta
    return w_elo, l_elo


def expected(a, b):
    return 1.0 / (1 + 10 ** ((b - a) / elo_width))


def compute_elos(reg, seeds):
    all_elos = []
    for season in sorted(set(seeds["Season"])):
        ss = reg.loc[reg["Season"] == season]
        ss = ss.loc[ss["win"] == 1].reset_index(drop=True)
        teams = set(ss["T1_TeamID"]) | set(ss["T2_TeamID"])
        elos = dict(zip(teams, [base_elo] * len(teams)))
        for i in range(ss.shape[0]):
            w_team = ss.loc[i, "T1_TeamID"]
            l_team = ss.loc[i, "T2_TeamID"]
            w_elo = elos[w_team]
            l_elo = elos[l_team]
            exp = expected(w_elo, l_elo)
            delta = k_factor * (1 - exp)
            w_new = w_elo + delta
            l_new = l_elo - delta
            elos[w_team] = w_new
            elos[l_team] = l_new
        df = pd.DataFrame.from_dict(elos, orient="index").reset_index()
        df = df.rename({"index": "TeamID", 0: "elo"}, axis=1)
        df["Season"] = season
        all_elos.append(df)
    all_elos = pd.concat(all_elos)

    elos_t1 = all_elos.copy().rename({"TeamID": "T1_TeamID", "elo": "T1_elo"}, axis=1)
    elos_t2 = all_elos.copy().rename({"TeamID": "T2_TeamID", "elo": "T2_elo"}, axis=1)
    return elos_t1, elos_t2


if __name__ == "__main__":
    pass
