import pandas as pd

def seed_avg(tourney_data, seeds, regular_data):
    seeds["seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))

    seeds_T1 = seeds[["Season", "TeamID", "seed"]].copy()
    seeds_T2 = seeds[["Season", "TeamID", "seed"]].copy()
    seeds_T1.columns = ["Season", "T1_TeamID", "T1_seed"]
    seeds_T2.columns = ["Season", "T2_TeamID", "T2_seed"]

    tourney_data = tourney_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]]
    tourney_data = pd.merge(tourney_data, seeds_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = pd.merge(tourney_data, seeds_T2, on=["Season", "T2_TeamID"], how="left")
    tourney_data["Seed_diff"] = tourney_data["T2_seed"] - tourney_data["T1_seed"]

    boxcols = [
        "T1_Score",
        "T1_FGM",
        "T1_FGA",
        "T1_FGM3",
        "T1_FGA3",
        "T1_FTM",
        "T1_FTA",
        "T1_OR",
        "T1_DR",
        "T1_Ast",
        "T1_TO",
        "T1_Stl",
        "T1_Blk",
        "T1_PF",
        "T2_Score",
        "T2_FGM",
        "T2_FGA",
        "T2_FGM3",
        "T2_FGA3",
        "T2_FTM",
        "T2_FTA",
        "T2_OR",
        "T2_DR",
        "T2_Ast",
        "T2_TO",
        "T2_Stl",
        "T2_Blk",
        "T2_PF",
        "PointDiff",
    ]

    ss = regular_data.groupby(["Season", "T1_TeamID"])[boxcols].agg("mean").reset_index()

    ss_T1 = ss.copy()
    ss_T1.columns = [
        "T1_avg_" + x.replace("T1_", "").replace("T2_", "opponent_")
        for x in list(ss_T1.columns)
    ]
    ss_T1 = ss_T1.rename({"T1_avg_Season": "Season", "T1_avg_TeamID": "T1_TeamID"}, axis=1)
    ss_T2 = ss.copy()
    ss_T2.columns = [
        "T2_avg_" + x.replace("T1_", "").replace("T2_", "opponent_")
        for x in list(ss_T2.columns)
    ]
    ss_T2 = ss_T2.rename({"T2_avg_Season": "Season", "T2_avg_TeamID": "T2_TeamID"}, axis=1)

    tourney_data = pd.merge(tourney_data, ss_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = pd.merge(tourney_data, ss_T2, on=["Season", "T2_TeamID"], how="left")

    return tourney_data, ss_T1, ss_T2
