import pandas as pd


def add_seed_and_avgs(tour, s, reg):
    s["seed"] = s["Seed"].apply(lambda x: int(x[1:3]))

    s_t1 = s[["Season", "TeamID", "seed"]].copy()
    s_t2 = s[["Season", "TeamID", "seed"]].copy()
    s_t1.columns = ["Season", "T1_TeamID", "T1_seed"]
    s_t2.columns = ["Season", "T2_TeamID", "T2_seed"]

    tour = tour[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]]
    tour = pd.merge(tour, s_t1, on=["Season", "T1_TeamID"], how="left")
    tour = pd.merge(tour, s_t2, on=["Season", "T2_TeamID"], how="left")
    tour["Seed_diff"] = tour["T2_seed"] - tour["T1_seed"]

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

    ss = reg.groupby(["Season", "T1_TeamID"])[boxcols].agg("mean").reset_index()

    ss_t1 = ss.copy()
    ss_t1.columns = [
        "T1_avg_" + x.replace("T1_", "").replace("T2_", "opponent_")
        for x in list(ss_t1.columns)
    ]
    ss_t1 = ss_t1.rename(
        {"T1_avg_Season": "Season", "T1_avg_TeamID": "T1_TeamID"}, axis=1
    )

    ss_t2 = ss.copy()
    ss_t2.columns = [
        "T2_avg_" + x.replace("T1_", "").replace("T2_", "opponent_")
        for x in list(ss_t2.columns)
    ]
    ss_t2 = ss_t2.rename(
        {"T2_avg_Season": "Season", "T2_avg_TeamID": "T2_TeamID"}, axis=1
    )

    tour = pd.merge(tour, ss_t1, on=["Season", "T1_TeamID"], how="left")
    tour = pd.merge(tour, ss_t2, on=["Season", "T2_TeamID"], how="left")

    return tour, ss_t1, ss_t2


if __name__ == "__main__":
    pass
