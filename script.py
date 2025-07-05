import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
M_regular_results = pd.read_csv("MRegularSeasonDetailedResults.csv")
M_tourney_results = pd.read_csv("MNCAATourneyDetailedResults.csv")
M_seeds = pd.read_csv("MNCAATourneySeeds.csv")

W_regular_results = pd.read_csv("WRegularSeasonDetailedResults.csv")
W_tourney_results = pd.read_csv("WNCAATourneyDetailedResults.csv")
W_seeds = pd.read_csv("WNCAATourneySeeds.csv")

#####
regular_results = pd.concat([M_regular_results, W_regular_results])
tourney_results = pd.concat([M_tourney_results, W_tourney_results])
seeds = pd.concat([M_seeds, W_seeds])

##########
season = 2003  # temporalchanges thres
regular_results = regular_results.loc[regular_results["Season"] >= season]
tourney_results = tourney_results.loc[tourney_results["Season"] >= season]
seeds = seeds.loc[seeds["Season"] >= season]


##########
# Sample data and visualizations
season = 2024
teamid = 3163

r = regular_results.loc[
    (regular_results["Season"] == season)
    & ((regular_results["WTeamID"] == teamid) | (regular_results["LTeamID"] == teamid))
]
t = tourney_results.loc[
    (tourney_results["Season"] == season)
    & ((tourney_results["WTeamID"] == teamid) | (tourney_results["LTeamID"] == teamid))
]
r["win"] = np.where(r["WTeamID"] == teamid, "win", "lose")
t["win"] = np.where(t["WTeamID"] == teamid, "win", "lose")
r["type"] = "regular season"
t["type"] = "tournament"

rt = pd.concat([r, t])
rt[["DayNum", "WScore", "LScore", "type", "win"]]

############
# seed check
s = W_seeds.loc[W_seeds["Season"] == 2015]
[s.loc[s["Seed"].str.startswith(d)] for d in ("X", "Y", "Z", "W")]


####################
def prepare_data(df):
    df = df[
        [
            "Season",
            "DayNum",
            "LTeamID",
            "LScore",
            "WTeamID",
            "WScore",
            "NumOT",
            "LFGM",
            "LFGA",
            "LFGM3",
            "LFGA3",
            "LFTM",
            "LFTA",
            "LOR",
            "LDR",
            "LAst",
            "LTO",
            "LStl",
            "LBlk",
            "LPF",
            "WFGM",
            "WFGA",
            "WFGM3",
            "WFGA3",
            "WFTM",
            "WFTA",
            "WOR",
            "WDR",
            "WAst",
            "WTO",
            "WStl",
            "WBlk",
            "WPF",
        ]
    ]

    adjot = (40 + 5 * df["NumOT"]) / 40
    adjcols = [
        "LScore",
        "WScore",
        "LFGM",
        "LFGA",
        "LFGM3",
        "LFGA3",
        "LFTM",
        "LFTA",
        "LOR",
        "LDR",
        "LAst",
        "LTO",
        "LStl",
        "LBlk",
        "LPF",
        "WFGM",
        "WFGA",
        "WFGM3",
        "WFGA3",
        "WFTM",
        "WFTA",
        "WOR",
        "WDR",
        "WAst",
        "WTO",
        "WStl",
        "WBlk",
        "WPF",
    ]
    for col in adjcols:
        df[col] = df[col] / adjot

    dfswap = df.copy()
    df.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in list(df.columns)]
    dfswap.columns = [
        x.replace("L", "T1_").replace("W", "T2_") for x in list(dfswap.columns)
    ]
    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output["PointDiff"] = output["T1_Score"] - output["T2_Score"]
    output["win"] = (output["PointDiff"] > 0) * 1
    output["men_women"] = (
        output["T1_TeamID"].apply(lambda t: str(t).startswith("1"))
    ) * 1
    return output


regular_data = prepare_data(regular_results)
tourney_data = prepare_data(tourney_results)

######### check
season = regular_data["Season"] == 2025
t1, t2 = 1182, 1433
match1 = (regular_data["T1_TeamID"] == t1) & (regular_data["T2_TeamID"] == t2)
match2 = (regular_data["T1_TeamID"] == t2) & (regular_data["T2_TeamID"] == t1)
regular_data.loc[season & (match1 | match2)]


###########  Feature engineering
seeds["seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))

seeds_T1 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T2 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T1.columns = ["Season", "T1_TeamID", "T1_seed"]
seeds_T2.columns = ["Season", "T2_TeamID", "T2_seed"]

tourney_data = tourney_data[
    ["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]
]
tourney_data = pd.merge(tourney_data, seeds_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, seeds_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["Seed_diff"] = tourney_data["T2_seed"] - tourney_data["T1_seed"]

#####  EDA (seed v pdiff)
import matplotlib.pyplot as plt
import seaborn

### line plot and std bands
##  mean and standard dev
tmpmean = tourney_data.pivot_table(
    columns="men_women", index="T1_seed", values="PointDiff", aggfunc="mean"
).ffill()
tmpstd = tourney_data.pivot_table(
    columns="men_women", index="T1_seed", values="PointDiff", aggfunc="std"
).ffill()
fig, axis = plt.subplots(ncols=2, figsize=(12, 4))
(line_1,) = axis[0].plot(tmpmean.index, tmpmean[0], "b-")
fill_1 = axis[0].fill_between(
    tmpmean.index, tmpmean[0] - tmpstd[0], tmpmean[0] + tmpstd[0], color="b", alpha=0.1
)
(line_2,) = axis[1].plot(tmpmean.index, tmpmean[1], "r--")
fill_2 = axis[1].fill_between(
    tmpmean.index, tmpmean[1] - tmpstd[1], tmpmean[1] + tmpstd[1], color="r", alpha=0.1
)
plt.margins(x=0)
plt.legend([(line_1, fill_1), (line_2, fill_2)], ["Women", "Men"])


# let's seed_diff vs pdiff
tmpmean = tourney_data.pivot_table(
    columns="men_women", index="Seed_diff", values="PointDiff", aggfunc="mean"
).ffill()
tmpstd = tourney_data.pivot_table(
    columns="men_women", index="Seed_diff", values="PointDiff", aggfunc="std"
).ffill()
fig, axis = plt.subplots(ncols=2, figsize=(12, 4))
(line_1,) = axis[0].plot(tmpmean.index, tmpmean[0], "b-")
fill_1 = axis[0].fill_between(
    tmpmean.index, tmpmean[0] - tmpstd[0], tmpmean[0] + tmpstd[0], color="b", alpha=0.1
)
(line_2,) = axis[1].plot(tmpmean.index, tmpmean[1], "r--")
fill_2 = axis[1].fill_between(
    tmpmean.index, tmpmean[1] - tmpstd[1], tmpmean[1] + tmpstd[1], color="r", alpha=0.1
)
plt.margins(x=0)
plt.legend([(line_1, fill_1), (line_2, fill_2)], ["Women", "Men"])


#######################################
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
]  # filtered


# season averages
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

# average team and oppont data

########  elo


def update_elo(winner_elo, loser_elo):
    expected_win = expected_result(winner_elo, loser_elo)
    change_in_elo = k_factor * (1 - expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo


def expected_result(elo_a, elo_b):
    return 1.0 / (1 + 10 ** ((elo_b - elo_a) / elo_width))


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
        w_elo_new, l_elo_new = update_elo(w_elo, l_elo)
        elo[w_team] = w_elo_new
        elo[l_team] = l_elo_new
    elo = pd.DataFrame.from_dict(elo, orient="index").reset_index()
    elo = elo.rename({"index": "TeamID", 0: "elo"}, axis=1)
    elo["Season"] = season
    elos.append(elo)
elos = pd.concat(elos)

elos_T1 = elos.copy().rename({"TeamID": "T1_TeamID", "elo": "T1_elo"}, axis=1)
elos_T2 = elos.copy().rename({"TeamID": "T2_TeamID", "elo": "T2_elo"}, axis=1)
tourney_data = pd.merge(tourney_data, elos_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, elos_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["elo_diff"] = tourney_data["T1_elo"] - tourney_data["T2_elo"]

####  EDAs

## ML model
features = [
    "men_women",
    "T1_seed",
    "T2_seed",
    "Seed_diff",
    "T1_avg_Score",
    # "T1_avg_FGM",
    "T1_avg_FGA",
    # "T1_avg_FGM3",
    # "T1_avg_FGA3",
    # "T1_avg_FTM",
    # "T1_avg_FTA",
    "T1_avg_OR",
    "T1_avg_DR",
    # "T1_avg_Ast",
    # "T1_avg_TO",
    # "T1_avg_Stl",
    "T1_avg_Blk",
    "T1_avg_PF",
    # "T1_avg_opponent_Score",
    # "T1_avg_opponent_FGM",
    "T1_avg_opponent_FGA",
    # "T1_avg_opponent_FGM3",
    # "T1_avg_opponent_FGA3",
    # "T1_avg_opponent_FTM",
    # "T1_avg_opponent_FTA",
    # "T1_avg_opponent_OR",
    # "T1_avg_opponent_DR",
    # "T1_avg_opponent_Ast",
    # "T1_avg_opponent_TO",
    # "T1_avg_opponent_Stl",
    "T1_avg_opponent_Blk",
    "T1_avg_opponent_PF",
    "T1_avg_PointDiff",
    "T2_avg_Score",
    # "T2_avg_FGM",
    "T2_avg_FGA",
    # "T2_avg_FGM3",
    # "T2_avg_FGA3",
    # "T2_avg_FTM",
    # "T2_avg_FTA",
    "T2_avg_OR",
    "T2_avg_DR",
    # "T2_avg_Ast",
    # "T2_avg_TO",
    # "T2_avg_Stl",
    "T2_avg_Blk",
    "T2_avg_PF",
    # "T2_avg_opponent_Score",
    # "T2_avg_opponent_FGM",
    "T2_avg_opponent_FGA",
    # "T2_avg_opponent_FGM3",
    # "T2_avg_opponent_FGA3",
    # "T2_avg_opponent_FTM",
    # "T2_avg_opponent_FTA",
    # "T2_avg_opponent_OR",
    # "T2_avg_opponent_DR",
    # "T2_avg_opponent_Ast",
    # "T2_avg_opponent_TO",
    # "T2_avg_opponent_Stl",
    "T2_avg_opponent_Blk",
    "T2_avg_opponent_PF",
    "T2_avg_PointDiff",
    "T1_elo",
    "T2_elo",
    "elo_diff",
]

import xgboost as xgb

param = {}
param["objective"] = "reg:squarederror"
param["booster"] = "gbtree"
param["eta"] = 0.0093  # 0.009 #0.01
param["subsample"] = 0.6
param["colsample_bynode"] = 0.8
param["num_parallel_tree"] = 2
param["min_child_weight"] = 4
param["max_depth"] = 4
param["tree_method"] = "hist"
param["grow_policy"] = "lossguide"
param["max_bin"] = 38  # 32

num_rounds = 704  # 704 #700


from sklearn.metrics import mean_absolute_error, brier_score_loss

models = {}
oof_mae = []
oof_preds = []
oof_targets = []
oof_ss = []

# leave-one-season out models
for oof_season in set(tourney_data.Season):
    x_train = tourney_data.loc[tourney_data["Season"] != oof_season, features].values
    y_train = tourney_data.loc[tourney_data["Season"] != oof_season, "PointDiff"].values
    x_val = tourney_data.loc[tourney_data["Season"] == oof_season, features].values
    y_val = tourney_data.loc[tourney_data["Season"] == oof_season, "PointDiff"].values
    s_val = tourney_data.loc[tourney_data["Season"] == oof_season, "Season"].values

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    models[oof_season] = xgb.train(
        params=param,
        dtrain=dtrain,
        num_boost_round=num_rounds,
    )
    preds = models[oof_season].predict(dval)
    print(f"oof season {oof_season} mae: {mean_absolute_error(y_val, preds)}")
    oof_mae.append(mean_absolute_error(y_val, preds))
    oof_preds += list(preds)
    oof_targets += list(y_val)
    oof_ss += list(s_val)

print(f"average mae: {np.mean(oof_mae)}")

####
df = pd.DataFrame(
    {
        "Season": oof_ss,
        "pred": oof_preds,
        "label": [(t > 0) * 1 for t in oof_targets],
        "men_women": tourney_data["men_women"],
    }
)
df["pred_pointdiff"] = df["pred"].astype(int)

xdf_all = (
    df.clip(-30, 30)
    .groupby("pred_pointdiff")["label"]
    .mean()
    .reset_index(name="average_win_pct")
)
xdf_men = (
    df.clip(-30, 30)
    .loc[df["men_women"] == 0]
    .groupby("pred_pointdiff")["label"]
    .mean()
    .reset_index(name="average_win_pct")
)
xdf_women = (
    df.clip(-30, 30)
    .loc[df["men_women"] == 1]
    .groupby("pred_pointdiff")["label"]
    .mean()
    .reset_index(name="average_win_pct")
)

seaborn.lineplot(x=xdf_all["pred_pointdiff"], y=xdf_all["average_win_pct"])
seaborn.lineplot(x=xdf_men["pred_pointdiff"], y=xdf_men["average_win_pct"])
seaborn.lineplot(x=xdf_women["pred_pointdiff"], y=xdf_women["average_win_pct"])
from scipy.interpolate import UnivariateSpline

t = 25
dat = list(zip(oof_preds, np.array(oof_targets) > 0))
dat = sorted(dat, key=lambda x: x[0])
pred, label = list(zip(*dat))
spline_model = UnivariateSpline(np.clip(pred, -t, t), label, k=5)
spline_fit = np.clip(spline_model(np.clip(oof_preds, -t, t)), 0.01, 0.99)
print(f"brier: {brier_score_loss(np.array(oof_targets)>0, spline_fit)}")
df["spline"] = spline_fit
xdf = (
    df.clip(-30, 30).groupby("pred_pointdiff")[["spline", "label"]].mean().reset_index()
)

plt.figure()
plt.plot(xdf["pred_pointdiff"], xdf["label"])
plt.plot(xdf["pred_pointdiff"], xdf["spline"])
print(f"brier: {brier_score_loss(np.array(oof_targets)>0, spline_fit)}")

for oof_season in set(tourney_data.Season):
    x = df.loc[df["Season"] == oof_season, "spline"].values
    y = df.loc[df["Season"] == oof_season, "label"].values
    print(oof_season, np.round(brier_score_loss(y, x), 5))

# construct dataframe for submission
X = pd.read_csv(f"Submit.csv")
X["Season"] = X["ID"].apply(lambda t: int(t.split("_")[0]))
X["T1_TeamID"] = X["ID"].apply(lambda t: int(t.split("_")[1]))
X["T2_TeamID"] = X["ID"].apply(lambda t: int(t.split("_")[2]))
X["men_women"] = X["T1_TeamID"].apply(lambda t: 0 if str(t)[0] == "1" else 1)
X = pd.merge(X, ss_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, ss_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, seeds_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, seeds_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, elos_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, elos_T2, on=["Season", "T2_TeamID"], how="left")
X["Seed_diff"] = X["T2_seed"] - X["T1_seed"]
X["elo_diff"] = X["T1_elo"] - X["T2_elo"]
# run models on given dataset
preds = []
for oof_season in set(tourney_data.Season):
    dtest = xgb.DMatrix(X[features].values)
    margin_preds = (
        models[oof_season].predict(dtest) * 1.0
    )  # aggressive submissions >1, conservative submissions <1
    probs = np.clip(spline_model(np.clip(margin_preds, -t, t)), 0.01, 0.99)
    preds.append(probs)
X["Pred"] = np.array(preds).mean(axis=0)
