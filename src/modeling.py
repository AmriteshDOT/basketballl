import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, brier_score_loss
from scipy.interpolate import UnivariateSpline

def modelling(tourney_data, ss_T1, ss_T2, seeds_T1, seeds_T2, elos_T1, elos_T2):
    tourney_data = pd.merge(tourney_data, elos_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = pd.merge(tourney_data, elos_T2, on=["Season", "T2_TeamID"], how="left")
    tourney_data["elo_diff"] = tourney_data["T1_elo"] - tourney_data["T2_elo"]

    features = [
        "men_women",
        "T1_seed",
        "T2_seed",
        "Seed_diff",
        "T1_avg_Score",
        "T1_avg_FGA",
        "T1_avg_OR",
        "T1_avg_DR",
        "T1_avg_Blk",
        "T1_avg_PF",
        "T1_avg_opponent_FGA",
        "T1_avg_opponent_Blk",
        "T1_avg_opponent_PF",
        "T1_avg_PointDiff",
        "T2_avg_Score",
        "T2_avg_FGA",
        "T2_avg_OR",
        "T2_avg_DR",
        "T2_avg_Blk",
        "T2_avg_PF",
        "T2_avg_opponent_FGA",
        "T2_avg_opponent_Blk",
        "T2_avg_opponent_PF",
        "T2_avg_PointDiff",
        "T1_elo",
        "T2_elo",
        "elo_diff",
    ]

    param = {}
    param["objective"] = "reg:squarederror"
    param["booster"] = "gbtree"
    param["eta"] = 0.0093
    param["subsample"] = 0.6
    param["colsample_bynode"] = 0.8
    param["num_parallel_tree"] = 2
    param["min_child_weight"] = 4
    param["max_depth"] = 4
    param["tree_method"] = "hist"
    param["grow_policy"] = "lossguide"
    param["max_bin"] = 38

    num_rounds = 704

    models = {}
    oof_mae = []
    oof_preds = []
    oof_targets = []
    oof_ss = []

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

    plt_df = df

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
    preds = []
    for oof_season in set(tourney_data.Season):
        dtest = xgb.DMatrix(X[features].values)
        margin_preds = (
            models[oof_season].predict(dtest) * 1.0
        )
        probs = np.clip(spline_model(np.clip(margin_preds, -t, t)), 0.01, 0.99)
        preds.append(probs)
    X["Pred"] = np.array(preds).mean(axis=0)
    return X
