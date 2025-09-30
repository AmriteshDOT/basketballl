import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, brier_score_loss
from scipy.interpolate import UnivariateSpline


def run(tour, ss_t1, ss_t2, s_t1, s_t2, e_t1, e_t2):
    tour = pd.merge(tour, e_t1, on=["Season", "T1_TeamID"], how="left")
    tour = pd.merge(tour, e_t2, on=["Season", "T2_TeamID"], how="left")
    tour["elo_diff"] = tour["T1_elo"] - tour["T2_elo"]

    feat = [
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

    import joblib

    params = joblib.load("best_params.joblib")

    models = {}
    oof_mae = []
    oof_preds = []
    oof_targets = []
    oof_ss = []

    for oof_season in set(tour.Season):
        x_train = tour.loc[tour["Season"] != oof_season, feat].values
        y_train = tour.loc[tour["Season"] != oof_season, "PointDiff"].values
        x_val = tour.loc[tour["Season"] == oof_season, feat].values
        y_val = tour.loc[tour["Season"] == oof_season, "PointDiff"].values
        s_val = tour.loc[tour["Season"] == oof_season, "Season"].values

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        models[oof_season] = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=2000,
            evals=[(dval, "validation")],
            early_stopping_rounds=50,
            # verbose_eval=True,
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
            "men_women": tour["men_women"],
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
        df.clip(-30, 30)
        .groupby("pred_pointdiff")[["spline", "label"]]
        .mean()
        .reset_index()
    )

    #############  Predictions  ###############
    X = pd.read_csv(f"Submit.csv")
    X["Season"] = X["ID"].apply(lambda t: int(t.split("_")[0]))
    X["T1_TeamID"] = X["ID"].apply(lambda t: int(t.split("_")[1]))
    X["T2_TeamID"] = X["ID"].apply(lambda t: int(t.split("_")[2]))
    X["men_women"] = X["T1_TeamID"].apply(lambda t: 0 if str(t)[0] == "1" else 1)
    X = pd.merge(X, ss_t1, on=["Season", "T1_TeamID"], how="left")
    X = pd.merge(X, ss_t2, on=["Season", "T2_TeamID"], how="left")
    X = pd.merge(X, s_t1, on=["Season", "T1_TeamID"], how="left")
    X = pd.merge(X, s_t2, on=["Season", "T2_TeamID"], how="left")
    X = pd.merge(X, e_t1, on=["Season", "T1_TeamID"], how="left")
    X = pd.merge(X, e_t2, on=["Season", "T2_TeamID"], how="left")
    X["Seed_diff"] = X["T2_seed"] - X["T1_seed"]
    X["elo_diff"] = X["T1_elo"] - X["T2_elo"]
    preds = []
    for oof_season in set(tour.Season):
        dtest = xgb.DMatrix(X[feat].values)
        margin_preds = models[oof_season].predict(dtest) * 1.0
        probs = np.clip(spline_model(np.clip(margin_preds, -t, t)), 0.01, 0.99)
        preds.append(probs)
    X["Pred"] = np.array(preds).mean(axis=0)
    return X
