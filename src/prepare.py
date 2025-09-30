import pandas as pd
import numpy as np


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

    adj = (40 + 5 * df["NumOT"]) / 40
    cols = [
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

    for c in cols:
        df[c] = df[c] / adj

    df_swap = df.copy()
    df.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in list(df.columns)]
    df_swap.columns = [
        x.replace("L", "T1_").replace("W", "T2_") for x in list(df_swap.columns)
    ]
    out = pd.concat([df, df_swap]).reset_index(drop=True)
    out["PointDiff"] = out["T1_Score"] - out["T2_Score"]
    out["win"] = (out["PointDiff"] > 0) * 1
    out["men_women"] = (out["T1_TeamID"].apply(lambda t: str(t).startswith("1"))) * 1
    return out


def build(reg, tour):
    reg_data = prepare_data(reg)
    tour_data = prepare_data(tour)
    return reg_data, tour_data
