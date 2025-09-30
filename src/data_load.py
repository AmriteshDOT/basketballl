import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def load_mixed_ncaa_data():
    m_reg = pd.read_csv("m_reg_season.csv")
    m_tour = pd.read_csv("m_tourney.csv")
    m_seeds = pd.read_csv("m_seeds.csv")

    w_reg = pd.read_csv("w_reg_season.csv")
    w_tour = pd.read_csv("w_tourney.csv")
    w_seeds = pd.read_csv("w_seeds.csv")

    reg = pd.concat([m_reg, w_reg], ignore_index=True)
    tour = pd.concat([m_tour, w_tour], ignore_index=True)
    seeds = pd.concat([m_seeds, w_seeds], ignore_index=True)

    start = 2003
    reg = reg.loc[reg["Season"] >= start]
    tour = tour.loc[tour["Season"] >= start]
    seeds = seeds.loc[seeds["Season"] >= start]

    return reg, tour, seeds


if __name__ == "__main__":
    reg, tour, seeds = load_mixed_ncaa_data()
