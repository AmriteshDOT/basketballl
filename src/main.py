from data_load import load
from prepare import build
from feature_engineering import add_seed_and_averages
from elo import compute_elos
from modeling import run_models


def main():
    regular_results, tourney_results, seeds = load()
    regular_data, tourney_data = build(regular_results, tourney_results)
    tourney_data, ss_T1, ss_T2 = add_seed_and_averages(
        tourney_data, seeds, regular_data
    )
    seeds_T1 = seeds[["Season", "TeamID", "Seed"]].copy()
    seeds_T2 = seeds[["Season", "TeamID", "Seed"]].copy()
    seeds_T1.columns = ["Season", "T1_TeamID", "T1_seed"]
    seeds_T2.columns = ["Season", "T2_TeamID", "T2_seed"]
    elos_T1, elos_T2 = compute_elos(regular_data, seeds)
    X = run_models(tourney_data, ss_T1, ss_T2, seeds_T1, seeds_T2, elos_T1, elos_T2)


if __name__ == "__main__":
    main()
