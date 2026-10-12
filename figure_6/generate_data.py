"""Generate the experiments belonging to this figure, using only local files."""

import argparse
from run_finite_time import run, run_large
from run_experiments import benchmark, response_experiment
from analyze_mechanisms import run as working_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        choices=["empirical", "large", "steady", "working-points", "timing"],
    )
    args = parser.parse_args()
    if args.experiment == "empirical":
        run()
    elif args.experiment == "large":
        run_large()
    elif args.experiment == "steady":
        response_experiment([100, 500, 1000])
    elif args.experiment == "working-points":
        working_points()
    else:
        benchmark([100, 500, 1000, 2000])


if __name__ == "__main__":
    main()
