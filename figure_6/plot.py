"""Figure 6: 6-by-5 M curves from two sweeps and their matching endpoints."""

from plot_finite_time import build_model


from build_record import prepare, record
from manuscript_assets import INPUTS


def build():
    build_model("M")


if __name__ == "__main__":
    prepare()
    build()
    record(INPUTS)
