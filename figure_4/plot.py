"""Figure 4: 6-by-3 E curves and their matching endpoint ranks and gains."""

from plot_finite_time import build_model


from build_record import prepare, record
from manuscript_assets import INPUTS


def build():
    build_model("E")


if __name__ == "__main__":
    prepare()
    build()
    record(INPUTS)
