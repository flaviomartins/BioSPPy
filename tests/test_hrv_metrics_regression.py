import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from biosppy.signals import hrv as hrv_mod


# Golden outputs generated from the original committed implementation
# using examples/rri.txt and:
# hrv(rri=rri, parameters='all', features_only=True, show=False, show_individual=False)
EXPECTED_HRV_METRICS = {
    "appen": 0.979039959952277,
    "hf_nu": 0.5204339397007272,
    "hf_peak": 0.3359375,
    "hf_pwr": 417.786715133925,
    "hf_rpwr": 0.4197459803914782,
    "hr_max": 116.36363636363636,
    "hr_mean": 99.11551458618632,
    "hr_median": 105.20547945205479,
    "hr_min": 61.935483870967744,
    "hr_minmax": 54.428152492668616,
    "hti": 8.88888888888889,
    "lf_hf": 0.9214734545849274,
    "lf_nu": 0.47956606029927284,
    "lf_peak": 0.125,
    "lf_pwr": 384.97936767414683,
    "lf_rpwr": 0.38678477859947263,
    "nn50": 48.0,
    "pnn50": 10.020876826722338,
    "rmssd": 33.540352679878566,
    "rr_max": 968.75,
    "rr_mean": 623.33984375,
    "rr_median": 570.3125,
    "rr_min": 515.625,
    "rr_minmax": 453.125,
    "s": 3395.6829463347835,
    "sampen": 1.0946213295328995,
    "sd1": 23.71647344524039,
    "sd12": 0.5203828341381185,
    "sd2": 45.575049539289054,
    "sd21": 1.9216621579307955,
    "sdnn": 36.31289684883323,
    "tinn": 101.5625,
    "ulf_peak": 0.0,
    "ulf_pwr": 1.6687253719383748,
    "ulf_rpwr": 0.0016765510770821786,
    "vhf_peak": 0.4296875,
    "vhf_pwr": 48.59081798722302,
    "vhf_rpwr": 0.04881869096180513,
    "vlf_peak": 0.0390625,
    "vlf_pwr": 120.23707286913603,
    "vlf_rpwr": 0.12080094029480772,
}


def _load_rri_example(path="examples/rri.txt"):
    values = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values.append(float(line))

    return np.array(values, dtype=float)


def test_hrv_metrics_regression_examples_rri():
    rri = _load_rri_example()

    current = hrv_mod.hrv(
        rri=rri,
        parameters="all",
        features_only=True,
        show=False,
        show_individual=False,
    )

    assert set(current.keys()) == set(EXPECTED_HRV_METRICS.keys())

    # Tiny tolerance allows platform-level floating point noise while
    # preventing behavioral regressions in HRV metrics.
    for key, expected in EXPECTED_HRV_METRICS.items():
        got = float(current[key])
        assert np.isclose(got, expected, rtol=0.0, atol=1e-7), key
