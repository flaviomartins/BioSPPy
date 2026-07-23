import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from biosppy.signals import ecg as ecg_mod


# Golden outputs generated from the original committed implementation using
# examples/ecg.txt and:
# ecg(signal=signal, sampling_rate=1000.0, show=False, interactive=False)
EXPECTED_ECG_PIPELINE = {
    "signal_len": 15000,
    "ts_len": 15000,
    "filtered_len": 15000,
    "rpeaks_len": 15,
    "templates_shape": (15, 600),
    "hr_len": 14,
    "templates_ts_len": 600,
    "hr_ts_len": 14,
    "filtered_mean": -1.152026622245709e-15,
    "filtered_std": 56.63559974721516,
    "filtered_min": -62.50468057476873,
    "filtered_max": 425.7426804014808,
    "templates_mean": 3.695343935104208,
    "templates_std": 72.6369138273858,
    "templates_min": -62.50468057476873,
    "templates_max": 425.7426804014808,
    "hr_mean": 60.600424173736805,
    "hr_std": 1.292766429329584,
    "hr_min": 59.45751716652278,
    "hr_max": 64.37346158000352,
    "rpeaks_first10": [283, 1204, 2159, 3188, 4211, 5188, 6200, 7232, 8200, 9157],
    "rpeaks_last10": [5188, 6200, 7232, 8200, 9157, 10156, 11198, 12159, 13139, 14162],
    "filtered_first5": [
        -0.501995488116215,
        2.354600198988578,
        5.110195975872713,
        7.669102405139263,
        9.945935790366129,
    ],
    "filtered_last5": [
        12.313790377557645,
        9.52653327432651,
        6.384019822602106,
        2.9931236063940503,
        -0.525515715003209,
    ],
    "hr_first5": [
        64.37346158000352,
        62.094280945441646,
        59.92909647490886,
        59.45751716652278,
        59.78401704936694,
    ],
    "hr_last5": [
        60.112519573767834,
        60.02553251200594,
        60.41367575729203,
        60.77015992282828,
        59.508847527280714,
    ],
    "templates_row0_first5": [
        -1.1968512359958368,
        -1.6873498346137268,
        -2.132165836263262,
        -2.5028113459518977,
        -2.774249802087624,
    ],
    "templates_row0_last5": [
        -29.618176355915324,
        -29.320153939028515,
        -28.99264670503338,
        -28.635340472471317,
        -28.249393183842514,
    ],
}


def _load_ecg_example(path="examples/ecg.txt"):
    values = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values.append(float(line))

    return np.array(values, dtype=float)


def test_ecg_pipeline_regression_examples_ecg():
    signal = _load_ecg_example()

    out = ecg_mod.ecg(
        signal=signal,
        sampling_rate=1000.0,
        show=False,
        interactive=False,
    )

    ts = np.array(out["ts"], dtype=float)
    filtered = np.array(out["filtered"], dtype=float)
    rpeaks = np.array(out["rpeaks"], dtype=int)
    templates_ts = np.array(out["templates_ts"], dtype=float)
    templates = np.array(out["templates"], dtype=float)
    hr_ts = np.array(out["heart_rate_ts"], dtype=float)
    hr = np.array(out["heart_rate"], dtype=float)

    assert len(signal) == EXPECTED_ECG_PIPELINE["signal_len"]
    assert len(ts) == EXPECTED_ECG_PIPELINE["ts_len"]
    assert len(filtered) == EXPECTED_ECG_PIPELINE["filtered_len"]
    assert len(rpeaks) == EXPECTED_ECG_PIPELINE["rpeaks_len"]
    assert templates.shape == EXPECTED_ECG_PIPELINE["templates_shape"]
    assert len(hr) == EXPECTED_ECG_PIPELINE["hr_len"]
    assert len(templates_ts) == EXPECTED_ECG_PIPELINE["templates_ts_len"]
    assert len(hr_ts) == EXPECTED_ECG_PIPELINE["hr_ts_len"]

    assert np.isclose(filtered.mean(), EXPECTED_ECG_PIPELINE["filtered_mean"], rtol=0.0, atol=1e-10)
    assert np.isclose(filtered.std(), EXPECTED_ECG_PIPELINE["filtered_std"], rtol=0.0, atol=1e-10)
    assert np.isclose(filtered.min(), EXPECTED_ECG_PIPELINE["filtered_min"], rtol=0.0, atol=1e-10)
    assert np.isclose(filtered.max(), EXPECTED_ECG_PIPELINE["filtered_max"], rtol=0.0, atol=1e-10)

    assert np.isclose(templates.mean(), EXPECTED_ECG_PIPELINE["templates_mean"], rtol=0.0, atol=1e-10)
    assert np.isclose(templates.std(), EXPECTED_ECG_PIPELINE["templates_std"], rtol=0.0, atol=1e-10)
    assert np.isclose(templates.min(), EXPECTED_ECG_PIPELINE["templates_min"], rtol=0.0, atol=1e-10)
    assert np.isclose(templates.max(), EXPECTED_ECG_PIPELINE["templates_max"], rtol=0.0, atol=1e-10)

    assert np.isclose(hr.mean(), EXPECTED_ECG_PIPELINE["hr_mean"], rtol=0.0, atol=1e-10)
    assert np.isclose(hr.std(), EXPECTED_ECG_PIPELINE["hr_std"], rtol=0.0, atol=1e-10)
    assert np.isclose(hr.min(), EXPECTED_ECG_PIPELINE["hr_min"], rtol=0.0, atol=1e-10)
    assert np.isclose(hr.max(), EXPECTED_ECG_PIPELINE["hr_max"], rtol=0.0, atol=1e-10)

    assert np.array_equal(rpeaks[:10], np.array(EXPECTED_ECG_PIPELINE["rpeaks_first10"], dtype=int))
    assert np.array_equal(rpeaks[-10:], np.array(EXPECTED_ECG_PIPELINE["rpeaks_last10"], dtype=int))

    assert np.allclose(filtered[:5], EXPECTED_ECG_PIPELINE["filtered_first5"], rtol=0.0, atol=1e-10)
    assert np.allclose(filtered[-5:], EXPECTED_ECG_PIPELINE["filtered_last5"], rtol=0.0, atol=1e-10)
    assert np.allclose(hr[:5], EXPECTED_ECG_PIPELINE["hr_first5"], rtol=0.0, atol=1e-10)
    assert np.allclose(hr[-5:], EXPECTED_ECG_PIPELINE["hr_last5"], rtol=0.0, atol=1e-10)
    assert np.allclose(templates[0, :5], EXPECTED_ECG_PIPELINE["templates_row0_first5"], rtol=0.0, atol=1e-10)
    assert np.allclose(templates[0, -5:], EXPECTED_ECG_PIPELINE["templates_row0_last5"], rtol=0.0, atol=1e-10)
