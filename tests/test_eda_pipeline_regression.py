import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from biosppy.signals import eda as eda_mod


# Golden outputs generated from the original committed implementation using
# examples/eda.txt and the same preprocessing path as eda():
# 1) lowpass filter + boxzen smoother
# 2) onsets from eda_events(min_amplitude=0.1, size=0.9)
# 3) biosppy_decomposition(method='onsets', onsets=onsets)
EXPECTED_EDA_BIOSPPY_DECOMP = {
    "filtered_len": 150000,
    "onsets_len": 6,
    "edl_len": 150000,
    "edr_len": 149999,
    "edl_mean": 2463.6402347872736,
    "edl_std": 120.77783637864964,
    "edl_min": 2160.3221392051496,
    "edl_max": 2670.2134939766615,
    "edr_mean": -0.0011230774530110721,
    "edr_std": 0.01907760607250318,
    "edr_min": -0.03233570326548059,
    "edr_max": 0.10433067981975097,
    "edl_first5": [
        2670.2134939766615,
        2670.2040317435667,
        2670.1945695104723,
        2670.1851072773775,
        2670.175645044283,
    ],
    "edl_last5": [
        2501.3045813046515,
        2501.302450651538,
        2501.300319998424,
        2501.2981893453107,
        2501.296058692197,
    ],
    "edr_first5": [
        -0.007027901318284475,
        -0.007038220805603878,
        -0.0070485627354894866,
        -0.007058926774431024,
        -0.0070693125883529795,
    ],
    "edr_last5": [
        -0.002761077038380105,
        -0.0027540995253581358,
        -0.0027471454437913226,
        -0.002740214882164956,
        -0.0027333079284981047,
    ],
}


def _load_eda_example(path="examples/eda.txt"):
    values = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values.append(float(line))

    return np.array(values, dtype=float)


def test_eda_pipeline_regression_examples_eda():
    signal = _load_eda_example()
    sampling_rate = 1000.0

    aux, _, _ = eda_mod.st.filter_signal(
        signal=signal,
        ftype="butter",
        band="lowpass",
        order=4,
        frequency=5,
        sampling_rate=sampling_rate,
    )
    sm_size = int(0.75 * sampling_rate)
    filtered, _ = eda_mod.st.smoother(
        signal=aux,
        kernel="boxzen",
        size=sm_size,
        mirror=True,
    )

    onsets = eda_mod.eda_events(
        signal=filtered,
        sampling_rate=sampling_rate,
        min_amplitude=0.1,
        size=0.9,
    )["onsets"]

    out = eda_mod.biosppy_decomposition(
        signal=filtered,
        sampling_rate=sampling_rate,
        method="onsets",
        onsets=onsets,
    )

    edl = np.array(out["edl"], dtype=float)
    edr = np.array(out["edr"], dtype=float)

    assert len(filtered) == EXPECTED_EDA_BIOSPPY_DECOMP["filtered_len"]
    assert len(onsets) == EXPECTED_EDA_BIOSPPY_DECOMP["onsets_len"]
    assert len(edl) == EXPECTED_EDA_BIOSPPY_DECOMP["edl_len"]
    assert len(edr) == EXPECTED_EDA_BIOSPPY_DECOMP["edr_len"]

    assert np.isclose(np.mean(edl), EXPECTED_EDA_BIOSPPY_DECOMP["edl_mean"], rtol=0.0, atol=1e-10)
    assert np.isclose(np.std(edl), EXPECTED_EDA_BIOSPPY_DECOMP["edl_std"], rtol=0.0, atol=1e-10)
    assert np.isclose(np.min(edl), EXPECTED_EDA_BIOSPPY_DECOMP["edl_min"], rtol=0.0, atol=1e-10)
    assert np.isclose(np.max(edl), EXPECTED_EDA_BIOSPPY_DECOMP["edl_max"], rtol=0.0, atol=1e-10)

    assert np.isclose(np.mean(edr), EXPECTED_EDA_BIOSPPY_DECOMP["edr_mean"], rtol=0.0, atol=1e-10)
    assert np.isclose(np.std(edr), EXPECTED_EDA_BIOSPPY_DECOMP["edr_std"], rtol=0.0, atol=1e-10)
    assert np.isclose(np.min(edr), EXPECTED_EDA_BIOSPPY_DECOMP["edr_min"], rtol=0.0, atol=1e-10)
    assert np.isclose(np.max(edr), EXPECTED_EDA_BIOSPPY_DECOMP["edr_max"], rtol=0.0, atol=1e-10)

    assert np.allclose(edl[:5], EXPECTED_EDA_BIOSPPY_DECOMP["edl_first5"], rtol=0.0, atol=1e-10)
    assert np.allclose(edl[-5:], EXPECTED_EDA_BIOSPPY_DECOMP["edl_last5"], rtol=0.0, atol=1e-10)
    assert np.allclose(edr[:5], EXPECTED_EDA_BIOSPPY_DECOMP["edr_first5"], rtol=0.0, atol=1e-10)
    assert np.allclose(edr[-5:], EXPECTED_EDA_BIOSPPY_DECOMP["edr_last5"], rtol=0.0, atol=1e-10)
