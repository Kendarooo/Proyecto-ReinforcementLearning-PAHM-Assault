# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

import numpy as np

from pahm_stage2.unsupervised_model import GMMWindModel


def _clustered_feature_matrix(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [0.0, 0.01, 0.02, 0.0, 0.0001, 0.0001],
            [0.08, 0.12, 0.8, 2.0, 0.08, 0.05],
            [0.5, 0.02, 0.55, 0.0, 0.0005, 0.25],
            [0.0, 0.35, 0.9, 0.0, 0.20, 0.12],
        ],
        dtype=np.float32,
    )
    return np.vstack(
        [center + rng.normal(0.0, 0.005, size=(12, centers.shape[1])) for center in centers]
    )


def test_bic_selects_correct_n_components_on_clean_data():
    model = GMMWindModel(
        n_components_range=(2, 6),
        covariance_type="full",
        random_state=42,
    )

    result = model.fit(_clustered_feature_matrix())

    assert result.selected_n_components == 4
    assert model.predict(_clustered_feature_matrix()).shape == (48,)
    assert set(result.bic_scores) == {2, 3, 4, 5, 6}


def test_model_saves_and_loads_correctly(tmp_path):
    feature_matrix = _clustered_feature_matrix()
    model = GMMWindModel(
        n_components_range=(2, 6),
        covariance_type="full",
        random_state=42,
    )
    model.fit(feature_matrix)
    checkpoint_path = tmp_path / "gmm_wind_model.pkl"

    model.save(str(checkpoint_path))
    loaded = GMMWindModel.load(str(checkpoint_path))

    assert checkpoint_path.exists()
    np.testing.assert_array_equal(model.predict(feature_matrix), loaded.predict(feature_matrix))
    assert loaded.selected_n_components == model.selected_n_components

