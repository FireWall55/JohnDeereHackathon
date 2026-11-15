"""The upload path: saved model, supplier references and photo processing."""
import io
import pickle

import numpy as np
import pytest
from PIL import Image, ImageDraw

from earnedpass import config as C
from earnedpass import deploy, models, reference
from tests.test_pipeline import synthetic


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = synthetic(reps=8)
    path = tmp_path_factory.mktemp("m") / "screen_model.pkl"
    model = deploy.build(d, log=lambda *_: None, path=path)
    return d, model, path


def test_upload_features_match_the_training_features(built):
    d, model, _ = built
    s = d.samples
    is_ref = reference.split(s)
    open_rows = np.flatnonzero(~s.locked.to_numpy())
    X = reference.RefBlocks(d, models.Spectra(d.X_ftir), open_rows, is_ref).X(deploy.SENSORS)
    lab = s[["L", "a", "b"]].to_numpy(float)
    for code, ref in model.examples.items():
        rows = open_rows[s.code.to_numpy()[open_rows] == code]
        assert np.allclose(model.features(lab[rows], d.X_img[rows], ref), X[rows])


def test_predictions_are_well_formed_and_far_points_are_outside(built):
    d, model, path = built
    s = d.samples
    ref = model.examples["P"]
    rows = np.flatnonzero((s.code == "P").to_numpy() & ~s.locked.to_numpy())
    out = model.predict(s[["L", "a", "b"]].to_numpy(float)[rows], d.X_img[rows], ref)
    assert np.allclose(out["p_band"].sum(1), 1) and np.allclose(out["p_type"].sum(1), 1)
    assert (out["level"] >= 0).all() and (out["level"] <= 24).all()
    assert out["outside"].mean() < 0.05  # training-like samples are inside
    far = model.predict([np.asarray(ref.lab) + 40], np.asarray(ref.img) * 3, ref)
    assert far["outside"][0]
    assert isinstance(pickle.loads(path.read_bytes()), deploy.ScreenModel)  # it reloads from disk


def dish_photo(colour=(150, 95, 55)) -> bytes:
    im = Image.new("RGB", (1200, 900), (235, 238, 240))
    ImageDraw.Draw(im).ellipse([250, 100, 950, 800], fill=colour)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_photo_features_find_the_dish_and_reject_bad_photos():
    feats, preview = deploy.photo_features(dish_photo())
    assert feats.shape == (19,) and np.isfinite(feats).all()
    assert preview.size[0] <= 512
    blank = io.BytesIO()
    Image.new("RGB", (800, 600), (240, 240, 240)).save(blank, format="PNG")
    with pytest.raises(ValueError):
        deploy.photo_features(blank.getvalue())
    with pytest.raises(ValueError):
        deploy.photo_features(b"not an image")


def test_suppliers_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "SUPPLIERS_PATH", tmp_path / "suppliers.json")
    assert deploy.load_suppliers() == {}
    ref = deploy.Reference(name="Test Co", lab=[50.0, 10.0, 20.0], img=[0.0] * 19, n_photos=3, n_readings=2)
    deploy.save_suppliers({ref.name: ref})
    back = deploy.load_suppliers()["Test Co"]
    assert back.lab == ref.lab and back.n_photos == 3 and not back.example


def test_saved_model_exists_for_users_without_the_dataset():
    model = deploy.load_model()
    assert model is not None, "models/screen_model.pkl is missing: run python -m earnedpass.deploy"
    assert set(model.examples) == set(C.FOLDERS)
