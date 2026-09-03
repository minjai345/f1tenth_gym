import pytest, textwrap
from camsim import config

def test_default_loads():
    cfg = config.load()
    assert cfg.camera.image_width == 640
    assert cfg.camera.image_height == 400
    assert cfg.lane.track_width_m == 0.8
    assert cfg.waypoints.ahead_m == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert cfg.render.lidar_fov_rad == pytest.approx(4.7)

def test_missing_key_names_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        camera: {image_width: 640}
    """))
    with pytest.raises(config.ConfigError) as e:
        config.load(str(p))
    assert "camera.image_height" in str(e.value)

def test_unknown_key_names_key(tmp_path):
    src = open(config.DEFAULT_PATH).read() + "\nlane_typo: 1\n"
    p = tmp_path / "c.yaml"; p.write_text(src)
    with pytest.raises(config.ConfigError) as e:
        config.load(str(p))
    assert "lane_typo" in str(e.value)
