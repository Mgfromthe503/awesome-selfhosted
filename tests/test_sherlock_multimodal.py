import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sherlock_multimodal import SherlockTTS, SherlockVision


def test_tts_stub_output(tmp_path):
    tts = SherlockTTS()
    out = tmp_path / "tts.txt"
    result = tts.speak_to_file("hello sherlock", str(out))
    assert result.ok
    assert out.exists()


def test_vision_missing_file():
    vision = SherlockVision()
    result = vision.analyze_image("does_not_exist.png")
    assert not result.ok
