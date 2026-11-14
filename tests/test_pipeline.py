"""End-to-end check on the repo's own data: text -> sign video -> text."""
import pytest

from signconnect import config

pytestmark = pytest.mark.skipif(
    not (config.WORD_VIDEO_DIR.is_dir() and config.WORD_MODEL_PATH.exists()),
    reason="sign clip dataset or trained word model not present")


def test_text_to_video_and_back(tmp_path):
    from signconnect.gloss_to_english import glosses_to_english
    from signconnect.recognizer import SignRecognizer
    from signconnect.text_to_sign import text_to_sign_video

    rendered = text_to_sign_video("You are nice.", tmp_path / "out.mp4")
    assert rendered.path.exists() and rendered.duration > 1
    assert rendered.signed_from_video == ["you", "nice"] and not rendered.fingerspelled

    result = SignRecognizer().recognize_video(rendered.path)
    assert result.glosses == ["you", "nice"]
    assert glosses_to_english(result.glosses) == "You are nice."


def test_unknown_word_is_fingerspelled(tmp_path):
    from signconnect.text_to_sign import text_to_sign_video

    rendered = text_to_sign_video("Cab", tmp_path / "out.mp4")
    assert rendered.fingerspelled == ["cab"] and rendered.path.exists()
