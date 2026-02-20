from pala.planner.identity import load_identity_text


def test_identity_loader_reads_file(tmp_path):
    path = tmp_path / "identity.md"
    path.write_text("I am PALA.\n", encoding="utf-8")
    text = load_identity_text(str(path), "fallback")
    assert text == "I am PALA."


def test_identity_loader_uses_fallback_when_missing(tmp_path):
    text = load_identity_text(str(tmp_path / "missing.md"), "fallback identity")
    assert text == "fallback identity"
