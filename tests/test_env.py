import pytest
from sgi.vlm.env import MissingKeyError, load_env, require_key

def test_load_env_reads_pairs_and_skips_comments_and_blanks(tmp_path, capsys):
    path = tmp_path / ".env"
    path.write_text('GEMINI_API_KEY=abc-123\n# GEMINI_API_KEY=an-old-key\n\nOPENAI_API_KEY="sk-quoted"\n'
                    "  SPACED = 'single quoted'  \nURL=https://host/path?a=1&b=2\nEMPTY=\nnot a pair\n")
    assert load_env(path) == {"GEMINI_API_KEY": "abc-123", "OPENAI_API_KEY": "sk-quoted", "SPACED": "single quoted",
                              "URL": "https://host/path?a=1&b=2", "EMPTY": ""}
    assert load_env(tmp_path / "absent.env") == {} and require_key("OPENAI_API_KEY", path) == "sk-quoted"
    assert capsys.readouterr() == ("", "")                            # a key is never printed

def test_missing_or_empty_key_names_the_file_and_the_variable(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("GEMINI_API_KEY=\nOPENAI_API_KEY=sk-secret\n")    # .env.example copied, one key filled in
    monkeypatch.setenv("GEMINI_API_KEY", "from-the-shell")            # the shell is not a source of keys
    for env_path in (path, tmp_path / "absent.env"):
        with pytest.raises(MissingKeyError) as err:
            require_key("GEMINI_API_KEY", env_path)
        assert str(env_path) in str(err.value) and "GEMINI_API_KEY" in str(err.value)
        assert "sk-secret" not in str(err.value)
