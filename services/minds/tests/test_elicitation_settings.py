from pathlib import Path

import pytest

from twl_minds.elicitation.settings import ElicitSettings

MODEL = "TWL_MINDS_ELICIT_MODEL"
RUN = "TWL_MINDS_ELICIT_RUN_CAP_USD"
DAILY = "TWL_MINDS_ELICIT_DAILY_CAP_USD"
STORE = "TWL_MINDS_ELICIT_STORE"
MAX = "TWL_MINDS_ELICIT_MAX_TOKENS"


def test_defaults_apply_to_an_empty_environment() -> None:
    settings = ElicitSettings.from_env({})
    assert settings.model == "claude-haiku-5-5"
    assert settings.run_cap_usd == 1.0
    assert settings.daily_cap_usd == 5.0
    assert Path(settings.store_path) == Path("data/cache/minds/elicitations.duckdb")
    assert settings.max_tokens == 1024


def test_every_variable_is_read() -> None:
    settings = ElicitSettings.from_env(
        {MODEL: "claude-haiku-5-5", RUN: "0.25", DAILY: "2.5", STORE: "/tmp/x.duckdb", MAX: "2048"}
    )
    assert settings.run_cap_usd == 0.25
    assert settings.daily_cap_usd == 2.5
    assert Path(settings.store_path) == Path("/tmp/x.duckdb")
    assert settings.max_tokens == 2048


def test_unrelated_variables_are_ignored_and_the_api_key_is_not_retained() -> None:
    settings = ElicitSettings.from_env({"ANTHROPIC_API_KEY": "sk-secret-123", "PATH": "/bin"})
    assert "sk-secret-123" not in repr(settings)


def test_an_unknown_model_is_refused_naming_the_variable() -> None:
    with pytest.raises(ValueError, match=MODEL):
        ElicitSettings.from_env({MODEL: "claude-unpriced-9"})


def test_a_blank_model_is_refused() -> None:
    with pytest.raises(ValueError, match=MODEL):
        ElicitSettings.from_env({MODEL: " "})


@pytest.mark.parametrize("bad", ["abc", "", "0", "-1", "nan", "inf", "-inf"])
def test_run_cap_refusals_name_the_variable(bad: str) -> None:
    with pytest.raises(ValueError, match=RUN):
        ElicitSettings.from_env({RUN: bad})


@pytest.mark.parametrize("bad", ["abc", "", "0", "-1", "nan", "inf", "-inf"])
def test_daily_cap_refusals_name_the_variable(bad: str) -> None:
    with pytest.raises(ValueError, match=DAILY):
        ElicitSettings.from_env({DAILY: bad})


def test_run_cap_above_the_daily_cap_is_refused() -> None:
    with pytest.raises(ValueError, match=r"RUN_CAP|DAILY_CAP"):
        ElicitSettings.from_env({RUN: "3", DAILY: "2"})


def test_run_cap_above_the_default_daily_cap_is_refused() -> None:
    with pytest.raises(ValueError, match=r"RUN_CAP|DAILY_CAP"):
        ElicitSettings.from_env({RUN: "5.01"})


def test_run_cap_equal_to_the_daily_cap_is_allowed() -> None:
    settings = ElicitSettings.from_env({RUN: "2", DAILY: "2"})
    assert settings.run_cap_usd == settings.daily_cap_usd == 2.0


@pytest.mark.parametrize("bad", ["abc", "", "1.5", "255", "0", "-1", "16001"])
def test_max_tokens_refusals_name_the_variable(bad: str) -> None:
    with pytest.raises(ValueError, match=MAX):
        ElicitSettings.from_env({MAX: bad})


@pytest.mark.parametrize("good", ["256", "16000"])
def test_max_tokens_bounds_are_inclusive(good: str) -> None:
    assert ElicitSettings.from_env({MAX: good}).max_tokens == int(good)


def test_from_env_defaults_to_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MAX, "512")
    monkeypatch.delenv(RUN, raising=False)
    monkeypatch.delenv(DAILY, raising=False)
    monkeypatch.delenv(MODEL, raising=False)
    assert ElicitSettings.from_env().max_tokens == 512
