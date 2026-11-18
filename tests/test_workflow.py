"""The GitHub Actions daily-run workflow keeps its key safety properties."""
from pathlib import Path

import yaml

WF = Path(".github/workflows/daily-paper.yml")


def load():
    return yaml.safe_load(WF.read_text())


def test_schedule_is_1630_ist_on_weekdays_and_manual_trigger_exists():
    wf = load()
    on = wf.get("on") or wf.get(True)          # PyYAML reads the bare key `on` as True
    assert on["schedule"] == [{"cron": "0 11 * * 1-5"}]
    assert "workflow_dispatch" in on


def test_api_key_only_from_secrets_and_not_hard_coded():
    text = WF.read_text()
    assert "secrets.GOOGLE_API_KEY" in text
    for line in text.splitlines():
        if "GOOGLE_API_KEY:" in line:
            assert "secrets.GOOGLE_API_KEY" in line or "''" in line


def test_ai_book_off_for_first_five_runs_and_needs_explicit_switch():
    text = WF.read_text()
    assert '-ge 5' in text and "vars.ENABLE_AI_BOOK" in text and "--no-llm" in text


def test_state_is_saved_and_failures_or_halts_fail_the_job():
    steps = load()["jobs"]["daily"]["steps"]
    names = [s.get("name", "") for s in steps]
    commit = steps[names.index("Commit state and reports")]
    fail = steps[names.index("Fail if the run failed or a book is HALTED")]
    assert commit.get("if") == "always()" and "git push" in commit["run"]
    assert fail.get("if") == "always()" and "HALTED_" in fail["run"]
    assert "status=1" in fail["run"] and "exit $status" in fail["run"]
    assert load()["permissions"] == {"contents": "write"}
