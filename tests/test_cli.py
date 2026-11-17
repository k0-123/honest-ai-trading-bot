from nsebot.cli import main


def test_live_refuses_with_sebi_reminder(capsys):
    assert main(["live"]) == 3
    out = capsys.readouterr().out
    assert "Algo-ID" in out and "static IP" in out and "NOT BUILT" in out


def test_reset_halt_requires_confirm(capsys):
    assert main(["reset-halt", "--book", "baseline"]) == 2
