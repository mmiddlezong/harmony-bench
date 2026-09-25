from typer.testing import CliRunner

from harmonybench import cli


def test_build_refuses_without_a_seed(monkeypatch):
    """The real test set's seed lives only in .env; never fall back to a public default."""
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv(cli.SEED_ENV, raising=False)
    called = []
    monkeypatch.setattr("harmonybench.build.generate", lambda *a: called.append(a) or [])
    result = CliRunner().invoke(cli.app, ["build"])
    assert result.exit_code == 1 and "No seed" in result.output
    assert not called


def test_build_reads_seed_from_env(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv(cli.SEED_ENV, "12345")
    seeds = []
    monkeypatch.setattr("harmonybench.build.generate", lambda out, per_chord, seed: seeds.append(seed) or [])
    CliRunner().invoke(cli.app, ["build"])
    CliRunner().invoke(cli.app, ["build", "--seed", "7"])
    assert seeds == [12345, 7]
