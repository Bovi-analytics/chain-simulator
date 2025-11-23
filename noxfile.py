import nox


python_versions = ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14"]

@nox.session(python=python_versions, venv_backend="uv")
def tests(session: nox.Session) -> None:
    """Run tests on specified Python versions."""
    # Install the package and test dependencies with uv
    session.run_install(
        "uv",
        "sync",
        "--locked",
        "--group=tests",
        "--no-default-groups",
        "--quiet",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )

    # Run pytest with common options
    session.run(
        "pytest",
        "tests/",
        "-v",                   # verbose output
        "-s",                   # don't capture output
        "--tb=short",           # shorter traceback format
        "--strict-markers",     # treat unregistered markers as errors
        *session.posargs        # allows passing additional pytest args from command line
    )
