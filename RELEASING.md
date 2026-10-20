# Release procedure

Pushing a commit or a version tag does **not** publish a package. Both
publication workflows require a manual action on the `main` branch.

## 1. Prepare GitHub

- Commit and push the changes, then wait for the test matrix to pass.
- Check that `PYPI_API_TOKEN` is a **TestPyPI** token. The historical name is
  retained; it must not contain a real PyPI token.
- `TRUE_PYPI_API_TOKEN` is the separate **PyPI** token, used only by the
  official publication workflow. Never put either token in source files.
- For additional protection, create the GitHub environment `pypi` in
  Settings → Environments, restrict it to `main`, and configure a required
  reviewer where available. The YAML alone does not configure reviewers.
  The PyPI token can be stored as an environment secret there.

## 2. Publish and test on TestPyPI

In Actions, select **Publish and test on TestPyPI**, then **Run workflow**
on `main`. This runs the test matrix, builds the distributions, and sends
them **only to TestPyPI**.

The workflow downloads `nh3pred` exclusively from TestPyPI, checks that the
wheel is byte-identical to the one built, installs it in a new environment
(dependencies from ordinary PyPI), and executes the prediction/reference
tests. It then retains the wheel and source archive for **30 days** as
`testpypi-validated-distributions`. The summary records the source commit,
workflow run ID and archive checksums.

Review the resulting TestPyPI page and the successful workflow. You can
also check the installation yourself in a fresh virtual environment:

```bash
python -m venv /tmp/nh3pred-testpypi-check
source /tmp/nh3pred-testpypi-check/bin/activate
python -m pip download --no-deps --only-binary=:all: \
  --index-url https://test.pypi.org/simple/ --dest /tmp/nh3pred-testpypi-download \
  nh3pred==0.2.0
python -m pip install --index-url https://pypi.org/simple/ \
  /tmp/nh3pred-testpypi-download/nh3pred-0.2.0-py3-none-any.whl
python -m pip check
```

Then run the prediction examples from the README. The separate download
avoids mixing TestPyPI and PyPI when resolving the package and dependencies.

If this version already exists on TestPyPI, choose a new version and repeat
the process. Upload errors are not ignored: the workflow must not silently
mix new archives with files from an older attempt. If an upload succeeds but
the later tests fail, fix the problem and use a new version for the next
attempt. Do not delete/reuse an existing release.

## 3. Stop and review before official publication

Nothing automatically continues from TestPyPI to PyPI. Do not launch the
next workflow until you explicitly intend to publish the release.

When ready, select **Publish validated distributions to PyPI (manual)**:

1. Select `main`.
2. Enter the successful TestPyPI **workflow run ID**.
3. Enter the exact version (for example `0.2.0`).
4. Check the explicit authorization box.
5. Approve the `pypi` environment if reviewers were configured.

The workflow verifies the origin and success of the selected run, downloads
its validated archives, checks the requested version and publishes them
**without rebuilding**. Missing/expired artifacts cause a failure, not an
implicit rebuild. The current branch tip may have moved: the published code
is the source commit recorded by the selected TestPyPI run.

After publication, any release tag should point to that recorded source
commit. Tags do not trigger either publication workflow.

References:
- [TestPyPI guide](https://packaging.python.org/en/latest/guides/using-testpypi/)
- [Downloading artifacts from another workflow run](https://github.com/actions/download-artifact#download-artifacts-from-other-workflow-runs-or-repositories)
