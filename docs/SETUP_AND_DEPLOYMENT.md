# Setup and Deployment

## Local setup

```bash
git clone https://github.com/nhamhhung/disaster-tweets-nlp.git
cd disaster-tweets-nlp
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r app/requirements.txt
pytest tests/
python scripts/smoke_app.py
streamlit run app/streamlit_app.py
```

The trained model is included, while the app obtains its source dataset through the Kaggle API.

## Deploy your fork

1. Fork or clone this repository and replace the owner and URLs in `docs/deployment-config.yml`.
2. Push to your own public GitHub repository with `main` as the default branch.
3. In Settings → Pages, select **GitHub Actions** as the source.
4. In Streamlit Community Cloud, create an app from your repository, branch `main`, file `app/streamlit_app.py`.
5. Configure Kaggle access using either KAGGLE_API_TOKEN or a [kaggle] section with username and key.
6. Wait for CI and Pages to pass, then record the URLs and revision in `docs/DEPLOYMENT_ACCEPTANCE.md`.

Do not copy another owner's tokens or URL values. GitHub Pages hosts the static report; Streamlit Community Cloud runs the interactive Python app.

## Required checks

Protect `main` after bootstrap. Require `ruff`, `pytest (3.11)`, `pytest (3.12)`, `smoke (3.11)`, `smoke (3.12)`, and `pages-build`. Disable force pushes and direct pushes; use up-to-date pull requests and squash merges.

## Rollback

Revert later commits on a `rollback/<date>` branch, pass all required checks, and merge normally. Verify both Pages and Streamlit before closing the rollback.
