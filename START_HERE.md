# Start here: run the Disaster Tweets notebook and app on your computer

This guide is for anyone, including people who have never installed Python. You **don't** need
Python, Git, VS Code, a Kaggle account or a Streamlit account. You double-click one file, and it
sets everything up.

**What you get**

- **The notebook**: a step-by-step walk-through of how the model that tells real disaster tweets
  from figures of speech is built, that you can run, change and re-run in your web browser.
- **The app**: the interactive Disaster Tweets NLP app (type any tweet and see the prediction,
  plus the data, the features and the model explained) running on your own computer.

**What you need**

- Windows 10 or 11, or macOS 12 or newer (Apple silicon or Intel).
- An internet connection the first time (it downloads a few hundred MB).
- About 2 GB of free disk space.

---

## Windows

### 1. Download the project

1. Go to <https://github.com/nhamhung/disaster-tweets-nlp>.
2. Click the green **Code** button, then **Download ZIP**.
3. Open your **Downloads** folder, right-click `disaster-tweets-nlp-main.zip` and choose
   **Extract All...**. Extract it to a simple place such as
   `C:\Users\<you>\Documents\disaster-tweets-nlp-main`.

> Don't run it from inside the ZIP file, and avoid folders synced by OneDrive if you can
> (they work, but setup is slower).

### 2. Start it

1. Open the extracted folder and double-click **`start_windows.bat`**.
2. If Windows shows **"Windows protected your PC"**, click **More info**, then **Run anyway**.
   (This appears because the file was downloaded from the internet.)
3. A black window opens. **The first time only**, it installs *uv* (a small tool that sets up
   Python for this project) and downloads Python and the packages. This takes about 2–5 minutes.
   Later starts take a few seconds.
4. When it's ready, you see a menu:

   ```
   === Disaster Tweets NLP ===
     1  Open the notebooks (JupyterLab)
     2  Open the app (Streamlit)
     3  Quit
   Type 1, 2 or 3 and press Enter:
   ```

5. If Windows Firewall asks whether to allow Python, click **Allow** (it only talks to your own
   computer).

### 3. Stop it

Click the black window and press **Ctrl + C**. If it asks *"Terminate batch job (Y/N)?"*, type
**Y** and press Enter. Or simply close the black window.

---

## macOS

### 1. Download the project

1. Go to <https://github.com/nhamhung/disaster-tweets-nlp>.
2. Click the green **Code** button, then **Download ZIP**.
3. Open your **Downloads** folder and double-click `disaster-tweets-nlp-main.zip` to unzip it.
   Move the `disaster-tweets-nlp-main` folder somewhere simple, such as your **Documents** folder.

### 2. Start it

1. Open the folder. **The first time**, right-click (or Control-click) **`start_mac.command`**,
   choose **Open**, then click **Open** again in the warning. (macOS asks because the file was
   downloaded from the internet. After this, a normal double-click works.)
   - If macOS says it *"cannot be opened because Apple cannot check it"* and there is no Open
     button: open **System Settings → Privacy & Security**, scroll down, click **Open Anyway**
     next to `start_mac.command`, then try again.
2. A Terminal window opens. **The first time only**, it installs *uv* (a small tool that sets up
   Python for this project) and downloads Python and the packages. This takes about 2–5 minutes.
   Later starts take a few seconds.
   On most Macs the first start also prints *"One-time fix: downloading the
   OpenMP runtime..."*: that is a small library (about 300 kB) the model needs on macOS, and it is
   stored inside the project folder.
3. When it's ready, you see the same menu as on Windows: type **1** (notebooks) or **2** (app)
   and press Return.

### 3. Stop it

Click the Terminal window and press **Control + C**, then close the window.

---

## Using the notebook (choice 1)

JupyterLab opens in your web browser.

1. In the file list on the left, double-click **`01_eda_and_modeling.ipynb`**.
2. Run it from top to bottom with **Run → Run All Cells**. The whole notebook takes about 1–3
   minutes depending on your computer. A `[*]` next to a cell means it is still running.
3. To run just one cell, click it and press **Shift + Enter**.
4. **Experiment freely.** Change a number in a cell and run it again. The notebook ends with
   suggested exercises ("try changing X and see Y").
5. If things get into a strange state, use **Kernel → Restart Kernel and Run All Cells**.

The kernel (the Python behind the notebook) is already chosen for you: **Python 3 (ipykernel)**.
Your edits are saved in the notebook file. To keep the original, first use **File → Save
Notebook As...** and work on the copy (or download the ZIP again later).

## Using the app (choice 2)

The app opens in your browser by itself (if it doesn't, copy the `http://localhost:...` address
shown in the black/Terminal window into your browser). Pages:

- **Predict**: type or paste any tweet (optionally a keyword and a location) and see whether the
  model thinks it's about a real disaster, with the probability. Or load a random real tweet and
  compare with the label a person gave it.
- **Dataset Overview**, **Feature Engineering** and **Model Insights**: the data, how a tweet
  becomes numbers, and how the models compare. The first visit to Model Insights takes a little
  while (it trains and compares the models live).

Nothing you do in the app changes the project files.

## Next time

Just double-click `start_windows.bat` or `start_mac.command` again. No internet needed once the
first setup has finished.

---

## Optional: use VS Code instead of the browser

If you prefer VS Code for the notebook:

1. Run the start file once (above), so the project's Python environment (the `.venv` folder)
   exists. You can quit from the menu.
2. Install VS Code from <https://code.visualstudio.com>.
3. In VS Code: **File → Open Folder...** and choose the project folder. Accept the prompt to
   install the recommended **Python** and **Jupyter** extensions.
4. Open `notebooks/01_eda_and_modeling.ipynb`. Click **Select Kernel** (top right) →
   **Python Environments** → the one marked **`.venv`** (Recommended).

That is the only kernel step, and you do it once.

On a Mac, VS Code doesn't get the launcher's OpenMP fix: if a cell fails with *"Library not
loaded: @rpath/libomp.dylib"*, either use the launcher's JupyterLab instead, or install
[Homebrew](https://brew.sh) and run `brew install libomp` in Terminal once.

---

## Optional: the full dataset from Kaggle

The project includes a 1,000-tweet sample, so the notebook works immediately without Kaggle.
For the complete dataset (`data/raw/tweets.csv`, 11,370 tweets), the launcher can download it with
a free Kaggle API token. Results on the sample will differ from the full-data numbers in the text:

1. Create a free account at <https://www.kaggle.com> and sign in.
2. Open <https://www.kaggle.com/settings/api> (or click your profile picture → **Settings** →
   **API**) and click **Generate New Token**. Kaggle shows a long token text: click the copy
   button.
3. Save that text in a file named `access_token` (no file extension) in a folder called
   `.kaggle` in your user folder:
   - **Windows:** open *Notepad*, paste the token, choose **File → Save As...**, set
     **Save as type** to **All files**, and save it as
     `C:\Users\<you>\.kaggle\access_token` (create the `.kaggle` folder, with the dot, in the
     Save dialog if it doesn't exist; `%USERPROFILE%` typed in the address bar takes you to your
     user folder).
   - **macOS:** open *Terminal* and run these lines, pasting your token in place of
     `PASTE-YOUR-TOKEN-HERE` (this creates the hidden folder and the file and makes it private
     to you):

     ```
     mkdir -p ~/.kaggle
     echo "PASTE-YOUR-TOKEN-HERE" > ~/.kaggle/access_token
     chmod 600 ~/.kaggle/access_token
     ```
   - If you have an older `kaggle.json` file from Kaggle (a "legacy API key"), putting it in the
     same `.kaggle` folder works too.
4. Double-click the start file again and choose **1**. The launcher finds the token and asks
   *"Download the full dataset now?"*: type **y** and press Enter.

This dataset is a normal Kaggle *dataset* (`vstepanenko/disaster-tweets`), so there are no
competition rules to accept. The project's separate Kaggle *competition* scripts (for submitting
to the leaderboard) do need you to join the competition first; see the README.

---

## If something goes wrong

| Problem | What to do |
|---|---|
| macOS: *"Library not loaded: @rpath/libomp.dylib"* or *"brew install libomp"* | The launcher normally fixes this by itself (it needs the internet once). If it says the fix didn't work, install [Homebrew](https://brew.sh), run `brew install libomp` in Terminal, and start again. |
| Installing uv fails (often on work computers that block scripts) | Install uv another way, then double-click the start file again. **Windows:** open *Command Prompt* and run `winget install --id=astral-sh.uv -e`. **macOS:** if you use Homebrew, run `brew install uv` in Terminal. Or follow <https://docs.astral.sh/uv/getting-started/installation/>. |
| macOS: *"permission denied"* or the file opens in a text editor | Open **Terminal**, type `bash ` (with a space), drag `start_mac.command` into the Terminal window, and press Return. |
| The browser didn't open | Copy the `http://localhost:...` address from the black/Terminal window into your browser. |
| *"Address already in use"* | Another copy is probably running. Close the other black/Terminal window, or just pick the option again: the launcher picks a free port. |
| The launcher says the notebook's full dataset is missing | The notebook still runs on a 1,000-tweet sample. To get the full file, see [Optional: the full dataset from Kaggle](#optional-the-full-dataset-from-kaggle). |
| Errors after editing the notebook | **Kernel → Restart Kernel and Run All Cells**. Still broken? Download the ZIP again for a fresh copy. |
| Something else is broken | Delete the `.venv` folder inside the project and double-click the start file again: it rebuilds a clean setup. |

## Uninstall

1. Delete the project folder.
2. Optional, to remove uv and the downloaded Python too:
   - **Windows:** delete `uv.exe` and `uvx.exe` from `C:\Users\<you>\.local\bin`, the folder
     `%APPDATA%\uv`, and the folder `%LOCALAPPDATA%\uv`.
   - **macOS:** in Terminal, run `rm -rf ~/.local/bin/uv ~/.local/bin/uvx ~/.local/share/uv ~/.cache/uv`.
3. Optional: delete the `.kaggle` folder in your user folder if you created one.

---

## For developers

The project is a standard [uv](https://docs.astral.sh/uv/) project: `pyproject.toml` lists the
direct dependencies at the exact versions it was tested with, and `uv.lock` pins everything else.
(On Intel Macs only, numpy is pinned one version lower, 2.3.5, because SHAP needs numba and
numba's last Intel-Mac release needs numpy < 2.4.)

```bash
uv sync                                   # create .venv (add --no-dev to skip pytest/ruff)
uv run pytest                             # tests
uv run jupyter lab notebooks              # notebooks
uv run streamlit run app/streamlit_app.py # app
```

`scripts/launch.py` is what the start files run. `requirements.txt` and `app/requirements.txt`
stay for Streamlit Community Cloud. The leaderboard transformer model
(`scripts/make_transformer_submission.py`) is **not** installed by the start files: it needs
PyTorch and `transformers` (`requirements-transformer.txt`; the torch wheel alone is ~125 MB to
download and ~580 MB installed on a Mac, and has no build for Intel Macs), downloads the
`cardiffnlp/twitter-roberta-base` model from Hugging Face on first run (internet needed), and needs
the Kaggle competition files. Run it with
`uv run --with torch==2.14.1 --with transformers==5.18.0 python scripts/make_transformer_submission.py`.
To publish your own changes to GitHub, see the setup manual at
<https://github.com/nhamhung/nhamhung.github.io/blob/main/README.md> (Git, VS Code, clone, push).
