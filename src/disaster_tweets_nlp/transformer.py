"""The leaderboard model: a 5-fold ensemble of `cardiffnlp/twitter-roberta-base`
(RoBERTa pretrained on tweets), fine-tuned on the competition's own
train set. Public LB F1 0.84339, vs. 0.67269 for the TF-IDF model in
`model.py` — see the README's "Climbing the leaderboard" section.

Deliberately separate from `model.py`, not a replacement for it:

- The TF-IDF + LightGBM model stays the one the notebook, the app, and
  SHAP explain — it trains in seconds and its features are words a
  person can read. This model needs `torch` + `transformers` (install
  `requirements-transformer.txt`, ~1GB) and its features are 768
  anonymous hidden dimensions.
- Never import this module in the same process as `lightgbm` *after*
  `torch` has loaded: on macOS their bundled OpenMP runtimes collide
  and the process segfaults (verified directly). `torch` is imported
  lazily inside the functions below so importing this module alone is
  harmless.

Input handling follows the public recipe this reproduces: `keyword` and
the *raw* tweet text are encoded as a sentence pair, with no URL/@mention
stripping — the model was pretrained on raw tweets, so it already knows
those patterns.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from . import config
from .features import clean_keyword


def build_pair_inputs(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(keywords, texts) for sentence-pair encoding. `keyword` gets the
    same URL-decoding fix as the TF-IDF model (a real data bug, not
    stylistic cleanup); the text is passed through unchanged.
    """
    keywords = X[config.KEYWORD_COL].apply(clean_keyword).tolist()
    texts = X[config.TEXT_COL].astype(str).tolist()
    return keywords, texts


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_loader(tokenizer, keywords, texts, labels=None, batch_size=64, shuffle=False):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    enc = tokenizer(
        keywords,
        texts,
        truncation=True,
        padding=True,
        max_length=config.TRANSFORMER_MAX_LENGTH,
        return_tensors="pt",
    )
    tensors = [enc["input_ids"], enc["attention_mask"]]
    if labels is not None:
        tensors.append(torch.tensor(labels))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def _predict_disaster_proba(model, loader, device) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            out.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy())
    return np.concatenate(out)


def _fine_tune(model, loader, device, epochs: int) -> None:
    import torch
    from transformers import get_linear_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.TRANSFORMER_LEARNING_RATE, weight_decay=config.TRANSFORMER_WEIGHT_DECAY
    )
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(config.TRANSFORMER_WARMUP_FRACTION * total_steps), total_steps
    )
    model.train()
    for _ in range(epochs):
        for input_ids, attention_mask, labels in loader:
            optimizer.zero_grad()
            loss = model(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device)
            ).loss
            loss.backward()
            optimizer.step()
            scheduler.step()


def train_kfold_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    n_folds: int = config.TRANSFORMER_N_FOLDS,
    epochs: int = config.TRANSFORMER_EPOCHS,
    folds_dir: Path = config.TRANSFORMER_FOLDS_DIR,
    log=print,
) -> tuple[np.ndarray, np.ndarray]:
    """Fine-tune one fresh model per stratified fold. Returns
    (out-of-fold "Disaster" probabilities for every training row,
    test-set probabilities averaged across all fold models).

    Each fold's predictions are cached to `folds_dir` as soon as it
    finishes (a full run is ~30 min on an Apple M4), so an interrupted
    run resumes from the last finished fold instead of starting over.
    Delete `folds_dir` to force a clean retrain.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(config.RANDOM_SEED)
    device = _device()
    folds_dir.mkdir(parents=True, exist_ok=True)

    labels = (y == "Disaster").astype(int).to_numpy()
    keywords, texts = (np.array(v, dtype=object) for v in build_pair_inputs(X))
    test_keywords, test_texts = build_pair_inputs(X_test)

    tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
    test_loader = _make_loader(tokenizer, test_keywords, test_texts)

    oof = np.full(len(labels), np.nan)
    test_probas = []
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED)

    for fold, (tr, va) in enumerate(splitter.split(texts, labels), start=1):
        fold_file = folds_dir / f"fold{fold}_of_{n_folds}_e{epochs}.npz"
        if fold_file.exists():
            cached = np.load(fold_file)
            oof[va], test_proba = cached["val_proba"], cached["test_proba"]
            log(f"Fold {fold}/{n_folds}: loaded cached predictions from {fold_file}")
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                config.TRANSFORMER_MODEL_NAME, num_labels=2
            ).to(device)
            train_loader = _make_loader(
                tokenizer,
                keywords[tr].tolist(),
                texts[tr].tolist(),
                labels[tr],
                batch_size=config.TRANSFORMER_BATCH_SIZE,
                shuffle=True,
            )
            _fine_tune(model, train_loader, device, epochs)

            val_loader = _make_loader(tokenizer, keywords[va].tolist(), texts[va].tolist())
            oof[va] = _predict_disaster_proba(model, val_loader, device)
            test_proba = _predict_disaster_proba(model, test_loader, device)
            np.savez(fold_file, val_proba=oof[va], test_proba=test_proba)

            del model
            if device.type == "mps":
                torch.mps.empty_cache()

        fold_f1 = f1_score(labels[va], (oof[va] >= 0.5).astype(int))
        log(f"Fold {fold}/{n_folds}: validation F1 (Disaster class) = {fold_f1:.4f}")
        test_probas.append(test_proba)

    return oof, np.mean(test_probas, axis=0)
