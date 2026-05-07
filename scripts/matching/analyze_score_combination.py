"""Analyze optimal combination of bi-encoder + claim CE + comment CE scores.

Uses the test splits saved during training to find the best weighting.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sentence_transformers import CrossEncoder
from itertools import product

SCRIPTS_DIR = Path(__file__).resolve().parent

def load_test_split(model_dir):
    """Load the test split saved during training."""
    test_path = model_dir / "test_split.csv.gz"
    if test_path.exists():
        return pd.read_csv(test_path)
    return None

def score_with_model(model_path, pairs, max_length, batch_size=64):
    """Score pairs with a cross-encoder model."""
    model = CrossEncoder(str(model_path), max_length=max_length)
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    return np.array(scores)

def main():
    comment_model_dir = SCRIPTS_DIR / "cross_encoder_models" / "modernbert-comment-cross-encoder"
    claim_model_dir = SCRIPTS_DIR / "cross_encoder_models" / "modernbert-claim-cross-encoder-v2"

    # Load the combined training data to get test split with both comment and claim text
    combined = pd.read_csv(SCRIPTS_DIR / "training_data" / "combined_training_data.csv")
    combined = combined[combined["llm_label"].isin(["yes", "no"])].copy()
    combined["label"] = (combined["llm_label"] == "yes").astype(int)

    # Load comment CE test split to get the test dockets
    comment_test = load_test_split(comment_model_dir)
    if comment_test is not None:
        # Get test docket IDs from the comment CE split
        test_doc_ids = set(comment_test["doc_id"].astype(str))
        test_df = combined[combined["doc_id"].astype(str).isin(test_doc_ids)].copy()
        print(f"Test set from comment CE split: {len(test_df)} rows")
    else:
        # Fallback: use docket-based split
        print("No test split found, using docket-based split")
        dockets = combined["docket_id"].fillna("unknown").unique()
        rng = np.random.RandomState(42)
        rng.shuffle(dockets)
        n_test = max(1, int(len(dockets) * 0.10))
        test_dockets = set(dockets[:n_test])
        test_df = combined[combined["docket_id"].isin(test_dockets)].copy()
        print(f"Test set: {len(test_df)} rows from {n_test} dockets")

    labels = test_df["label"].values
    print(f"Labels: {labels.sum()} positive, {(1-labels).sum()} negative")

    # Score with comment CE
    print("\nScoring with comment CE...")
    comment_pairs = list(zip(
        test_df["response_text"].tolist(),
        test_df["candidate_text"].tolist(),  # candidate_text = comment_text
    ))
    comment_scores = score_with_model(
        comment_model_dir / "final", comment_pairs, max_length=8192
    )

    # Score with claim CE (only for rows that have claim_text)
    has_claims = test_df["claim_text"].notna() & (test_df["claim_text"].str.len() > 0)
    print(f"\nRows with claims: {has_claims.sum()}/{len(test_df)}")

    claim_scores = np.full(len(test_df), 0.5)  # default for rows without claims
    if has_claims.any():
        print("Scoring with claim CE...")
        claim_pairs = list(zip(
            test_df.loc[has_claims, "response_text"].tolist(),
            test_df.loc[has_claims, "claim_text"].tolist(),
        ))
        claim_scores_subset = score_with_model(
            claim_model_dir / "final", claim_pairs, max_length=4096
        )
        claim_scores[has_claims.values] = claim_scores_subset

    # Load thresholds
    with open(comment_model_dir / "final" / "optimal_threshold.json") as f:
        comment_threshold = json.load(f)["threshold"]
    with open(claim_model_dir / "final" / "optimal_threshold.json") as f:
        claim_threshold = json.load(f)["threshold"]

    print(f"\nComment CE threshold: {comment_threshold}")
    print(f"Claim CE threshold: {claim_threshold}")

    # Individual model performance
    print("\n=== Individual Model Performance ===")
    for name, scores, threshold in [
        ("Comment CE", comment_scores, comment_threshold),
        ("Claim CE", claim_scores, claim_threshold),
    ]:
        preds = (scores >= threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        try:
            auc = roc_auc_score(labels, scores)
        except:
            auc = 0.0
        print(f"{name}: F1={f1:.4f}, P={prec:.4f}, R={rec:.4f}, AUC={auc:.4f}")

    # Grid search for optimal combination weights
    print("\n=== Grid Search: Weighted Combination ===")
    best_f1 = 0
    best_config = None

    # Try different weight combinations
    weights_grid = np.arange(0, 1.05, 0.1)

    for w_comment in weights_grid:
        for w_claim in weights_grid:
            if w_comment + w_claim > 1.01:
                continue
            if w_comment + w_claim < 0.01:
                continue

            # Weighted combination
            combined_score = w_comment * comment_scores + w_claim * claim_scores

            # Find optimal threshold for this combination
            for t in np.arange(0.05, 0.95, 0.05):
                preds = (combined_score >= t).astype(int)
                f1 = f1_score(labels, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_config = {
                        "w_comment": round(w_comment, 2),
                        "w_claim": round(w_claim, 2),
                        "threshold": round(t, 2),
                        "f1": round(f1, 4),
                        "precision": round(precision_score(labels, preds, zero_division=0), 4),
                        "recall": round(recall_score(labels, preds, zero_division=0), 4),
                    }

    print(f"Best combination: {json.dumps(best_config, indent=2)}")

    # Also try: comment CE alone with optimized threshold
    print("\n=== Comment CE with optimized threshold ===")
    best_comment_f1 = 0
    best_comment_t = 0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (comment_scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_comment_f1:
            best_comment_f1 = f1
            best_comment_t = t

    preds = (comment_scores >= best_comment_t).astype(int)
    print(f"Best threshold: {best_comment_t:.2f}")
    print(f"F1={best_comment_f1:.4f}, P={precision_score(labels, preds, zero_division=0):.4f}, R={recall_score(labels, preds, zero_division=0):.4f}")

    # Also try: OR logic (match if either CE says yes)
    print("\n=== OR Logic (either CE matches) ===")
    or_preds = ((comment_scores >= comment_threshold) | (claim_scores >= claim_threshold)).astype(int)
    f1 = f1_score(labels, or_preds, zero_division=0)
    prec = precision_score(labels, or_preds, zero_division=0)
    rec = recall_score(labels, or_preds, zero_division=0)
    print(f"F1={f1:.4f}, P={prec:.4f}, R={rec:.4f}")

    # AND logic
    print("\n=== AND Logic (both CEs match) ===")
    and_preds = ((comment_scores >= comment_threshold) & (claim_scores >= claim_threshold)).astype(int)
    f1 = f1_score(labels, and_preds, zero_division=0)
    prec = precision_score(labels, and_preds, zero_division=0)
    rec = recall_score(labels, and_preds, zero_division=0)
    print(f"F1={f1:.4f}, P={prec:.4f}, R={rec:.4f}")

    # Save results
    results = {
        "best_weighted_combination": best_config,
        "comment_ce_alone": {
            "threshold": round(best_comment_t, 2),
            "f1": round(best_comment_f1, 4),
        },
        "comment_threshold": comment_threshold,
        "claim_threshold": claim_threshold,
    }

    out_path = SCRIPTS_DIR / "score_combination_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Manual inspection: sample 10 pairs near the decision boundary
    print("\n=== MANUAL INSPECTION: 10 pairs near decision boundary ===")
    combined_score = best_config["w_comment"] * comment_scores + best_config["w_claim"] * claim_scores
    boundary = best_config["threshold"]
    distances = np.abs(combined_score - boundary)
    near_boundary = np.argsort(distances)[:10]

    for idx in near_boundary:
        row = test_df.iloc[idx]
        print(f"\n--- Score={combined_score[idx]:.3f} (threshold={boundary}) | label={row['llm_label']} ---")
        print(f"Comment CE={comment_scores[idx]:.3f}, Claim CE={claim_scores[idx]:.3f}")
        print(f"Response: {str(row['response_text'])[:200]}")
        print(f"Comment: {str(row['candidate_text'])[:200]}")

if __name__ == "__main__":
    main()
