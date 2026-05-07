"""Compare entity resolution approaches on GPT-labeled pairs.

Loads the 652 GPT-4o-mini labeled (comment, DIME) pairs and scores each pair
using multiple matching methods. Computes precision/recall/AUC for each.

Methods compared:
  1. splink match_probability (already in the data)
  2. Jaro-Winkler weighted field score (first + last + state + city + zip)
  3. Exact rule-based tiers (name+state, name-only, nickname)
  4. splink + first-name JW filter (post-filter requiring JW > threshold)
  5. Simple first+last JW product (no other fields)

Outputs a comparison table and saves detailed scores.
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
LABELED_FILE = DATA_DIR / "dime_splink_labeled_pairs.jsonl"

# ---------------------------------------------------------------------------
# Nickname map (subset for scoring)
# ---------------------------------------------------------------------------
NICKNAMES = {
    "bob": "robert", "rob": "robert", "bill": "william", "will": "william",
    "jim": "james", "jimmy": "james", "dick": "richard", "rick": "richard",
    "mike": "michael", "tom": "thomas", "joe": "joseph", "dan": "daniel",
    "dave": "david", "steve": "steven", "ed": "edward", "ted": "theodore",
    "tony": "anthony", "al": "albert", "alex": "alexander", "andy": "andrew",
    "ben": "benjamin", "charlie": "charles", "chuck": "charles",
    "chris": "christopher", "don": "donald", "doug": "douglas",
    "frank": "francis", "fred": "frederick", "jerry": "gerald",
    "hank": "henry", "jack": "john", "johnny": "john", "jake": "jacob",
    "jeff": "jeffrey", "ken": "kenneth", "larry": "lawrence",
    "matt": "matthew", "nick": "nicholas", "pete": "peter", "phil": "philip",
    "ray": "raymond", "ron": "ronald", "sam": "samuel", "tim": "timothy",
    "walt": "walter", "liz": "elizabeth", "beth": "elizabeth",
    "betty": "elizabeth", "sue": "susan", "pat": "patricia",
    "peg": "margaret", "maggie": "margaret", "kate": "katherine",
    "kathy": "katherine", "jenny": "jennifer", "jen": "jennifer",
    "sally": "sarah", "sandy": "sandra", "barb": "barbara",
    "deb": "deborah", "debbie": "deborah", "pam": "pamela",
    "connie": "constance", "cindy": "cynthia", "vicky": "victoria",
    "nate": "nathaniel",
}

def _norm(s):
    if not s or not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"[^a-z\s\-]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_dime_first_last(dime_name):
    if not dime_name or not isinstance(dime_name, str):
        return "", ""
    parts = dime_name.split(",", 1)
    if len(parts) == 2:
        last = _norm(parts[0])
        first_middle = parts[1].strip()
        first = _norm(first_middle.split()[0]) if first_middle.split() else ""
        return first, last
    return "", _norm(dime_name)


def _parse_comment_first_last(comment_name):
    if not comment_name or not isinstance(comment_name, str):
        return "", ""
    parts = _norm(comment_name).split()
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return parts[0] if parts else "", ""


def _expand(first):
    return NICKNAMES.get(first, first)


# ---------------------------------------------------------------------------
# Scoring methods
# ---------------------------------------------------------------------------

try:
    import jellyfish
    jw = jellyfish.jaro_winkler_similarity
except ImportError:
    def jw(a, b):
        if a == b:
            return 1.0
        return 0.0


def score_jw_fields(pair):
    """Weighted Jaro-Winkler across all fields."""
    c_first, c_last = _parse_comment_first_last(pair.get("comment_name", ""))
    d_first, d_last = _parse_dime_first_last(pair.get("dime_name", ""))
    c_state = str(pair.get("comment_state", "")).strip().upper()[:2]
    d_state = str(pair.get("dime_state", "")).strip().upper()[:2]

    # Nickname expansion on first names
    c_first_exp = _expand(c_first)
    d_first_exp = _expand(d_first)

    first_jw = max(jw(c_first, d_first), jw(c_first_exp, d_first_exp))
    last_jw = jw(c_last, d_last)
    state_match = 1.0 if (c_state and d_state and c_state == d_state) else 0.0

    # Weighted combination: first=0.35, last=0.40, state=0.25
    score = 0.35 * first_jw + 0.40 * last_jw + 0.25 * state_match
    return score, first_jw, last_jw, state_match


def score_first_last_product(pair):
    """Simple product of first-name and last-name JW. Strict."""
    c_first, c_last = _parse_comment_first_last(pair.get("comment_name", ""))
    d_first, d_last = _parse_dime_first_last(pair.get("dime_name", ""))
    c_first_exp = _expand(c_first)
    d_first_exp = _expand(d_first)
    first_jw = max(jw(c_first, d_first), jw(c_first_exp, d_first_exp))
    last_jw = jw(c_last, d_last)
    return first_jw * last_jw


def score_rule_based(pair):
    """Rule-based tier: 1=best, 5=worst, 0=no match."""
    c_first, c_last = _parse_comment_first_last(pair.get("comment_name", ""))
    d_first, d_last = _parse_dime_first_last(pair.get("dime_name", ""))
    c_state = str(pair.get("comment_state", "")).strip().upper()[:2]
    d_state = str(pair.get("dime_state", "")).strip().upper()[:2]
    c_first_exp = _expand(c_first)
    d_first_exp = _expand(d_first)

    first_match = (c_first == d_first) or (c_first_exp == d_first_exp)
    last_match = c_last == d_last
    state_match = c_state and d_state and c_state == d_state

    if first_match and last_match and state_match:
        return 5, "exact_name_state"
    if first_match and last_match:
        return 4, "exact_name_only"
    if (c_first_exp == d_first_exp) and last_match and state_match:
        return 3, "nickname_state"
    if (c_first_exp == d_first_exp) and last_match:
        return 2, "nickname_only"
    if last_match:
        return 1, "last_name_only"
    return 0, "no_match"


def score_splink_filtered(pair, jw_threshold=0.7):
    """splink match_prob with a first-name JW floor."""
    c_first, c_last = _parse_comment_first_last(pair.get("comment_name", ""))
    d_first, d_last = _parse_dime_first_last(pair.get("dime_name", ""))
    c_first_exp = _expand(c_first)
    d_first_exp = _expand(d_first)
    first_jw = max(jw(c_first, d_first), jw(c_first_exp, d_first_exp))
    if first_jw < jw_threshold:
        return 0.0
    return pair.get("match_prob", 0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(pairs, score_fn, threshold, name):
    """Compute precision, recall, F1 at a given threshold."""
    tp = fp = fn = tn = 0
    for p in pairs:
        score = score_fn(p)
        if isinstance(score, tuple):
            score = score[0]
        predicted_match = score >= threshold
        actual_match = p["label"] == "yes"

        if predicted_match and actual_match:
            tp += 1
        elif predicted_match and not actual_match:
            fp += 1
        elif not predicted_match and actual_match:
            fn += 1
        else:
            tn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"  {name:35s} threshold={threshold:.2f}  P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}  (TP={tp} FP={fp} FN={fn} TN={tn})")
    return {"method": name, "threshold": threshold, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    # Load labeled pairs
    pairs = [json.loads(l) for l in LABELED_FILE.open()]
    print(f"Loaded {len(pairs)} labeled pairs")

    # Clean labels
    for p in pairs:
        raw = str(p.get("gpt_label", "")).strip().lower()
        if raw.startswith("yes"):
            p["label"] = "yes"
        elif raw.startswith("maybe"):
            p["label"] = "maybe"
        else:
            p["label"] = "no"

    label_dist = Counter(p["label"] for p in pairs)
    print(f"Labels: {dict(label_dist)}")

    # For evaluation, treat "yes" as positive, "no" and "maybe" as negative
    # (conservative — only count definite matches)
    print("\n=== STRICT evaluation (yes=positive, no+maybe=negative) ===\n")

    results = []

    # Method 1: splink raw
    results.append(evaluate(pairs, lambda p: p.get("match_prob", 0), 0.95, "splink (raw, t=0.95)"))
    results.append(evaluate(pairs, lambda p: p.get("match_prob", 0), 0.85, "splink (raw, t=0.85)"))

    # Method 2: splink + first-name JW filter
    results.append(evaluate(pairs, lambda p: score_splink_filtered(p, 0.7), 0.95, "splink + first_JW>0.7 (t=0.95)"))
    results.append(evaluate(pairs, lambda p: score_splink_filtered(p, 0.7), 0.85, "splink + first_JW>0.7 (t=0.85)"))
    results.append(evaluate(pairs, lambda p: score_splink_filtered(p, 0.8), 0.85, "splink + first_JW>0.8 (t=0.85)"))

    # Method 3: JW weighted fields
    results.append(evaluate(pairs, lambda p: score_jw_fields(p)[0], 0.85, "JW fields weighted (t=0.85)"))
    results.append(evaluate(pairs, lambda p: score_jw_fields(p)[0], 0.80, "JW fields weighted (t=0.80)"))
    results.append(evaluate(pairs, lambda p: score_jw_fields(p)[0], 0.70, "JW fields weighted (t=0.70)"))

    # Method 4: first*last JW product
    results.append(evaluate(pairs, score_first_last_product, 0.90, "first*last JW product (t=0.90)"))
    results.append(evaluate(pairs, score_first_last_product, 0.80, "first*last JW product (t=0.80)"))

    # Method 5: rule-based
    results.append(evaluate(pairs, lambda p: score_rule_based(p)[0], 4, "rule-based (exact name)"))
    results.append(evaluate(pairs, lambda p: score_rule_based(p)[0], 3, "rule-based (+ nickname)"))
    results.append(evaluate(pairs, lambda p: score_rule_based(p)[0], 2, "rule-based (+ nick no state)"))

    # Also evaluate with "yes+maybe" as positive (lenient)
    print("\n=== LENIENT evaluation (yes+maybe=positive, no=negative) ===\n")
    for p in pairs:
        p["label_lenient"] = "yes" if p["label"] in ("yes", "maybe") else "no"

    label_dist2 = Counter(p["label_lenient"] for p in pairs)
    print(f"Labels: {dict(label_dist2)}\n")

    # Best methods from strict
    for p in pairs:
        p["label"], p["label_lenient"] = p["label_lenient"], p["label"]
    evaluate(pairs, lambda p: score_splink_filtered(p, 0.7), 0.95, "splink + first_JW>0.7 (t=0.95)")
    evaluate(pairs, lambda p: score_splink_filtered(p, 0.8), 0.85, "splink + first_JW>0.8 (t=0.85)")
    evaluate(pairs, score_first_last_product, 0.90, "first*last JW product (t=0.90)")
    evaluate(pairs, lambda p: score_rule_based(p)[0], 4, "rule-based (exact name)")

    # Detailed field-level analysis
    print("\n=== FIELD-LEVEL ANALYSIS (all pairs) ===\n")
    for p in pairs:
        _, first_jw, last_jw, state_m = score_jw_fields(p)
        p["first_jw"] = first_jw
        p["last_jw"] = last_jw
        p["state_match"] = state_m

    df = pd.DataFrame(pairs)
    for lbl in ["yes", "no", "maybe"]:
        sub = df[df["label_lenient"] == ("yes" if lbl in ("yes", "maybe") else "no")]
        if lbl == "yes":
            sub = df[df.get("gpt_label", df["label"]).apply(lambda x: str(x).startswith("yes"))]
        elif lbl == "no":
            sub = df[df.get("gpt_label", df["label"]).apply(lambda x: str(x).startswith("no"))]
        else:
            sub = df[df.get("gpt_label", df["label"]).apply(lambda x: str(x).startswith("maybe"))]
        if len(sub) == 0:
            continue
        print(f"{lbl:6s} (n={len(sub):3d}): first_jw={sub['first_jw'].mean():.3f}  "
              f"last_jw={sub['last_jw'].mean():.3f}  state_match={sub['state_match'].mean():.3f}  "
              f"splink_prob={sub['match_prob'].mean():.3f}")


if __name__ == "__main__":
    main()
