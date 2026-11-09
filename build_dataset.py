#!/usr/bin/env python3
"""Build pre-match tennis features from Sackmann-format singles CSVs (stdlib only)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import urllib.request

SOURCE = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main"
STAT_NAMES = ("ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced")
SURFACES = ("Hard", "Clay", "Grass", "Carpet", "Unknown")
LEVELS = ("G", "M", "A", "C", "F", "D", "Other")
ROUNDS = ("RR", "R128", "R64", "R32", "R16", "QF", "SF", "F", "Other")
RATES = ("ace_rate", "double_fault_rate", "first_serve_in_rate",
         "first_serve_win_rate", "second_serve_win_rate", "serve_point_win_rate",
         "return_point_win_rate", "break_point_save_rate", "hold_rate", "break_rate")
TARGETS = ("a_wins", "total_games", "game_margin_a", "sets_a", "sets_b", "set_score_class")
METADATA = ("match_id", "tournament_date", "history_cutoff", "split", "a_id", "b_id",
            "a_name", "b_name", "tournament", "surface", "score_winner_first")


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator if 0 <= numerator <= denominator else None


def score_targets(score, best_of):
    """Only completed conventional singles; exclude RET, W/O, defaults, match TBs."""
    tokens = score.strip().split()
    if not tokens or best_of not in (3, 5):
        return None
    pairs = []
    for token in tokens:
        found = re.fullmatch(r"(\d+)-(\d+)(?:\(\d+\))?", token)
        if not found:
            return None
        a, b = map(int, found.groups())
        high, low = max(a, b), min(a, b)
        # Advantage sets (e.g. 70-68) are valid; 10-point match tiebreaks aren't games.
        if not ((high == 6 and low <= 4) or (high == 7 and low in (5, 6))
                or (high >= 8 and high - low == 2)):
            return None
        pairs.append((a, b))
    needed = best_of // 2 + 1
    sw = sum(a > b for a, b in pairs)
    sl = len(pairs) - sw
    if sw != needed or sl >= needed or pairs[-1][0] < pairs[-1][1]:
        return None
    # A match must end as soon as either player wins the required number of sets.
    if sum(a > b for a, b in pairs[:-1]) >= needed:
        return None
    return sum(a for a, _ in pairs), sum(b for _, b in pairs), sw, sl


def download(raw_dir, start, end, tour):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year in range(start, end + 1):
        dest = raw_dir / f"{tour}_matches_{year}.csv"
        if dest.exists():
            print(f"Using cached {dest}", flush=True)
            continue
        url = f"{SOURCE}/{tour}/{dest.name}"
        print(f"Downloading {url}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "tennis-training-parser/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        header = content.decode("utf-8-sig").splitlines()[0].split(",")
        if not {"winner_id", "loser_id", "tourney_date", "score"}.issubset(header):
            raise ValueError(f"Unexpected CSV response: {url}")
        temporary = dest.with_suffix(".csv.tmp")
        temporary.write_bytes(content)
        temporary.replace(dest)


def read_matches(paths):
    matches, seen, counts = [], {}, Counter()
    required = {"tourney_id", "match_num", "tourney_date", "winner_id", "loser_id", "score", "best_of"}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing required columns in {path}: {required - set(reader.fieldnames or [])}")
            for row in reader:
                counts["source_rows"] += 1
                key = f"{row['tourney_id']}:{row['match_num']}"
                if key in seen:
                    if seen[key] != row:
                        raise ValueError(f"Conflicting duplicate match: {key}")
                    counts["duplicate_rows"] += 1
                    continue
                seen[key] = row
                try:
                    date = datetime.strptime(row["tourney_date"], "%Y%m%d").date()
                except ValueError:
                    counts["invalid_date"] += 1
                    continue
                best_of = number(row["best_of"])
                target = score_targets(row["score"], best_of)
                if target is None:
                    counts["incomplete_or_unsupported_score"] += 1
                    continue
                if not row["winner_id"] or not row["loser_id"] or row["winner_id"] == row["loser_id"]:
                    counts["invalid_player_ids"] += 1
                    continue
                matches.append(dict(row=row, date=date, key=key, score=target))
    matches.sort(key=lambda match: (match["date"], match["key"]))
    counts["accepted_completed_matches"] = len(matches)
    return matches, counts


def player_record(match, winner):
    row = match["row"]
    prefix, opp = ("w_", "l_") if winner else ("l_", "w_")
    rec = {name: number(row.get(prefix + name)) for name in STAT_NAMES}
    rec.update({"opp_" + name: number(row.get(opp + name)) for name in STAT_NAMES})
    rec.update(date=match["date"], won=int(winner), surface=row.get("surface") or "Unknown",
               opponent=row["loser_id" if winner else "winner_id"], minutes=number(row.get("minutes")))
    return rec


def aggregate(records):
    """Aggregate ratios as sum(numerator)/sum(denominator) over valid paired observations."""
    sums = defaultdict(lambda: [0.0, 0.0])
    serve_count = 0
    for rec in records:
        def add(name, n, d):
            if ratio(n, d) is not None:
                sums[name][0] += n
                sums[name][1] += d

        def add_fields(name, n, d):
            add(name, rec[n], rec[d])

        for name, n, d in (("ace_rate", "ace", "svpt"), ("double_fault_rate", "df", "svpt"),
                           ("first_serve_in_rate", "1stIn", "svpt"),
                           ("first_serve_win_rate", "1stWon", "1stIn"),
                           ("break_point_save_rate", "bpSaved", "bpFaced")):
            add_fields(name, n, d)
        if ratio(rec["1stIn"], rec["svpt"]) is not None:
            serve_count += 1
            add("second_serve_win_rate", rec["2ndWon"], rec["svpt"] - rec["1stIn"])
        if rec["1stWon"] is not None and rec["2ndWon"] is not None:
            add("serve_point_win_rate", rec["1stWon"] + rec["2ndWon"], rec["svpt"])
        if all(rec[k] is not None for k in ("opp_svpt", "opp_1stWon", "opp_2ndWon")):
            add("return_point_win_rate", rec["opp_svpt"] - rec["opp_1stWon"] - rec["opp_2ndWon"], rec["opp_svpt"])
        if all(rec[k] is not None for k in ("SvGms", "bpFaced", "bpSaved")):
            if rec["bpFaced"] >= rec["bpSaved"] >= 0:
                add("hold_rate", rec["SvGms"] - rec["bpFaced"] + rec["bpSaved"], rec["SvGms"])
        if all(rec[k] is not None for k in ("opp_SvGms", "opp_bpFaced", "opp_bpSaved")):
            if rec["opp_bpFaced"] >= rec["opp_bpSaved"] >= 0:
                add("break_rate", rec["opp_bpFaced"] - rec["opp_bpSaved"], rec["opp_SvGms"])
    result = {name: ratio(*sums[name]) for name in RATES}
    result["serve_stats_matches"] = serve_count
    for name in RATES:
        result[name + "_opportunities"] = sums[name][1]
    return result


class History:
    def __init__(self, window):
        self.recent = deque(maxlen=window)
        self.year = deque()
        self.total = self.wins = 0
        self.surface = defaultdict(lambda: [0, 0])
        self.h2h = defaultdict(lambda: [0, 0])

    def add(self, record):
        self.recent.append(record)
        self.year.append(record)
        self.total += 1
        self.wins += record["won"]
        for group, key in ((self.surface, record["surface"]), (self.h2h, record["opponent"])):
            group[key][0] += record["won"]
            group[key][1] += 1

    def features(self, date, surface, opponent):
        # A 365-day window measured from the target tournament start, subject to embargo.
        while self.year and (date - self.year[0]["date"]).days > 365:
            self.year.popleft()
        result = dict(history_matches=self.total, career_win_rate=ratio(self.wins, self.total),
                      recent_matches=len(self.recent),
                      recent_win_rate=ratio(sum(r["won"] for r in self.recent), len(self.recent)),
                      year_matches=len(self.year),
                      year_win_rate=ratio(sum(r["won"] for r in self.year), len(self.year)))
        for label, pair in (("surface", self.surface[surface]), ("h2h", self.h2h[opponent])):
            result[label + "_matches"] = pair[1]
            result[label + "_win_rate"] = ratio(*pair)
        for label, records in (("recent", self.recent), ("surface_recent", [r for r in self.recent if r["surface"] == surface])):
            result.update({label + "_" + k: v for k, v in aggregate(records).items()})
        return result


def build(matches, args):
    histories = defaultdict(lambda: History(args.window))
    result, next_history = [], 0
    for match in matches:
        cutoff = match["date"] - timedelta(days=args.history_lag_days)
        # Strictly earlier than cutoff. No ordering of matches within tournaments is assumed.
        while next_history < len(matches) and matches[next_history]["date"] < cutoff:
            old = matches[next_history]
            for winner, side in ((True, "winner"), (False, "loser")):
                histories[old["row"][side + "_id"]].add(player_record(old, winner))
            next_history += 1
        if match["date"].year < args.output_start_year:
            continue
        row = match["row"]
        ids = sorted((row["winner_id"], row["loser_id"]))
        # Assignment is independent of outcome; reproducible across runs and input order.
        digest = hashlib.sha256(f"{args.seed}:{match['key']}:{','.join(ids)}".encode()).digest()
        a_id = ids[digest[0] % 2]
        a_wins = int(a_id == row["winner_id"])
        sides = ("winner", "loser") if a_wins else ("loser", "winner")
        features, side_features = {}, []
        for label, side, other in (("a", sides[0], sides[1]), ("b", sides[1], sides[0])):
            player = histories[row[side + "_id"]]
            f = player.features(match["date"], row.get("surface") or "Unknown", row[other + "_id"])
            for source, name in (("rank", "rank"), ("rank_points", "rank_points"), ("age", "age"), ("ht", "height_cm"), ("seed", "seed")):
                f[name] = number(row.get(side + "_" + source))
            f["log_rank"] = math.log(f["rank"]) if f["rank"] is not None and f["rank"] > 0 else None
            for hand in ("R", "L", "U"):
                f["hand_" + hand] = int((row.get(side + "_hand") or "U") == hand)
            entry = row.get(side + "_entry") or "Direct"
            for category in ("Direct", "Q", "WC", "LL", "PR", "Other"):
                f["entry_" + category] = int((entry if entry in ("Direct", "Q", "WC", "LL", "PR") else "Other") == category)
            side_features.append(f)
            features.update({label + "_" + k: v for k, v in f.items()})
        for name in side_features[0]:
            if not name.startswith(("hand_", "entry_")):
                a, b = side_features[0][name], side_features[1][name]
                features["diff_" + name] = a - b if a is not None and b is not None else None
        features["best_of"] = number(row["best_of"])
        features["draw_size"] = number(row.get("draw_size"))
        for field, categories in (("surface", SURFACES), ("tourney_level", LEVELS), ("round", ROUNDS)):
            value = row.get(field)
            if value not in categories:
                value = "Unknown" if field == "surface" else "Other"
            features.update({field + "_" + category: int(value == category) for category in categories})
        if min(side_features[0]["history_matches"], side_features[1]["history_matches"]) < args.min_history:
            continue
        wg, lg, ws, ls = match["score"]
        ag, bg, sa, sb = (wg, lg, ws, ls) if a_wins else (lg, wg, ls, ws)
        targets = dict(a_wins=a_wins, total_games=ag + bg, game_margin_a=ag - bg,
                       sets_a=sa, sets_b=sb, set_score_class=f"{sa}-{sb}")
        year = match["date"].year
        split = "train" if year <= args.train_end_year else "validation" if year <= args.validation_end_year else "test"
        metadata = dict(match_id=match["key"], tournament_date=match["date"].isoformat(),
                        history_cutoff=cutoff.isoformat(), split=split, a_id=row[sides[0] + "_id"],
                        b_id=row[sides[1] + "_id"], a_name=row.get(sides[0] + "_name"),
                        b_name=row.get(sides[1] + "_name"), tournament=row.get("tourney_name"),
                        surface=row.get("surface"), score_winner_first=row["score"])
        result.append((features, targets, metadata))
    return result


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("Data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("Data/processed"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--tour", choices=("atp", "wta"), default="atp")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--output-start-year", type=int, default=2012)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--history-lag-days", type=int, default=28)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--validation-end-year", type=int, default=2022)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.window < 1 or args.history_lag_days < 1 or args.min_history < 0:
        parser.error("window and history-lag-days must be positive; min-history cannot be negative")
    if not args.start_year <= args.output_start_year <= args.end_year:
        parser.error("Require start-year <= output-start-year <= end-year")
    if args.validation_end_year < args.train_end_year:
        parser.error("validation-end-year must be >= train-end-year")
    if args.download:
        download(args.raw_dir, args.start_year, args.end_year, args.tour)
    paths = [args.raw_dir / f"{args.tour}_matches_{year}.csv" for year in range(args.start_year, args.end_year + 1)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        parser.error("Missing source files (use --download): " + ", ".join(missing))
    matches, counts = read_matches(paths)
    rows = build(matches, args)
    if not rows:
        parser.error("No output matches; check years, min-history, and source data")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0][0])
    write_csv(args.output_dir / "X.csv", fields, (f for f, _, _ in rows))
    write_csv(args.output_dir / "y.csv", TARGETS, (y for _, y, _ in rows))
    write_csv(args.output_dir / "metadata.csv", METADATA, (m for _, _, m in rows))
    write_csv(args.output_dir / "training.csv", [*METADATA, *fields, *TARGETS], ({**m, **f, **y} for f, y, m in rows))
    manifest = dict(source=SOURCE, attribution="Jeff Sackmann / Tennis Abstract; archival mirror by Aneeshers",
                    data_license="CC BY-NC-SA 4.0", parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    counts=dict(counts), output_rows=len(rows), feature_count=len(fields), feature_columns=fields,
                    target_columns=list(TARGETS), metadata_columns=list(METADATA),
                    splits=dict(Counter(m["split"] for _, _, m in rows)),
                    class_counts=dict(Counter(str(y["a_wins"]) for _, y, _ in rows)),
                    feature_missing_counts={k: sum(f[k] is None for f, _, _ in rows) for k in fields},
                    source_files=[dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths],
                    notes=["X/y/metadata rows align by position. Only X columns are model inputs.",
                           "History uses tournament starts strictly before target tournament start minus history-lag-days.",
                           "28-day default is a conservative approximation; exact match timestamps are unavailable.",
                           "Incomplete and unsupported scores excluded from both history and targets.",
                           "Empty feature cells represent missing/undefined values; fit imputation/scaling on training only."])
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("output_rows", "feature_count", "splits", "class_counts", "counts")}, indent=2))


if __name__ == "__main__":
    main()
