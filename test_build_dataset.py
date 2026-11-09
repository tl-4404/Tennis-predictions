import argparse
from datetime import date
import unittest

from build_dataset import aggregate, build, player_record, score_targets


def match(day, key, winner="1", loser="2", score="6-4 6-4", aces="5"):
    row = dict(winner_id=winner, loser_id=loser, winner_name="Winner", loser_name="Loser",
               surface="Hard", best_of="3", score=score)
    for prefix in ("w_", "l_"):
        row.update({prefix + "ace": aces, prefix + "df": "2", prefix + "svpt": "60",
                    prefix + "1stIn": "40", prefix + "1stWon": "30", prefix + "2ndWon": "10",
                    prefix + "SvGms": "10", prefix + "bpFaced": "4", prefix + "bpSaved": "2"})
    return dict(date=date.fromisoformat(day), key=key, row=row, score=score_targets(score, 3))


def args():
    return argparse.Namespace(window=20, history_lag_days=28, output_start_year=2020,
                              seed=42, min_history=0, train_end_year=2021,
                              validation_end_year=2022)


class ParserTests(unittest.TestCase):
    def test_score_targets(self):
        self.assertEqual(score_targets("7-6(5) 3-6 6-2", 3), (16, 14, 2, 1))
        self.assertEqual(score_targets("6-4 4-6 70-68", 3), (80, 78, 2, 1))
        for value in ("6-4 2-1 RET", "W/O", "6-4 4-6 [10-8]", "6-4 6-4 6-4", "6-4", "6-4 4-6 4-6"):
            self.assertIsNone(score_targets(value, 3), value)

    def test_ratios_pool_counts_and_skip_missing_pairs(self):
        first = player_record(match("2020-01-01", "a"), True)
        second = dict(first, svpt=120, ace=20)
        third = dict(first, svpt=None, ace=100)
        pooled = aggregate([first, second, third])
        self.assertAlmostEqual(pooled["ace_rate"], 25 / 180)
        self.assertAlmostEqual(aggregate([first])["return_point_win_rate"], 20 / 60)
        self.assertAlmostEqual(aggregate([first])["hold_rate"], .8)
        self.assertIsNone(aggregate([])["ace_rate"])

    def test_cutoff_and_current_tournament_are_excluded(self):
        matches = [match("2020-01-01", "old"), match("2020-01-05", "boundary"),
                   match("2020-02-02", "current_a"), match("2020-02-02", "current_b")]
        rows = build(matches, args())
        for features, _, _ in rows[-2:]:
            self.assertEqual(features["a_history_matches"], 1)
            self.assertEqual(features["b_history_matches"], 1)
        self.assertEqual(rows[-1][2]["history_cutoff"], "2020-01-05")

    def test_changing_current_or_future_outcomes_does_not_change_inputs(self):
        history = match("2020-01-01", "history")
        current = match("2020-03-01", "current")
        original = build([history, current, match("2020-05-01", "future")], args())
        changed = build([history, match("2020-03-01", "current", "2", "1", "6-0 6-0", "50"),
                         match("2020-05-01", "future", "2", "1", "6-1 6-1")], args())
        # Names aren't model features. Player ordering stays fixed when winner flips.
        self.assertEqual(original[1][0], changed[1][0])
        self.assertNotEqual(original[1][1]["a_wins"], changed[1][1]["a_wins"])
        self.assertEqual(original[1][2]["a_id"], changed[1][2]["a_id"])

    def test_regression_targets_follow_player_orientation(self):
        features, target, meta = build([match("2020-01-01", "one", score="6-1 6-2")], args())[0]
        self.assertEqual(target["total_games"], 15)
        self.assertEqual(target["game_margin_a"], 9 if target["a_wins"] else -9)
        self.assertEqual(target["sets_a"], 2 if target["a_wins"] else 0)
        self.assertFalse(set(target) & set(features))
        self.assertFalse(set(meta) & set(features))


if __name__ == "__main__":
    unittest.main()
