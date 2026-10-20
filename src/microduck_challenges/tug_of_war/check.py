"""Which of two Tug of War policies pulls the other off the platform?

    uv run --frozen --group workshop python -m microduck_challenges.tug_of_war.check A.onnx B.onnx [--bouts 10]

Plays bouts between the two on the Arena's own physics (arena.tug.run_bout), bout k
on seed k, sides swapped after each (A starts on the west platform), and prints a
line per bout, then the tally.
"""

import argparse
import statistics
import sys
from pathlib import Path

from arena.manifest import load_event
from arena.policy import PolicyError, load_policy
from arena.tug import BoutOutcome, run_bout

EVENT = "tug-of-war"
SIDES = ("west", "east")


def play(manifest, a, b, bouts: int) -> list[tuple[str, BoutOutcome]]:
    """(A's side, outcome) per bout: bout k on seed k + 1, A west when k is even."""
    results = []
    for k in range(bouts):
        side = SIDES[k % 2]
        west, east = (a, b) if side == "west" else (b, a)
        results.append((side, run_bout(manifest, west, east, k + 1)))
    return results


def summary(results) -> list[str]:
    lines = []
    for k, (side, outcome) in enumerate(results, 1):
        if outcome.loser is None:
            verdict = "drawn"
        else:
            loser = "A" if SIDES[outcome.loser] == side else "B"
            verdict = f"{loser} loses by {outcome.how}"
        lines.append(f"bout {k} · seed {outcome.seed} · A {side} · {verdict} at {outcome.time_s:.2f} s")
    outcomes = [outcome for _, outcome in results]
    decided = [(side, o) for side, o in results if o.loser is not None]
    a_loses = sum(SIDES[o.loser] == side for side, o in decided)
    hows = [sum(o.how == how for o in outcomes) for how in ("pit", "fall", "marker")]
    lines.append(
        f"A {len(decided) - a_loses} · B {a_loses} · draws {len(results) - len(decided)}"
        f" · by pit {hows[0]} · by fall {hows[1]} · by marker {hows[2]}"
        f" · median decision {statistics.median(o.time_s for o in outcomes):.1f} s"
    )
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Play Tug of War bouts between two policies.")
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--bouts", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        a, b = load_policy(Path(args.a)), load_policy(Path(args.b))
    except PolicyError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("\n".join(summary(play(load_event(EVENT), a, b, args.bouts))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
