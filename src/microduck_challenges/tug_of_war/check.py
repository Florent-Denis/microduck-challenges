"""Which of two Tug of War policies pulls the other off the platform?

    uv run --frozen --group workshop python -m microduck_challenges.tug_of_war.check A.onnx [B.onnx] [--bouts 10]

Plays bouts between the two (B defaults to the reference duck) on the Arena's own physics
(arena.tug.run_bout), bout k on seed k, sides swapped after each (A starts on the west platform),
and prints a line per bout (who lost and how it was pulled), then the tally.
"""

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
from arena.manifest import load_event
from arena.policy import PolicyError, load_policy
from arena.constants import CTRL_DT
from arena.course import string_offset
from arena.tug import BoutOutcome, marker_x, run_bout, string_ends

from .opponent import REFERENCE

EVENT = "tug-of-war"
SIDES = ("west", "east")
OFFSET = string_offset("legs")
TAUT_M = 0.995  # the string's ends this far apart: it is taut
BACKED_S = 2.0  # where A's retreat is read


def play(manifest, a, b, bouts: int) -> list[tuple[str, BoutOutcome]]:
    """(A's side, outcome) per bout: bout k on seed k + 1, A west when k is even."""
    results = []
    for k in range(bouts):
        side = SIDES[k % 2]
        west, east = (a, b) if side == "west" else (b, a)
        results.append((side, run_bout(manifest, west, east, k + 1)))
    return results


def pull(outcome: BoutOutcome, side: str) -> str:
    """How A pulled: when the string first went taut, how far A's string end moved away from the pit
    from the settle's last frame to BACKED_S, and where the marker ended, in cm towards A."""
    towards_a = 1.0 if side == "west" else -1.0
    ends = string_ends(outcome.trace, OFFSET)
    taut = np.flatnonzero(np.linalg.norm(ends[0] - ends[1], axis=1) >= TAUT_M)
    taut_at = f"taut at {(taut[0] + 1) * CTRL_DT:.2f} s" if taut.size else "never taut"
    start = string_ends(outcome.settle[:, -1:], OFFSET)[SIDES.index(side), 0, 0]
    at = min(round(BACKED_S / CTRL_DT), len(outcome.trace[0])) - 1
    backed = -towards_a * (ends[SIDES.index(side), at, 0] - start) * 100
    marker = -towards_a * marker_x(outcome.trace[:, -1:], OFFSET)[0] * 100
    return f"{taut_at} · A backed {round(backed, 1) + 0.0:.1f} cm by {BACKED_S:.0f} s · marker {round(marker, 1) + 0.0:.1f} cm to A"


def summary(results) -> list[str]:
    lines = []
    for k, (side, outcome) in enumerate(results, 1):
        if outcome.loser is None:
            verdict = "drawn"
        else:
            loser = "A" if SIDES[outcome.loser] == side else "B"
            verdict = f"{loser} loses by {outcome.how}"
        lines.append(
            f"bout {k} · seed {outcome.seed} · A {side} · {verdict} at {outcome.time_s:.2f} s"
            f" · {pull(outcome, side)}"
        )
    outcomes = [outcome for _, outcome in results]
    decided = [(side, o) for side, o in results if o.loser is not None]
    a_loses = sum(SIDES[o.loser] == side for side, o in decided)
    pits = [sum(o.how == "pit" and (SIDES[o.loser] == side) == a for side, o in results) for a in (True, False)]
    lines.append(
        f"A {len(decided) - a_loses} · B {a_loses} · draws {len(results) - len(decided)}"
        f" · A in the pit {pits[0]} · B in the pit {pits[1]} · by marker {sum(o.how == 'marker' for o in outcomes)}"
        f" · median decision {statistics.median(o.time_s for o in outcomes):.1f} s"
    )
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Play Tug of War bouts between two policies.")
    parser.add_argument("a")
    parser.add_argument("b", nargs="?", default=str(REFERENCE))
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
