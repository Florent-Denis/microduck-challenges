# AGENTS.md: Tug of War

You are helping a user train a Microduck policy for the Arena's **Tug of War** (event `tug-of-war`,
task `Mjlab-TugOfWar-MicroDuck`, this folder). The user may be new to reinforcement learning. Work so
that training stays:

- **simple**: the user decides the goal, the budget and what gets published; you do the mechanics;
- **transparent**: say what you change and why before each run, one commit per run, and report the
  Arena evaluator's numbers, never an impression from the curves;
- **efficient**: one change per run, a smoke test before any long run, every checkpoint judged,
  a failed run stopped instead of paid for.

Never launch a Hugging Face job, publish, enter the Arena or push without the user's explicit yes.

## The challenge

Two ducks stand neck to neck over a pit, a string between them, each pulling by walking
backwards. The rules, from `arena/events/tug-of-war.toml` in the installed Arena:

- **Course**: a wrestling ring 3.00 m from rope to rope, cut across the middle by a pit 0.30 m wide
  and 0.30 m deep. The ring's three ropes (6, 12 and 18 cm high) are the field's edges: a duck that
  backs into them bounces off. The two ducks start on either side of the pit. Start from the `STAND`
  pose, 0.5 s untimed settle, joints perturbed by ±0.01 rad, heading by ±2°.
- **The string**: 1.00 m, tied at the base of each neck, laid straight at the start. It is a real
  rope, 80 g and 8 mm thick in sixteen pieces: it sags, and a duck can step or lie on it. Between two
  ducks pulling, its ends stand about 0.9 m apart (at most 0.966 m in the bouts traced), its lowest
  point at the canvas or down over the pit.
- **Command**: the policy reads `[-0.30, 0, 0]`, constant for the whole bout: walk backwards at
  0.30 m/s. Head and body commands are zeros. It is never told its heading or its position.
- **A bout**: only the pit decides. The first duck whose trunk drops into the pit loses; a duck that
  falls, sits or lies down stays in play and can still be dragged in. If neither is in the pit by
  the time limit (**45 s**), or both go in on one step, the marker decides: the string's middle
  point, pulled towards one side, that side's duck wins.
- **The rating**: five bouts against the Arena's **reference duck** (`reference.onnx` here, t15-250,
  trained on its feet on this ring and string), the entry west on odd seeds. A bout scores the
  centimetres of ground the marker ends from the centre line towards the entry, plus **100 cm** when
  only the opponent is in the pit (minus 100 when only the entry is). The rating is the mean of the
  five, **in centimetres, higher wins**. `arena score` prints it. The ducks trained before it rate
  -132 to -139 cm against it ("What is known"): a first entry that rates negative is normal.
- **The weekly cup**: every entry plays it, bout against bout; the rating ranks the board.

## The policy

61 observations → 14 joint position targets at 50 Hz, exported to ONNX (≤ 64 MiB); `publish`
checks the shape. The observations, as the Arena builds them: gyro (3), projected gravity (3), joint
positions from the default pose (14), joint velocities one control step old (14), last action (14),
then the 13 command numbers: twist (3), head (4), body (6). Joint order: left leg (hip_yaw,
hip_roll, hip_pitch, knee, ankle), neck_pitch, head_pitch, head_yaw, head_roll, right leg (same
order). The Arena simulates the same robot with the same BAM actuator model as training (MuJoCo,
implicitfast integrator, 10/20 contact solver).

## The recipe

The task trains the learner (always the west duck) against `reference.onnx`, the frozen reference
duck, in the Arena's bout: the ring, the string, the pit, 45 s, only the pit decides. The opponent
runs as the Arena runs it, with no randomisation: its servos on the Arena's nominal battery (7.35 V,
sag gain 0.1), where the learner's are drawn per env.

- **The learner starts as the reference duck** (`ReferenceActor`, `RL_CFG`'s actor: the
  reference's layers and observation normalizer, action noise std 0.3, the critic fresh). Trained
  from scratch it plateaued or lay flat (t8r, t9r); started as the reference it dragged it into the
  pit by 500 (t10, t12). To train from scratch anyway, add `--agent.actor.class-name MLPModel`.
- **The string is the Arena's from the first iteration**: there is no slack phase. Every reset lays
  it straight between the two ducks' string ends.
- **1000 iterations, a checkpoint every 250**: the warm-started duck peaks early (at 250 or 500 in
  every run so far), then decays. Keep the best checkpoint, not the last: judge every one.
- **Stay upright** (`stay_upright`, off by default). At 0 a duck may crouch, sit, lie or crawl, as
  the Arena allows, and the Arena's fall rule (`fell`: tilted past 60° for 0.3 s) ends a training
  episode only. Above 0, `down` replaces `fell`: the learner is down when its trunk is under 7 cm
  (`UPRIGHT_Z_M`) or tilted past 45° (`UPRIGHT_TILT_DEG`), and down for 0.3 s straight ends the
  training episode and costs that many pits (-50 each). Standing, the trunk is at 12 cm; a puller
  leaning back on its feet keeps it over 7 cm. Turn it on for a duck that must stay on its feet:
  the owner's ducks trained with it at 1.0, and Win ground at 4.0 (at 2.0 they stood and never
  pulled).
- `win` pays (+50) when the opponent drops into the pit, `out` costs (-50) when the learner does (and
  when `fell` ends the episode), `marker_ground` pays the new ground the marker wins towards the
  learner, `pit_margin_reward` the ground kept between the learner's string end and the pit.

## Files

| File | What it holds | Change it? |
|---|---|---|
| `challenge.toml` `[params]` | The four knobs below | Yes, first. Keep each value within its `min`/`max` (the workshop refuses others) |
| `challenge.toml` `[recipe]` | `num_envs` 4096, `max_iterations` 1000, `seed` 1 | Yes, when agreed with the user |
| `env.py` | The whole environment: the ring and the string (copies of the Arena's), rewards, terminations, the opponent's wiring, domain randomization, `RL_CFG` (the actor, starting as the reference duck; PPO; checkpoints every 250 iterations) | Yes, once the knobs are not enough; not the copied constants (below) |
| `rewards.py` | The tug's own terms: the marker's new ground, the pit margin, the Arena's fall rule (`ArenaFall`) and the upright rule (`Down`, with stay upright on) | Yes |
| `opponent.py`, `reference.onnx` | The frozen opponent: the reference duck's network and the action term that drives it; `ReferenceActor`, the learner's actor, starting as that duck | No |
| `check.py` | Plays two policies against each other (B defaults to the reference), says how each bout was pulled and how A stood | No |
| `tasks.py`, `event`/`task`/`kind` in `challenge.toml` | Registration with mjlab and the Arena | No |

| Knob | Label | Default | Range | The owner's ducks | Sets in `env.py` |
|---|---|---|---|---|---|
| `pull_reward` | Win ground | 2.0 | 0.5–5.0 | 4.0 | `marker_ground` weight |
| `back_reward` | Walk backwards | 3.0 | 0.0–4.0 | 3.0 | `track_linear_velocity` weight |
| `upright_reward` | Keep the trunk level | 1.0 | 0.0–4.0 | 1.0 | `upright` weight |
| `stay_upright` | Stay upright | 0.0 | 0.0–2.0 | 1.0 | the cost of going down, in pits (`down` weight -50 × value); above 0, `down` replaces `fell` |

Do not change, or the Arena cannot race the policy or the tests fail:

- the robot (`MICRODUCK_STANDUP_ROBOT_CFG`) and the 61-number observation layout: a command slot the
  task does not use keeps its tiny sampling range, never deleted;
- `experiment_name = "tug_of_war"` in `RL_CFG`: runs must land in `logs/rsl_rl/tug_of_war/`;
- the twist command, fixed at `-0.30` m/s, and the terminations: the pits (`in_pit`,
  `opponent_in_pit`) and a training-only end for a duck that goes down, `fell` or, with stay upright
  on, `down` (the Arena judges the pit alone; without one an untrained duck lies on its side from
  1 s and never learns to stand);
- `opponent.py`, `reference.onnx` and the opponent's servos (`OPPONENT_CFG`, the Arena's battery):
  the same bytes and servos the Arena rates against; any other opponent trains a duck the rating
  does not measure;
- the ring and the string copied from the Arena at the top of `env.py` (`RING_M`, `RING_ROPES_M`,
  the rope and post radii, `PIT_M`, `PIT_DEPTH_M`, `STRING_M` and every `STRING_*` number, the bout's
  length, the fall rule) and `STRING`/`MIDDLE` in `rewards.py`: the job installs no Arena, and the
  parity tests in `tests/test_tug_of_war.py` pin each against it (`test_the_constants_are_the_arenas`,
  `test_the_course_tiles_the_arenas`, `test_the_rings_ropes_and_posts_are_the_arenas_in_pieces`,
  `test_the_string_is_the_arenas`, `test_the_string_ends_are_the_arenas`,
  `test_the_string_end_is_where_the_arena_ties_it`, `test_the_marker_is_where_the_arena_reads_it`).
  The string's numbers are the Arena's measured ones: each step down from them was unstable, and
  without its armature every world turned NaN within a second. If the Arena's event changes, change
  the Arena first, then here, then the parity tests;
- the course's 2 m tiles (`TILE_M`) and the solver's buffers (`nconmax` 200, `njmax` 1500): see
  "What is known";
- every domain randomization range must contain the nominal value the Arena simulates;
- `pyproject.toml`, `uv.lock` and the other challenges' folders.

The robot model and the reward functions (`microduck_mdp`) live in the `mjlab-microduck` package
and are not edited here. Its [AGENTS.md](https://github.com/pollen-robotics/microduck_rl/blob/develop/AGENTS.md)
holds the reward-design rules learned on this robot: read its "Reward design" section before
writing a reward term.

## Running a training

### Once per session

```bash
uv sync --group workshop
nvidia-smi              # an NVIDIA GPU here trains locally; otherwise Hugging Face Jobs
uv run hf auth whoami   # the account a job runs and bills under, and publish publishes to
```

Agree with the user before the first run:

- the goal: a first entry, a given rating, a given place on the board, a duck that stays on its
  feet or not;
- the budget: how many runs; local GPU, or Hugging Face Jobs (default hardware `l4x1`, $0.80 per
  hour, billed to their account, which needs a payment method);
- what you may launch without asking each time (by default: local runs you announce and start;
  each Hugging Face job asked for).

### Each run

1. **State the hypothesis** in one sentence: what you change, why, and what you expect to see.
2. **Change one thing**, a knob first. Combine changes only on purpose, and say so.
3. **Test**: `uv run --group workshop pytest`.
4. **Smoke test** (under a minute, also on a machine without a GPU):

   ```bash
   uv run train Mjlab-TugOfWar-MicroDuck --gpu-ids None --env.scene.num-envs 2 \
       --agent.max-iterations 2 --agent.logger tensorboard --agent.run-name smoke
   rm -rf logs/rsl_rl/tug_of_war/*_smoke
   ```

5. **Commit** the change, its message the hypothesis:
   `git add src/microduck_challenges/tug_of_war && git commit -m "<hypothesis>"`. `publish` writes
   that commit into the policy's recipe, and refuses a run started from uncommitted changes.
6. **Launch** in the background, the output in a log under `logs/` (git-ignored). `<name>` is short,
   unique, and says the change (`pull4`, `upright1`):

   ```bash
   # local NVIDIA GPU
   uv run train Mjlab-TugOfWar-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 1000 \
       --agent.seed 1 --agent.logger tensorboard --agent.run-name <name> > logs/<name>.log 2>&1 &
   # Hugging Face Jobs: the same, plus
   #   --hf-jobs --no-wandb --namespace <user> --ckpt-repo <user>/mjlab-tugofwar-microduck-<name>
   ```

   A local run writes `logs/rsl_rl/tug_of_war/<YYYY-MM-DD_HH-MM-SS>_<name>/`: `model_<N>.pt` every
   250 iterations and the last one, TensorBoard events, `provenance.json`. A job uploads the same
   tree to its checkpoint repo. Its log prints `[job] id: <id>`. Killing the local process only
   detaches from the job: stop it with `uv run hf jobs cancel <id> --namespace <user>`.

### Following a run

- Progress: `grep -o "Learning iteration [0-9]*/[0-9]*" logs/<name>.log | tail -1`.
- A job's checkpoints and curves: `uv run hf download <user>/mjlab-tugofwar-microduck-<name> --local-dir logs/rsl_rl`
  (the repo mirrors `logs/rsl_rl/`); its state: `uv run hf jobs inspect <id> --namespace <user>`.
- Check once per checkpoint (every 250 iterations, several minutes apart), not more often.
- Curves (`Episode_Reward/<term>` is the weighted value; penalties must read ≤ 0; with stay upright
  on, ask for `Episode_Termination/down` in place of `Episode_Termination/fell`):

  ```bash
  uv run python -c "
  import sys; from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E
  ea = E(sys.argv[1], size_guidance={'scalars': 0}).Reload()
  for tag in sys.argv[2:]: print(tag, [(e.step, round(e.value, 3)) for e in ea.Scalars(tag)][-8:])
  " logs/rsl_rl/tug_of_war/<run> Train/mean_episode_length Episode_Termination/in_pit Episode_Termination/opponent_in_pit Episode_Termination/fell Episode_Reward/marker_ground Episode_Reward/win
  ```

  Healthy, as measured with stay upright on and Win ground 4.0 (t15, t17): `down` falls from 2 to
  10 at iteration 50 to about 0.2 by 250; `marker_ground`, `win` and
  `opponent_in_pit` rise and the opponent's pit cuts episodes short (t15 at 250: `marker_ground`
  0.27, `win` 0.015, `opponent_in_pit` 2.0, episodes 1734 control steps); the learner's `in_pit`
  stays at 0. Then it decays: t15's `marker_ground` fell to 0.18 and `win` to 0.005 by 750, its
  episodes back near 2200 control steps, the reference dragged in later each time and no longer at
  999; t17's `marker_ground` fell from 0.11 at 250 to 0.06 at 500. That decay is the known shape,
  not a broken run: judge the checkpoints at 250, 500, 750 and 1000 and keep the best, not the last.
  Warning signs: `marker_ground` flat near 0.035 with the learner's `in_pit` near 1 (t16 never
  found the pull and was dragged in); `marker_ground` flat near 0.01 while `upright` rises (t13,
  t14 stood and never pulled); `fell`, `down` or the learner's `in_pit` climbing; `nan_state` in more
  than a few worlds per step (the term counts worlds). A run that has collapsed and not recovered
  within about 500 iterations: tell the user and propose stopping it.

### Judging every checkpoint

The curves are not the score. Judge each checkpoint with the Arena's own evaluator:

- If the user has the workshop open (`uv run --group workshop workshop tug-of-war`), it finds every
  run under `logs/rsl_rl/`, yours included, and evaluates each checkpoint as it arrives. Read its
  results instead of evaluating again: `logs/rsl_rl/tug_of_war/<run>/workshop/model_<N>.json`.
- Otherwise export each checkpoint as the real publish does (`--dry-run` publishes nothing, the
  observation normalizer baked in, into `publish-eval/`) and score it:

  ```bash
  run=logs/rsl_rl/tug_of_war/<run>
  for n in $(ls $run | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n); do
    uv run publish --run $run --checkpoint $n --repo local/microduck-eval --dry-run --allow-dirty >/dev/null 2>&1 &&
    echo "checkpoint $n:" && uv run --group workshop arena score publish-eval/policy.onnx --event tug-of-war
  done
  ```

  `arena score` prints one line per bout (five), then `rating: X.X cm`.

To see how a checkpoint pulls, and how two policies pull against each other, play them (bout k on
seed k, sides swapped after each, the first policy starting west; the second defaults to the
reference duck):

```bash
uv run --frozen --group workshop python -m microduck_challenges.tug_of_war.check A.onnx [B.onnx] --bouts 10
```

It prints a line per bout, then the tally:

```
bout 1 · seed 1 · A west · B loses by pit at 14.20 s · never taut · A backed 7.2 cm by 2 s · marker 34.8 cm to A · on its feet 55 % · longest down 0.20 s
```

- **taut**: when the string's ends first stood 0.98 m apart (`TAUT_M`). Information only: "never
  taut" is normal, the 80 g string sags short of it even while a duck drags the other in.
- **A backed**: how far A's end of the string moved away from the pit in the first 2 s, signed.
- **marker**: where the string's middle ended, in cm towards A.
- **on its feet**, **longest down**: the share of A's frames upright by stay upright's rule (trunk
  at 7 cm or more, tilt 45° or less), and its longest straight run down; the frames with A's trunk
  within 8 cm of the pit's lip (`LIP_M`) are left out. Stay upright ends an episode on the run, not
  the share: a duck that dips under 7 cm every second but never for 0.3 s is on its feet.

The gates the owner's ducks were judged by (10 `check.py` bouts against the reference, the rating
from `arena score`):

- **U** (only for a duck meant to stay on its feet): in at least 9 of the 10 bouts `longest down`
  is under 0.30 s; for the checkpoint you pick, also over the round robin against the other top
  checkpoints: a cup is duck against duck (t18-250 and t18-500 passed U against the reference but
  sat 30 to 43 s against other ducks).
- **R1**: `nan_state` under 0.01 in the curves, and in at least 8 of the 10 bouts A backed +2.0 cm
  or more by 2 s.
- **R2**: a rating of +20.0 cm or more against the reference.
- **R3** (a target): the reference in the pit in at least 3 of the 10 bouts.

Pick the highest rating among the checkpoints that pass. Between ratings within 5 cm, a round robin
decides: `check.py A B --bouts 10` for each pair, more bouts won, then more pits.

Scores on this machine are indicative: only the Space rates an entry, and simulations on different
CPU architectures drift apart after a few seconds. GPU training varies a lot from run to run: before
building on a surprising result, good or bad, run the same commit with another `--agent.seed`.

### Reporting

After each run, tell the user in a few lines: the run and its commit (what changed, why), the best
checkpoint with its rating and its gates, the other checkpoints' ratings, the curves that matter
(`marker_ground`, `win`, `in_pit`, `fell` or `down`, episode length), and your proposal for the next
run and why.

### Publishing, on the user's yes

Publishing makes the policy public on their Hugging Face account and enters it on the Arena, which
rates it and prints its rating and the page of its result:

```bash
uv run publish --run logs/rsl_rl/tug_of_war/<run> --checkpoint <N> --repo <user>/microduck-<name> --no-private --enter
```

`--no-private` because the Arena reads public repos only; add `--livery <colour>` for its colour.
For the policy's code link to open, push the commit to the user's fork first (`origin` must be their
fork, never `pollen-robotics`; the workshop's home sets that up).

## What is known about this challenge

All numbers are from the Mac, indicative: the Space (x86) rates differently.

### On the ring and the string (2026-10-22)

- **The string runs**: a probe of 64 worlds on the Mac's CPU (stay upright on, t10-500 as the
  learner, 300 steps) had no NaN on either duck, at most 39 contacts a world (`nconmax` 200) and 190
  constraint rows (`njmax` 1500); `nan_state` read 0 at every judged checkpoint from t13 to t18 on
  the L4.
- **It sags**: in every bout judged for t13, t14, t15, t17 and t18 against the reference (160), the
  ends never stood 0.98 m apart, t15-250's ten pit wins included. Only t16, which never pulled and
  lost all its bouts, had it taut (28 of 30).
- **The reference search** (stay upright 1.0, warm-started from t7-250, the old reference, and
  against it; ratings against t7-250):
  - t13, t14 (Win ground 2.0, seeds 1 and 2): on their feet 99 to 100 % of every bout (trunk 7 to
    10 cm, tilt 24 to 38°) while t7-250 lay prone from 6 to 8 s, but they never dragged it: t13 -0.4,
    -2.6, -4.4 cm and t14 -3.5, -2.9, -4.4 cm at 250, 500, 750, every bout to the marker at 45 s.
    `marker_ground` stayed flat at 0.01 to 0.025 while `upright` rose from about 0.3 to 0.4 to 0.45:
    standing outpaid pulling. Cancelled at about 780.
  - t15, the one lever (Win ground 4.0, seed 1): `marker_ground` at 250 eleven times t13's (0.274
    against 0.025); +138.4 cm at 250 (the reference in the pit 10 of 10, median 18.4 s, on its feet
    81 to 84 %, longest down 0.10 to 0.14 s, backed +9.7 to +14.0 cm by 2 s), +134.4 at 500, +134.1
    at 750, then +16.9 at 999 with no pit: it peaks early, the pit coming later each time (18.4,
    24.3, 35.9 s).
  - t16 (t15's recipe, seed 2) never found the pull: -14.7, -130.8 (dragged into the pit in about
    7 s, down 0.32 to 0.58 s), -17.8 cm at 250, 500, 750. Cancelled. Seeds diverge: judge a second
    seed before building on one.
  - The pick, t15-250: it passed U and R1 with the highest rating, and beat t15-500 and t15-750
    10-0 each (17 pits). It became the reference duck.
- **Postures** (trunk height and tilt over a bout, and how long the trunk touched the floor):
  standing 12.0 cm; t15-250 a low crouch, median 8.3 cm (lowest 5.9), 34°, never touching the floor;
  t13-500 9.3 cm, 26°; t10-500 4.5 cm, 57°, never touching it either (its legs carry it; stay
  upright calls it down); t7-250 lies with its trunk on the floor 73 to 88 % of a bout.
- **The candidate search** (stay upright 1.0, Win ground 4.0, warm-started from t15-250 and against
  it; ratings against t15-250):
  - t17 (seed 1): +132.3 cm at 250 (the reference in the pit 10 of 10, median 14.3 s, on its feet 54
    to 62 %, longest down 0.16 to 0.28 s, trunk median 7.2 cm, tilt 40°, never touching the floor),
    +27.6 at 500, +64.5 at 750.
  - t18 (seed 2): +134.0 at 250 (longest down under 0.30 s in 9 bouts of 10), +133.2 at 500, +63.0
    at 750.
  - Both peaked at 250. Round robin of the three within 5 cm (40 bouts each): t18-250 won 33 (9
    pits), t17-250 23 (23 pits), t18-500 4; but t18-250 and t18-500 sat down for up to 30 and 43 s
    against other ducks, where t17-250 was never down past 0.24 s. t18-750 lost 0-10 to t17-250, all
    by pit.
  - The pick, the owner's candidate: t17-250, the only one that stays on its feet against other
    ducks, the most pits.
- **The ducks trained before t15-250** rate against it: t7-250 -139.1 cm, t10-500 -138.6,
  t9r-1000 -132.1, each dragged into the pit in every bout, within 13 to 29 s. t15-250 against
  itself rates -1.7 cm.
- The recipe as shipped (stay upright 0.0, Win ground 2.0) has not been trained on the ring yet;
  t12, the same knobs before it, rated +140.3 cm at 500 against t7-250.

### Before the ring (2026-10-20 and 21: a 4 m lane with walls, a massless string, t7-250 the reference)

- t2, t3 (only the pit ends a bout): a duck that falls at 1 s lies on its side for the episode and is
  dragged in slowly; training never learns to stand. Fix: the fall ends a training episode (`fell`).
- t4, t5: each dove just under the fall line, propped, and held by friction: standing still earned
  0.41 of what walking back paid. Fix: `back_reward` 3.0 and a tighter std.
- t8r, t9r (from scratch, seeds 1 and 2): t8r crouched and held, +12.6 cm at 1000 and +10.5 at
  1500; t9r lay on its belly from 2 s into each bout and held, -18.3 cm at 1000.
- t10, t11, t12 (starting as the reference duck): each dragged it over the pit's edge in 10 bouts of
  10 (t10 +139.6 cm at 500, t11 +140.5 at 250, t12, the recipe as shipped then, +140.3 at 500); t10
  and t11 then decayed into lying prone by 1000 (-17.6 and +19.7 cm, no pit).
- The course comes in 2 m tiles (`TILE_M`): on the Arena's 40 m boxes mujoco_warp put a lying duck's
  contact 60 to 130 m away and whole worlds went NaN (t8, t9). Keep the tiles if you edit the course.
- `nconmax` 400 ran an L4 out of memory at 4096 envs; 200 holds the string's 31 to 40 contacts a
  world.

## The competition

- The Arena is the Hugging Face Space
  [pollen-robotics/microduck-arena](https://huggingface.co/spaces/pollen-robotics/microduck-arena).
  The board ranks every entered policy by its rating in centimetres, higher first. Read it live:
  `curl -s https://pollen-robotics-microduck-arena.hf.space/api/events/tug-of-war/leaderboard`.
  Use it to tell the user what a place takes, never to promise one.
- Player points: +100 for a first publication in a challenge, +10 per publication (the first 3 per
  UTC day, all challenges together), +25 once per challenge for improving a personal best,
  +250/+150/+100 while holding first/second/third place. Podium points move with the rank; a
  withdrawn run loses its points.
- The challenge's page on the Arena has the same rules in plain words, and the **Train my own robot**
  button that opens the workshop.
