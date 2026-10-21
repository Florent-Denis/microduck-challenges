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

- **Course**: a lane 4.0 m long and 0.8 m wide, walls 0.15 m high, cut in the middle by a pit 0.30 m
  wide and 0.30 m deep. The two ducks start on either side, joined by a string 1.00 m long tied at
  the base of each neck. Start from the `STAND` pose, 0.5 s untimed settle, joints perturbed by
  ±0.01 rad, heading by ±2°.
- **Command**: the policy reads `[-0.30, 0, 0]`, constant for the whole bout: walk backwards at
  0.30 m/s. Head and body commands are zeros. It is never told its heading or its position.
- **A bout**: only the pit decides. The first duck whose trunk drops into the pit loses; a duck that
  falls over on the platform stays in play and can still be dragged in. If neither is in the pit by
  the time limit (**45 s**), or both go in on one step, the marker decides: the middle of the two
  string ends, pulled towards one side, that side's duck wins.
- **The rating**: five bouts against the Arena's **reference duck** (t7-250, a frozen walker), the
  entry west on odd seeds. A bout scores the centimetres of ground the marker ends from the centre
  line towards the entry, plus **100 cm** when only the opponent is in the pit (minus 100 when only
  the entry is). The rating is the mean of the five, **in centimetres, higher wins**. `arena score`
  prints it.
- **The weekly cup**: every entry plays it, bout against bout; the rating ranks the board.

## The policy

61 observations → 14 joint position targets at 50 Hz, exported to ONNX (≤ 64 MiB); `publish`
checks the shape. The observations, as the Arena builds them: gyro (3), projected gravity (3), joint
positions from the default pose (14), joint velocities one control step old (14), last action (14),
then the 13 command numbers: twist (3), head (4), body (6). Joint order: left leg (hip_yaw,
hip_roll, hip_pitch, knee, ankle), neck_pitch, head_pitch, head_yaw, head_roll, right leg (same
order). The Arena simulates the same robot with the same BAM actuator model as training (MuJoCo,
implicitfast integrator, 10/20 contact solver). The task trains the learner (always the west duck) against
`reference.onnx`, the frozen reference duck, in the Arena's bout: both ducks on one string over the
pit, 45 s, only the pit decides. For its first 250 iterations the string is slack (the duck learns
to walk back with no load), then it is taut as in the Arena. A `fell` termination (the Arena's
fall rule) ends a training episode only. `win` pays when the opponent drops into the pit, `out`
costs when the learner does, and `marker_ground` pays the ground the marker wins towards it.

## Files

| File | What it holds | Change it? |
|---|---|---|
| `challenge.toml` `[params]` | The three knobs below | Yes, first. Keep each value within its `min`/`max` (the workshop refuses others) |
| `challenge.toml` `[recipe]` | `num_envs` 4096, `max_iterations` 3000, `seed` 1 | Yes, when agreed with the user |
| `env.py` | The whole environment: rewards, terminations, the opponent's wiring, domain randomization, `RL_CFG` (network, PPO, checkpoints every 250 iterations) | Yes, once the knobs are not enough |
| `rewards.py` | The tug's own terms: the marker's new ground, the pit margin, the Arena's fall rule | Yes |
| `opponent.py`, `reference.onnx` | The frozen opponent: the reference duck's network and the action term that drives it; `ReferenceActor`, the learner starting as that duck | No |
| `check.py` | Plays two policies against each other (B defaults to the reference) | No |
| `tasks.py`, `event`/`task`/`kind` in `challenge.toml` | Registration with mjlab and the Arena | No |

| Knob | Label | Default | Range | Sets in `env.py` |
|---|---|---|---|---|
| `pull_reward` | Win ground | 2.0 | 0.5–5.0 | `marker_ground` weight |
| `back_reward` | Walk backwards | 3.0 | 0.0–4.0 | `track_linear_velocity` weight |
| `upright_reward` | Stay upright | 1.0 | 0.0–4.0 | `upright` weight |

Do not change, or the Arena cannot race the policy or the tests fail:

- the robot (`MICRODUCK_STANDUP_ROBOT_CFG`) and the 61-number observation layout: a command slot the
  task does not use keeps its tiny sampling range, never deleted;
- `experiment_name = "tug_of_war"` in `RL_CFG`: runs must land in `logs/rsl_rl/tug_of_war/`;
- the twist command, fixed at `-0.30` m/s, and the terminations: the pits (`in_pit`,
  `opponent_in_pit`) and `fell`, which ends a *training* episode only (the Arena judges the pit alone;
  without it an untrained duck lies on its side from 1 s and never learns to stand);
- `opponent.py` and `reference.onnx` (the same bytes the Arena rates against; any other opponent
  trains a duck the rating does not measure), and the slack string until iteration 250 (`SLACK_M`,
  `TAUT_FROM_ITERATION`): the duck learns to walk back with no load, then meets the opponent;
- the constants copied from the Arena at the top of `env.py` (course, string, bout length, fall rule): the
  job installs no Arena, and `tests/test_tug_of_war.py` pins each against it. If the Arena's event
  changes, change the Arena first, then here, then the parity tests;
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

- the goal: a first entry, a given rating, a given place on the board;
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
   unique, and says the change (`pull4`, `back2`):

   ```bash
   # local NVIDIA GPU
   uv run train Mjlab-TugOfWar-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 3000 \
       --agent.seed 1 --agent.logger tensorboard --agent.run-name <name> > logs/<name>.log 2>&1 &
   # Hugging Face Jobs: the same, plus
   #   --hf-jobs --no-wandb --namespace <user> --ckpt-repo <user>/mjlab-tugofwar-microduck-<name>
   ```

   To start the learner as the reference duck (its actor and normalizer, action noise std 0.3, the
   critic fresh) instead of from scratch, add
   `--agent.actor.class-name microduck_challenges.tug_of_war.opponent:ReferenceActor`.

   A local run writes `logs/rsl_rl/tug_of_war/<YYYY-MM-DD_HH-MM-SS>_<name>/`: `model_<N>.pt` every
   250 iterations and the last one, TensorBoard events, `provenance.json`. A job uploads the same
   tree to its checkpoint repo. Its log prints `[job] id: <id>`. Killing the local process only
   detaches from the job: stop it with `uv run hf jobs cancel <id> --namespace <user>`.

### Following a run

- Progress: `grep -o "Learning iteration [0-9]*/[0-9]*" logs/<name>.log | tail -1`.
- A job's checkpoints and curves: `uv run hf download <user>/mjlab-tugofwar-microduck-<name> --local-dir logs/rsl_rl`
  (the repo mirrors `logs/rsl_rl/`); its state: `uv run hf jobs inspect <id> --namespace <user>`.
- Check once per checkpoint (every 250 iterations, several minutes apart), not more often.
- Curves (`Episode_Reward/<term>` is the weighted value; penalties must read ≤ 0):

  ```bash
  uv run python -c "
  import sys; from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E
  ea = E(sys.argv[1], size_guidance={'scalars': 0}).Reload()
  for tag in sys.argv[2:]: print(tag, [(e.step, round(e.value, 3)) for e in ea.Scalars(tag)][-8:])
  " logs/rsl_rl/tug_of_war/<run> Train/mean_episode_length Episode_Termination/in_pit Episode_Termination/fell Episode_Reward/marker_ground Episode_Reward/win
  ```

  Healthy: `marker_ground` stays positive and grows, and after iteration 250 (the string goes taut)
  `win` rises as `in_pit` and `fell` fall; `Train/mean_episode_length` stays near 2275 control steps
  (45.5 s, the settle plus the bout) until a pit ends episodes sooner. Expect a dip at iteration 250,
  where the string goes taut. Warning signs: total reward rising while `marker_ground` stays flat;
  `fell` climbing; `nan_state` terminations. A run that has collapsed and not recovered within about
  500 iterations: tell the user and propose stopping it.

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

Pick by rating: the highest. The best checkpoint is often not the last. To see how two policies pull
against each other, play them (bout k on seed k, sides swapped after each, the first policy starting
west; the second defaults to the reference duck):

```bash
uv run --frozen --group workshop python -m microduck_challenges.tug_of_war.check A.onnx [B.onnx] --bouts 10
```

It prints a line per bout (who is in the pit or loses by the marker, and when) and how it was
pulled: when the string went taut, how far A's string end backed by 2 s, where the marker ended in
cm towards A; then the tally.

Scores on this machine are indicative: only the Space rates an entry, and simulations on different
CPU architectures drift apart after a few seconds. GPU training varies a lot from run to run: before
building on a surprising result, good or bad, run the same commit with another `--agent.seed`.

### Reporting

After each run, tell the user in a few lines: the run and its commit (what changed, why), the best
checkpoint with its rating, the other checkpoints' ratings, the curves that matter (`marker_ground`, `win`, `in_pit`,
`fell`, episode length), and your proposal for the next run and why.

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

All numbers are from the Mac, indicative: the Space (x86) rates differently. The ratings below are the
old bench's, in newtons; the Arena now rates in centimetres against the reference.

- t1 (the recipe as first written: the bench's winch, a fall loses): twelve trained checkpoints sat
  between 5.38 and 6.32 N, a plateau from iteration 250 on. Both ducks of a bout between two of them
  lean and hold; every bout was decided by the marker. The best rating (2250) lost 0-10 to 2750.
- t2, t3 (only the pit ends a bout): a duck that falls at 1 s lies on its side for the episode and is
  dragged in slowly; training never learns to stand. Fix: `fell` ends a training episode.
- t4, t5 (with it): each dove just under the fall line, propped, and held by friction: standing still
  earned 0.41 of what walking back paid. Fix: `back_reward` 3.0 and a tighter std.
- t6, t7 (locomotion first, no winch for 500 iterations): the walk back is learned by 250 and
  unlearned once the winch pulls. t7-250 walks back hard and pulls the string taut in 2 s: it became
  the reference duck.
- Why the winch went: a duck lying down held the winch's friction as well as any trained one (the
  zero policy rated 5.23 N, t1's best 6.32 N, a ceiling near 7 N), so the bench measured lying, not
  pulling. Training now runs against the reference duck in the Arena's bout, and the rating is
  bouts against it.
- t8, t9 (the recipe as it stands, seeds 1 and 2): see the training ledger when they land.

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
