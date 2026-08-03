# Target documentation outline

Planning artifact for ISSUES_PLAN.md #21. **Written for the v1.0.0 release on `main`** — no page
references a branch or a `git checkout`; `conf.py` carries `release = version = "1.0.0"` and
`github_version = "main"`. Batch F lands with the tag, after the behavioural work in #1-#20.

Planning artifact for ISSUES_PLAN.md #21. This is the skeleton the rewrite fills — page set,
Diataxis mode per page, ordered sections, and where each section's content comes from. Nothing
here is prose to be pasted; it is the structure to write against.

Rules of engagement: every section states what the reader can *do* after it, not what it covers. A
page whose sections can be freely shuffled has failed. Content marked PROTECTED survives verbatim
or near-verbatim — it is verified and hard-won.

## Site layout

```
Getting started : installation (how-to), quickstart (tutorial)
Guides          : rl (how-to), rendering (explanation), examples (how-to catalogue)
Explanation     : dynamics, tracks, sim2real, reproducibility
Reference       : configuration, observations, actions, api/index
```

One page-set change: `rewards_and_rl.rst` (405 lines, three jobs) splits into `rl.rst` and
`sim2real.rst`. Every other page keeps its filename.

| Page | Mode | Now | Target | Job |
|---|---|---:|---:|---|
| `index.rst` | landing | 103 | ~75 | What is this simulator, and which of the four doc sections do I open first? |
| `installation.rst` | how-to | 202 | ~95 | How do I get an install whose physics, map download and rendering I have actually verified? |
| `quickstart.rst` | tutorial | 209 | ~150 | Can I drive one episode end to end, understand why it stopped, and read enough state to steer? |
| `rl.rst` | how-to | — | ~175 | How do I get a flat, finite, single-agent Gymnasium interface with a reward that is not just survival time? |
| `rendering.rst` | explanation | 359 | ~185 | When does a frame actually get produced, and how does that relate to how fast the simulation is running? |
| `examples.rst` | how-to | 178 | ~125 | Which script in examples/ do I run to see the thing I care about, and what does it need before it will start? |
| `dynamics.rst` | explanation | 305 | ~160 | Which of the two usable models should I integrate, and what silently changes when I switch? |
| `tracks.rst` | explanation | 173 | ~165 | What is a track made of, and why is my lateral error already 0.81 m before I have moved? |
| `sim2real.rst` | explanation | — | ~150 | Which hardware imperfections can this simulator reproduce, where does each one enter the control loop, and how often is it redrawn? |
| `reproducibility.rst` | explanation | 195 | ~145 | What must I pin so an episode replays bit-for-bit, and what silently unpins it? |
| `configuration.rst` | reference | 617 | ~400 | What every EnvConfig field is called, what it defaults to, and what it rejects. |
| `observations.rst` | reference | 270 | ~200 | Which fields each observation preset returns, in what shape and dtype, and what bounds the space declares. |
| `actions.rst` | reference | 246 | ~130 | What the two action columns mean under each mode, and what the environment does with a value it never validates. |
| `api/index.rst` | reference | 66 | ~175 | Which public symbol lives on which import path, and what each one is in one line. |

Total: **3,328 lines today → ~2,330 target** — a 30% cut with no facts lost. The "now" column omits
`rewards_and_rl.rst` (405 lines), whose content is split between `rl.rst` and `sim2real.rst`.

The reduction comes almost entirely from the three tells in #21: templated openers and `See also`
closers (~250 lines), the same fact restated on 4-6 pages (~350 lines), and prose that narrates the
code block beside it (~400 lines). Where a page *grows* — `api/index.rst` 66 → ~175 — it is because
one-line summaries replace bare `automodule` walls.

---

## Getting started

### `docs/index.rst` — F1TENTH Gym

**Mode:** landing · **Target:** ~75 lines

**Job:** What is this simulator, and which of the four doc sections do I open first?

**Opening lines (draft):**

> ``f1tenth_gym`` races one or more 1/10th-scale cars on a real racetrack map behind a single
> Gymnasium ``Env`` — single-track or kinematic dynamics, a ray-cast 2D LiDAR, wall and car-to-car
> collisions, and a Frenet-frame view of the track. Every agent is a row in one shared state
> buffer, and a frozen :class:`~f1tenth_gym.envs.env_config.EnvConfig` is the only way in.

1. **The shortest complete program** `[code]`
   *Reader can:* The reader sees the entire API surface — make, reset, step — and one printed
   number that proves a command is a target rather than an instantaneous state, so they can judge
   in ten seconds whether this library fits.
   *Source:* NEW (replaces index.rst:39-58, which prints nothing). Verified output:
   `gym.make(config=EnvConfig(render_enabled=False))`, `reset(seed=42)`, one `step([[0.0, 2.0]])`
   -> `0.019 0.01`. The `.. note::` on the namespaced id / f110_gym divergence is index.rst:15-22
   cut to two lines and attached here, because the gym id in this block is exactly what an
   f110_gym tutorial mistypes.

2. **Where to start**
   *Reader can:* The reader picks one of three doors instead of scanning a sidebar of thirteen
   pages: never run it (quickstart), no working install yet (installation), knows the loop and
   needs a knob (configuration).
   *Source:* NEW. Absorbs the only navigational value in index.rst:24-37 (Highlights) and
   index.rst:22 ("Start with quickstart").

3. **Citing**
   *Reader can:* The reader can cite the environment and find the hardware build guide without
   hunting the README.
   *Source:* index.rst:60-76 verbatim (bibtex block + f1tenth.org/build.html link).

   **Cut:**
   - index.rst:24-37 — the whole `Highlights` list. It is a feature inventory that restates the
   toctree; the four surviving facts (Gymnasium-native 5-tuple, frozen config, multi-agent SoA,
   RL/sim2real) are already carried by the opener, the sample and the toctree captions.
   - index.rst:34 — "~40–55× real time". Unsourced, and the measurement is configuration-dependent
   (measured here: 0.224 ms/step single-agent ST + 1080 beams => 45×, but it scales ~0.2 ms per
   added agent). A perf number belongs next to its configuration, not on a landing page.
   - index.rst:12-13 — "everything needed to develop and evaluate planning and reinforcement-
   learning controllers". Banned self-praise framing (#21 rule 12); replaced by the enumeration of
   what the simulator actually contains.
   - index.rst:15-22 — the 8-line dev-humble/f110_gym `.. note::` shrinks to 2 lines and moves
   under the sample, next to the gym id that provokes it.
   - index.rst:39-58 — the 15-line `Quick example`: 100 steps, a `break`, a `close()`, and no
   output. Replaced by a 9-line sample with its printed values quoted.
   - index.rst:78-103 — the three visible toctrees become four `:hidden:` ones with the agreed
   captions (Getting started / Guides / Explanation / Reference); the `User guide` caption
   disappears and the `rewards_and_rl` entry is replaced by `rl` and `sim2real`.
   - index.rst:17 — "the ``dev-humble`` line". Per the v1.0.0 decision no page names a branch at
   all: the docs describe the tagged release on `main`, so this clause is deleted rather than
   corrected.

### `docs/installation.rst` — How to install f1tenth_gym

**Mode:** how-to · **Target:** ~95 lines

**Job:** How do I get an install whose physics, map download and rendering I have actually verified?

**Opening lines (draft):**

> ``f1tenth_gym`` resolves its ``maps/`` directory four levels up from its own source file and
> downloads tracks into it on first use, so an editable clone is the layout that behaves. Clone
> the repository and sync the environment:

1. **Install from a clone** `[code]`
   *Reader can:* The reader has an importable, editable package and knows that `uv sync` prunes
   anything not in the lock file — before that pruning silently removes a package they installed
   by hand.
   *Source:* installation.rst:52-64 (uv clone + sync), :66-72 (the `uv sync` uninstall warning —
   PROTECTED, keep verbatim), :74-91 (pip variants), :18-19 (Python >=3.9, demoted to one clause),
   :93-96 (branch-off-main note, demoted to a comment on the `git checkout` line). Non-editable
   `pip install git+...` survives as one clause naming its consequence: maps land in `site-
   packages/maps/`.

2. **Verify the install** `[code]`
   *Reader can:* The reader can distinguish a working install from a broken one by two printed
   numbers, and has already triggered the map download, so the network requirement is discovered
   here rather than mid-experiment.
   *Source:* installation.rst:121-149 (the snippet, kept) and :151-153 (the real pasted output —
   PROTECTED). The first-run download (:98-111) folds in as the read-out: a fresh machine prints
   `Downloading Files for: Spielberg` before any observation, and `track/utils.py:28` does four
   `.parent` hops, so the editable layout from the previous section is what makes that path land
   in the repo.

3. **Turn on rendering** `[code]`
   *Reader can:* The reader can open a window on a desktop, or produce frames on a headless box,
   and knows before trying that the GL backend has no offscreen fallback.
   *Source:* installation.rst:165-175 plus :25-26. Gains a runnable two-line `xvfb-run` command
   with `render_mode="rgb_array"` printing the returned frame's `shape` and `dtype` — the
   observable that proves the display path works. Deep configuration links to :doc:`rendering`.

4. **Run the test suite** `[code]`
   *Reader can:* The reader can validate a source change, and knows in advance that the suite
   needs network and rewrites `maps/`, so a failure is read as a prerequisite rather than a bug.
   *Source:* installation.rst:177-193, including the `env -u PYTHONPATH` / ROS 2 Humble note
   (PROTECTED, :190). The :180-182 sentence about network access and working-tree mutation moves
   ahead of the command as its lead-in.

   **Cut:**
   - installation.rst:1-6 — "This page covers installing ... and verifying". Page-about-page
   opener (#21 rule 1); the H1 already says install and the first section already shows the
   command.
   - installation.rst:8-13 — the f110_gym divergence `.. note::`. Canonical home is index.rst,
   which states it next to the gym id. One clause remains here at most.
   - installation.rst:15-27 — the `Requirements` bullet list. Every item is either enforced by the
   command that follows (Python version, dependency resolution) or restated where it bites
   (network on first run -> Verify; display -> Turn on rendering). Hoisted prerequisites are what
   pushes the install command to line 55.
   - installation.rst:28-47 — the `Core dependencies` inventory. Ten packages glossed one by one;
   the installer resolves them, and the list rots the moment `pyproject.toml` changes (it already
   lists `coverage`/`pandas` peers dropped in 595e605). The dev-group sentence at :45-47 goes with
   it.
   - installation.rst:98-104 — `First run downloads the map` as a standalone section. Its content
   survives, but as the read-out of the snippet that triggers the download; a caveat sits after
   the code that provokes it, not in its own section (#21 rule 4).
   - installation.rst:106-111 — the second `.. warning::` in that section. Merged into the same
   read-out; two admonitions on one page-screen with no prose between them is the pattern the
   batch is removing.
   - installation.rst:116-119 — "This snippet creates ... steps it a few times, prints a value,
   and closes." Narrates in prose what the 28-line block below says (#21 rule 8). Replaced by one
   lead-in sentence ending in a colon.
   - installation.rst:155-163 — the closing `.. note::`. Three registered facts restated at once
   (5-tuple, steering column 0, DIRECT has no pose_x); :160 is one of the six duplicate statements
   of the column-0 fact counted in #21 tell 2. Replaced by one clause pointing at :doc:`actions`
   and :doc:`observations`.
   - installation.rst:153 — the parenthetical "(``x=-1.567``, ``y=-1.257``)" labelled as the start
   pose. Measured: with `seed=42` the spawn is `x=-0.044, y=-0.849`; `-1.567 / -1.257` is the pose
   after the 100-step loop. The number is kept, the label is corrected. (CLAUDE.md carries the
   same mislabel.)
   - installation.rst:195-203 — the four-bullet `Next steps` closer. Re-lists the sidebar (#21
   rule 10); becomes one `Next:` line to :doc:`quickstart`.
   - installation.rst:4,59,84,91,95 — `dev-humble` in five places, including two `git checkout`
   lines and a `pip install "git+...@dev-humble"`. All five are deleted, not retargeted: at v1.0.0
   the install is a plain clone or `pip install`. The editable-clone recommendation survives on its
   own technical merit — the wheel ships no `maps/`, and `track/utils.py` resolves the download
   directory four `.parent` hops up from its own source.

### `docs/quickstart.rst` — Quickstart

**Mode:** tutorial · **Target:** ~150 lines

**Job:** Can I drive one episode end to end, understand why it stopped, and read enough state to steer?

**Opening lines (draft):**

> An episode ends when the simulator decides it does — on a crash, or once the lap target is met —
> not when your loop runs out of iterations. We will drive one to that end and work out which of
> the two stopped it.

1. **Drive one episode** `[code]`
   *Reader can:* You can produce a complete rollout, and you have seen the environment end it for
   you: the loop is `while not (terminated or truncated)`, and it exits after 1792 steps without
   any step budget in your code.
   *Source:* NEW program, distilled from quickstart.rst:17-47. Uses the *default*
   `EnvConfig(render_enabled=False)` — no `max_laps=None` yet, because the default ending is the
   thing the next section explains. Verified output: `steps: 1792 sim_time: 17.92`. The `seed=42`
   gets one clause (identical spawn on every run) plus a :doc:`reproducibility` link, replacing
   :86-88.

2. **Find out why it ended** `[code]`
   *Reader can:* You can diagnose any episode end instead of guessing, and you can choose how long
   episodes last — the two exits are a collision and the lap target, and only one of them was
   yours.
   *Source:* quickstart.rst:99-119 condensed to the two elements you act on
   (`terminated`/`truncated`, and `info`), plus :192-194 lifted out of the terminal traps bucket
   to the point where it bites. The `with_updates` nesting demo (:66-74) moves here, now
   motivated: it is how you set `max_laps=None`. Verified output: `collisions: [1.] lap_counts:
   [0.]` — a crash, not a lap, and the reward accumulated to 17.92 because the default reward is
   survival time.

3. **Read the car's state** `[code]`
   *Reader can:* You can get any quantity the car knows out of the observation dict: what the
   eight default keys are, that scalars are 0-d float32 arrays, and where x/y/yaw live under the
   default preset.
   *Source:* quickstart.rst:143-183, reordered so `std_state` (what the default gives you) comes
   before the `KINEMATIC_STATE` switch. Verified: `sorted(obs["agent_0"])` prints the eight DIRECT
   keys; `scan.max()` is `0.0` right after `reset` and `30.0` after one `step` — the PROTECTED
   :90-94 caveat, now with a printed observable instead of a claim. The `pose_x` absence
   (:199-203) shrinks to one line + :doc:`observations` link. NEW and load-bearing:
   `KINEMATIC_STATE` has only five keys and drops `scan`, so the swap the old page recommends
   costs you the LiDAR.

4. **Command the car** `[code]`
   *Reader can:* You can command steering and speed and predict how fast a command is realised,
   which is the last thing you need before writing a controller that reads state and acts on it.
   *Source:* quickstart.rst:124-135 condensed. NEW observable, verified: one step of `[[0.2,
   2.0]]` from rest leaves `std_state[2] = 0.032` and `std_state[3] = 0.019` — the commands are
   targets tracked by actuators, and `sv_max = 3.2 rad/s` over a `0.01 s` timestep is exactly
   `0.032`. The column-0 layout gets one clause + :doc:`actions` link (its canonical home, and the
   only page explaining why a swap cannot be caught).

   **Cut:**
   - quickstart.rst:4-6 — "This page walks through a minimal, fully runnable episode: build an
   environment, reset it, step ... and read the ego observation." A prose table of contents as
   sentence 1, and a comma-list of the page's own headings (#21 tell 1).
   - quickstart.rst:8-12 — the map-download `.. note::`. Canonical home is :doc:`installation`,
   where the first run actually happens; here it is a second copy on a page the reader reaches
   after installing.
   - quickstart.rst:49-64 — `Creating the environment`. Explanation on a tutorial page: frozen-
   dataclass philosophy, why the id is namespaced, what the defaults are. The defaults belong to
   :doc:`configuration`; the tutorial keeps one clause (single agent, Spielberg, ST) inside
   section 1's read-out.
   - quickstart.rst:76-88 — the `reset() -> (obs, info)` section, heading included. It restates
   the shape of a call the program above already makes, and :79 restates its own heading (#21 rule
   2).
   - quickstart.rst:99-111 — the five-bullet gloss of the 5-tuple. An API-shape dump; a tutorial
   teaches the two values that change what you do (`terminated`, `truncated`) and shows `info` by
   printing it.
   - quickstart.rst:137-141 — the `.. warning::` that a column swap "fails silently". Registered
   to actions.rst, and the wording is being replaced repo-wide under #8 because it reads as a
   library defect rather than the undetectable-by-construction fact it is.
   - quickstart.rst:185-203 — `The three classic traps`. A terminal Gotchas bucket holding one
   warning with three numbered items, all of them registered facts (max_laps, column 0,
   DIRECT/pose_x) already stated once each above. Every item redistributes to the section whose
   code provokes it: max_laps -> `Find out why it ended`, column 0 -> `Command the car`, pose_x ->
   `Read the car's state`.
   - quickstart.rst:205-209 — `Where to go next`. Becomes one `Next:` line to :doc:`examples`,
   which is the pure-pursuit controller this page's last two sections have just qualified the
   reader to read.
   - quickstart.rst:6 — "on branch ``dev-humble``". Deleted outright; a tutorial that opens by
   naming a branch is telling the reader about the repository, not the simulator.


---

## Guides

### `docs/rl.rst` — How to train an RL agent against this environment

**Mode:** how-to · **Target:** ~175 lines

**Job:** How do I get a flat, finite, single-agent Gymnasium interface with a reward that is not just survival time?

**Opening lines (draft):**

> The environment is natively multi-agent: ``reset`` hands back ``dict[agent_id -> dict[field ->
> ndarray]]`` and ``step`` expects an ``(num_agents, 2)`` array. Two wrappers and one reward mode
> collapse that into the flat finite ``Box`` pair that Stable-Baselines3, CleanRL, and
> ``gymnasium.utils.env_checker.check_env`` all assume.

1. **Build the flat single-agent env** `[code]`
   *Reader can:* The reader has an env object a standard RL library can consume, and a printed
   episode return proving it steps.
   *Source:* rewards_and_rl.rst:328-385 — the runnable program, PROMOTED from the twelfth section
   to the first; it is the only block on the old page that resets, steps and prints. Absorbs
   :296-310 (SingleAgentWrapper's contract) into the read-out and :387-391 (the flat (2,) action)
   into one clause + :doc:`actions` link. Verified output: obs.shape (5,), 'return: 58.5114
   progress: 0.03001', and check_env PASS on the composed stack.

2. **Choose what the policy sees** `[code]`
   *Reader can:* The reader can size the policy input and index the flat vector correctly, instead
   of discovering the ordering by trial.
   *Source:* NEW, both measured. A preset -> flattened-dim table: KINEMATIC_STATE (5,),
   DYNAMIC_STATE (7,), FRENET_DYNAMIC_STATE (8,), DIRECT (1101,) — the last being 1080 scan beams
   plus 21, which is the whole argument for not training on the default. Plus the
   FlattenObservation key order under KINEMATIC_STATE: ['delta', 'linear_vel_x', 'pose_theta',
   'pose_x', 'pose_y']. Field vocabulary itself stays in :doc:`observations`. Placed before both
   reward sections because a CUSTOM reward reads obs, so obs must be settled first.

3. **Shape the built-in reward** `[code]`
   *Reader can:* The reader can set the four weights against each other knowing the magnitude of
   each term, rather than guessing at ratios.
   *Source:* rewards_and_rl.rst:27-50 (RewardMode table); :66-73 (SURVIVAL, reduced to one table
   row and one sentence); :74-95 (PROGRESS, with the formula literal block at :79 kept verbatim —
   PROTECTED); :96-102 (the compute_frenet_frame ValueError, kept as the page's one warning);
   :104-116 (the config block, now extended to print); :118-120 (centerline frame -> one clause +
   :doc:`tracks`). NEW and load-bearing, measured: at 3 m/s with progress_weight=1.0 and
   velocity_weight=0.1 each step returns 0.330 = 0.030 progress + 0.300 velocity, so the velocity
   term outweighs progress ten to one and collision_penalty=10.0 erases thirty steps of driving.
   That number is exactly what ISSUES_PLAN tell #5 says a weight-tuning reader never learns.

4. **Compute the reward yourself** `[code]`
   *Reader can:* The reader is free of the built-in modes entirely and knows what raw material
   info carries each step.
   *Source:* rewards_and_rl.rst:122-144 (CUSTOM); :51-64 (the raw info signals, MOVED DOWN from
   ahead of the mode table to here — progress and collisions are the material a custom callable
   reads, not a preamble to the built-ins); :145-151 (the derived-field caveat -> the block's
   read-out + :doc:`observations` link)

5. **Scale to many environments** `[code]`
   *Reader can:* The reader can run N workers without hitting a pickling error on the wrappers.
   *Source:* rewards_and_rl.rst:291-294 (both wrappers are RecordConstructorArgs, so they pickle)
   + NEW: gym.make_vec with vectorization_mode 'sync' and 'async', both already pinned by
   tests/test_f110_env.py:156-229 and currently absent from the docs entirely

6. **What this environment does not ship**
   *Reader can:* The reader stops looking here for PPO or MPC and knows which repository to fetch
   each from.
   *Source:* rewards_and_rl.rst:10-25 — the 'Scope boundary' section, DEMOTED from first to last.
   It is a handoff, not an introduction, and putting it first is what forced the old page's code
   to arrive at line 35. Carries the single permitted 'Next:' line to :doc:`sim2real` and
   :doc:`reproducibility`.

   **Cut:**
   - rewards_and_rl.rst:153-199 (domain randomization), :202-238 (ControlConfig actuation noise),
   :240-286 (LiDAR noise), :311-326 (ObservationDelayWrapper) — all four move to
   docs/sim2real.rst, where they are reorganised around loop position rather than config class.
   - rewards_and_rl.rst:13 — 'simulation-only, **with clean interfaces**'. Banned by rule 12 and
   named in ISSUES_PLAN as self-praise.
   - rewards_and_rl.rst:393-405 — the 'Reproducibility' closer. Reduced to one clause +
   :doc:`reproducibility` link inside the final section; reproducibility.rst is the canonical home
   and EnvConfig.seed reaching nothing is its protected fact.
   - rewards_and_rl.rst:281-286 — the field_of_view with_updates no-op note. Registered to
   configuration.rst:399, which is the better version and is protected there.
   - rewards_and_rl.rst:4-8 — the four-line prose table of contents ('This page covers everything
   you need to...'). Rule 1.
   - rewards_and_rl.rst:296-310's bullet list — the three SingleAgentWrapper facts become the
   read-out of the lead program rather than a standalone section, so the wrapper is introduced by
   being used.

### `docs/rendering.rst` — Rendering

**Mode:** explanation · **Target:** ~185 lines

**Job:** When does a frame actually get produced, and how does that relate to how fast the simulation is running?

**Opening lines (draft):**

> The renderer is fully decoupled from the physics step: you own the step loop, and the
> environment only draws when you call ``env.render()``. Three clocks therefore run at once — how
> fast simulated time advances against the wall, how often a distinct frame is produced, and what
> frame rate a recorded video claims — and nearly every rendering question is really a question
> about which of the three was meant.

1. **The three clocks** `[code]`
   *Reader can:* The reader can predict, for any (timestep, render_fps, real_time_factor) triple,
   which steps emit a distinct frame and what fps a recorded video will claim.
   *Source:* rendering.rst:9-12 (thesis, protected — promoted to sentence 1); :90-105 (the
   canonical RTF-vs-redraw statement, now the only one); :107-111 (the 'unlimited' note, folded in
   as a table row); :113-124 (set_real_time_factor); :126-132 (metadata["render_fps"] note,
   PROTECTED — folded in as the third clock so it lands beside its two siblings instead of alone);
   NEW measured block

2. **Choosing a render mode** `[code]`
   *Reader can:* The reader has a window open or a frame in hand, and knows which clock their
   chosen mode pins.
   *Source:* rendering.rst:21-52 (the 'Enabling rendering' H2 dissolves — render_enabled and
   render_mode are two arguments, not a section); :54-81 (mode table; its 'Real-time factor'
   column becomes a back-reference now that clock 1 is defined); :207-209 (north-up orientation,
   moved here as the read-out of the first window the reader sees)

3. **Why the backend needs a display** `[code]`
   *Reader can:* A headless or Colab reader has a working GL context and understands why no
   offscreen path exists.
   *Source:* rendering.rst:14-19 (the top-of-page note, DEMOTED out of the header — a caveat must
   not be hoisted to the top); :82-88 (the no-$DISPLAY RuntimeError warning, restated here where
   it is actionable rather than beside the mode table); :313-340 (xvfb-run, pyvirtualdisplay, CI
   skip)

4. **Grabbing frames and recording video** `[code]`
   *Reader can:* The reader has an MP4 on disk that plays back at real time, and knows why its
   container fps is 100 and not 60.
   *Source:* rendering.rst:211-241 (rgb_array loop; keeps the printed (600, 600, 3) uint8 read-
   out); :243-256 (RecordVideo recipe); :258-263 (moviepy warning, compressed —
   installation.rst:68 is the canonical home of the uv sync uninstall trap)

5. **Drawing your own geometry** `[code]`
   *Reader can:* The reader can overlay a LiDAR scan, a local plan, or arbitrary world points on
   the live scene.
   *Source:* rendering.rst:265-302 (add_render_callback + make_lidar_scan_callback); :304-311 (the
   trailing note collapses into the block's read-out)

   **Cut:**
   - rendering.rst:134-209 — the entire RenderConfig field table. Canonical home is
   configuration.rst:418-463 per the register; replaced by one clause naming render_fps,
   real_time_factor and focus_on plus a :doc:`configuration` link. ~50 lines.
   - rendering.rst:197-205 — the configure()/with_updates re-config block. Config mutation is
   configuration.rst's job; one clause survives.
   - rendering.rst:4-7 — the 'f1tenth_gym ships a single OpenGL rendering backend' opening
   paragraph. Displaced so the protected :9 becomes sentence 1; the backend name survives as one
   clause in the top .. seealso::.
   - rendering.rst:57-60 — 'the three human modes draw and pace; rgb_array never sleeps'.
   Restatement #2 of the two-clock idea; the mode table now carries it as data.
   - rendering.rst:96-99 and :100-105 bullet framing — merged into the single clocks table. :100
   bolds a lone opening parenthesis (**(** ``render_fps`` **)**) purely to preserve the bullet
   template; gone with it.
   - rendering.rst:342-351 — 'A standalone dashboard'. telemetry_plot.py has its own entry in
   examples.rst; a rendering page should not host a pointer to a script that is explicitly not
   wired into the renderer.
   - rendering.rst:353-359 — the 'See also' closer (house rule 7). Its four targets are already
   linked inline.
   - rendering.rst:137-141 — 'the old ``RenderSpec`` was folded into it'. Repo archaeology, banned
   by rule 12.

### `docs/examples.rst` — How to run the bundled examples

**Mode:** how-to · **Target:** ~125 lines

**Job:** Which script in examples/ do I run to see the thing I care about, and what does it need before it will start?

**Opening lines (draft):**

> Five scripts in ``examples/`` drive the simulator through the ordinary Gymnasium API — no
> private hooks, no test harness. Invoke any of them by file path from the repository root: Python
> puts the script's own directory on ``sys.path``, which is what resolves their sibling imports of
> ``PurePursuitPlanner``.

1. **Run the first example** `[code]`
   *Reader can:* The reader has watched a car drive a lap and confirmed their install, map cache,
   and display all work.
   *Source:* examples.rst:4-8 (prerequisite prose); :54-58 (the waypoint_follow run command, with
   the malformed ``::`` + ``.. code-block::`` pair at :54 fixed per #24); :170-178 (the run-
   location and first-run-download prerequisites, MOVED from the last section to the first — a
   prerequisite stated after four run commands is not a prerequisite). The sibling-import
   instruction at :173-176 is CORRECTED: verified that `python examples/waypoint_follow.py` from
   the repo root already resolves the import, so the current 'run them from inside the examples/
   directory' advice is wrong.

2. **Pick a script**
   *Reader can:* The reader knows which of the five scripts matches their goal and what extra
   dependency it will cost them, without reading five sections.
   *Source:* NEW selection table (goal | script | command | extra install), condensing the five H2
   headers at :22, :70, :103, :129, :158 plus the two dependency notes at :146-156 (moviepy) and
   :163-168 (shapely)

3. **Drive a lap with pure pursuit** `[code]`
   *Reader can:* The reader can open the canonical fully-spelled-out EnvConfig and change one
   field of it with confidence.
   *Source:* examples.rst:22-46 (the 'two things at once' framing at :25 is PROTECTED — the
   planner, and build_config() as the config reference); :60-62 (the headless variant). The stale
   #9 warning at :64-68 is REPLACED, not deleted: beyond max_reacquire (20 m) plan() falls back to
   speed=4.0, steer=0.0; between the lookahead radius and 20 m it steers at the nearest raceline
   point.

4. **Record the rollout to MP4** `[code]`
   *Reader can:* The reader has a video file on disk and knows which directory it landed in.
   *Source:* examples.rst:129-144 (with the malformed ``::`` pair at :140 fixed); :146-156
   compressed to the command plus one clause — installation.rst:68 owns the uv sync uninstall trap
   and rendering.rst owns how rgb_array produces frames

5. **Watch the state live** `[code]`
   *Reader can:* The reader can plot any std_state channel from their own loop, at a refresh rate
   decoupled from the physics rate.
   *Source:* examples.rst:70-101 (with the malformed ``::`` pair at :92 fixed); the four flags at
   :98-101 become a small table (--map, --rtf, --fps, --window)

6. **Build your own track** `[code]`
   *Reader can:* The reader can substitute a synthetic reference line or a generated circuit for a
   downloaded map.
   *Source:* examples.rst:103-114 and :123-127 (run_in_empty_track, with the malformed ``::`` pair
   at :123 fixed) MERGED with :158-168 (random_trackgen). Both author tracks and both carry an
   undeclared dependency, so they are one job, not two. The force-closed-loop warning at :115-121
   becomes one clause + :doc:`tracks` link — tracks.rst:163 is its protected canonical home.

   **Cut:**
   - examples.rst:9-20 — the shared 'Every example creates the env the same way' note. Its :17-20
   restatement of the [steering, longitudinal] action layout is one of the six copies of a
   registered fact whose canonical home is actions.rst; the gym.make snippet duplicates
   quickstart.rst. Reduced to one clause + two links.
   - examples.rst:64-68 — the max_reacquire numba TypingError warning. Stale: fixed in commit
   43ebc25 (ISSUES_PLAN #9). Deleting it outright would lose a real behaviour, so it is replaced
   by the accurate fallback description rather than dropped.
   - examples.rst:161 — 'Included for completeness.' Banned by rule 12, and it was the entire body
   of that section; merging random_trackgen into 'Build your own track' removes the section that
   had nothing to say.
   - examples.rst:48-52 — the DIRECT-has-no-pose_x explanation. Registered to observations.rst;
   one clause + link survives.
   - examples.rst:115-121 — the force-closed-loop warning body. Registered to tracks.rst:163.
   - examples.rst:170-178 as a trailing section — the content moves to section 1 and its sibling-
   import claim is corrected; nothing is left to close the page with.
   - The four malformed ``::`` + ``.. code-block::`` pairs at :54, :92, :123, :140 (ISSUES_PLAN
   #24) — docutils emits 'Literal block expected; none found' and swallows a colon at each site.


---

## Explanation

### `docs/dynamics.rst` — Choosing a dynamics model

**Mode:** explanation · **Target:** ~160 lines

**Job:** Which of the two usable models should I integrate, and what silently changes when I switch?

**Opening lines (draft):**

> .. seealso:: :mod:`f1tenth_gym.envs.dynamic_models` — ``DynamicModel``, ``VehicleParameters``,
> the three presets. ``KS`` integrates a five-state kinematic bicycle; ``ST`` adds tyre stiffness,
> a yaw rate and a slip angle. Under the same steering command the two diverge by metres inside a
> single corner, not by centimetres.

1. **Slip is the whole difference** `[code]`
   *Reader can:* The reader picks KS or ST from a measured divergence rather than from an
   adjective, and stops considering MB.
   *Source:* Prose compressed from :58-74 (KS/ST paragraphs); model/state-layout table from
   :29-52; the runnable block is :242-283 retargeted — same seed, same action, an open 30 m circle
   via Track.from_refline so no wall truncates the run. MB warning from :76-83, cut to two lines
   and placed immediately under the MB table row (rule 5). NEW: pasted output, verified this
   session — 300 steps of [[0.2, 7.0]] from reset(seed=42): KS ends (29.3847, -1.2623) with beta
   0.0 and ang_vel_z 0.0; ST ends (27.5983, 2.7881) with beta -0.3126 and ang_vel_z 3.1187;
   separation 4.43 m. Read-out names the 4.43 m and the identically-zero KS beta as the
   observables.

2. **Where the pose is measured** `[code]`
   *Reader can:* The reader can compare logged trajectories across models without absorbing a
   silent 0.17145 m frame shift, and knows why no assertion catches it.
   *Source:* Canonical home per the register. Converts the :102-110 warning to prose (a whole
   section titled after the fact does not also need a box). std_state layout and the beta-not-V_Y
   correction from :85-97. NEW from CLAUDE.md: the spawn pose is written verbatim into both
   models' state[0:2], so at t=0 the two numbers are identical while the cars sit 0.17145 m apart
   — the offset is in interpretation, not in the value. NEW: the only two reconciliation sites
   (_build_scan_cache, _compute_collision_body_offset) both test `!= DynamicModel.KS`, i.e. 'is it
   KS', not 'is it rear-axle referenced'. Doctest verified this session: `>>> P.lr, P.lf + P.lr`
   -> `(0.17145, 0.3302)`.

3. **What the model reads** `[code]`
   *Reader can:* The reader can change the car — mass, grip, steering and speed limits — without
   changing the model, and sees the change land in an observation.
   *Source:* Presets from :112-123 (keep the FULLSCALE-only-populates-MB clause). Parameter table
   from :128-177, trimmed to the knobs actually turned and collapsing the twin min/max rows to ±
   form; keep the wheelbase line :178. PROTECTED :184 (terse field names — mass is `m`, CoG height
   is `h`, yaw inertia is `I`) stays as prose directly under the table, not folded into a cell.
   with_updates from :198-206. NEW: pasted output, verified this session — the same ST rollout
   with `with_updates(m=4.0, mu=0.9)` ends (28.1689, 2.4051) with beta -0.3595, 0.6872 m from the
   nominal run. That is the number the current page never produces.

4. **Why the parameter list is append-only**
   *Reader can:* The reader can add a vehicle parameter without silently corrupting every existing
   KS/ST rollout, because they know what the constraint buys.
   *Source:* PROTECTED :190 (append-only ABI) expanded from :188-193. NEW from CLAUDE.md:
   `to_array` is `astuple()[:parameter_count()]`, verified 18 entries for KS/ST and 89 for MB;
   every njit kernel indexes positionally (`mu = params[0]` … `v_max = params[15]`, `width` 16,
   `length` 17). The trade-off belongs here on an explanation page: a flat float32 array is why
   the kernels jit at all, and the price is that declaration order is an ABI — a dict or dataclass
   cannot cross the boundary.

5. **How far one step actually integrates** `[code]`
   *Reader can:* The reader trades integrator order against throughput on a measured difference,
   and knows where the float32 boundary spends part of what RK4 earned.
   *Source:* Integrator list and substep description from :215-225. NEW: pasted output, verified
   this session — the same ST rollout under EULER ends (27.6457, 2.7897) against RK4's (27.5983,
   2.7881), 0.0475 m apart after 3 s of hard cornering. Pair it with the measured cost from
   CLAUDE.md (RK4-over-Euler is ~9% of step time) so the trade is stated in both currencies. NEW:
   constraints re-apply inside every RK4 stage against intermediate states, and the result is re-
   cast to float32 at the step boundary. The timestep-multiple rule is one clause plus a
   :doc:`configuration` link.

   **Cut:**
   - :6-9 — the four-line prose table of contents; it lists the sidebar and states no fact.
   - :14-19 — the 'there is no RaceCar class in this fork' note; repo archaeology (house rule 12),
   and the SoA design is not a fact the model chooser needs.
   - :21-27 — the 'The models' H2 wrapper and its config-plumbing preamble; the model choice now
   opens the page, and the SimulationConfig.dynamics_model path is one clause.
   - :58-59 and :67-68 — the twin H3 headings 'Kinematic single-track (KS)' / 'Single-track (ST,
   default)'; identically-shaped sibling sections are the machine rhythm, and their content
   becomes one comparison.
   - :182-186 — the note wrapper only; PROTECTED :184 survives as prose (admonition budget).
   - :227-233 — the timestep-must-be-an-exact-multiple warning; canonical home is
   configuration.rst:253 per the register, and the plan says delete this copy.
   - :235-241 — the 'Runnable example' H2 and its three-sentence narration of what the code below
   does (house rule 8); the block itself moves up to section 1.
   - :259-269 — the fully-spelled-out EnvConfig boilerplate in that example; the comparison block
   sets only what differs between the two runs.
   - :285-296 — the live `configure()` block; one clause plus a :doc:`configuration` link.
   - :298-305 — the 'See also' closer (house rule 7); replaced by at most one 'Next:' line.

### `docs/tracks.rst` — Maps, racelines and the Frenet frame

**Mode:** explanation · **Target:** ~165 lines

**Job:** What is a track made of, and why is my lateral error already 0.81 m before I have moved?

**Opening lines (draft):**

> .. seealso:: :mod:`f1tenth_gym.envs.track` — ``Track``, ``Raceline``, ``CubicSplineND``. A track
> is an occupancy grid plus two closed reference lines, and they are not the same loop.
> Spielberg's centerline runs 343.32 m and its raceline 338.13 m.

1. **A track carries two lines, not one** `[code]`
   *Reader can:* The reader can read either reference line off a live track and sees, in printed
   numbers, that the two differ in length and in speed profile.
   *Source:* Prose from :80-83 (the two Raceline objects, the periodic 7-channel spline, the hard-
   coded fixed_speed=1.0 on centerlines built from a bare CSV). NEW: pasted output, verified this
   session — `centerline.spline.s[-1] = 343.3222`, `raceline.spline.s[-1] = 338.1253`,
   `centerline.vxs[:3] = [1. 1. 1.]` against `raceline.vxs[:3] = [8. 8. 8.]`. The read-out names
   the 1.0 m/s centerline profile as the observable that proves it is not a racing line. First
   position because every later section is a consequence of there being two lines.

2. **You spawn on one line and are measured against the other** `[code]`
   *Reader can:* The reader stops treating a non-zero lateral error at t=0 as a bug in their own
   code, and knows the exact baseline to subtract.
   *Source:* The keystone, promoted from :87-92. PROTECTED :92 survives as the ey-is-non-zero-at-
   spawn statement. THE ~1.5% LAP-LENGTH ERROR CLAUSE IN :92 MUST BE CUT — CLAUDE.md verifies it
   is false: state.frenet[:,0] is centerline arclength and _check_done divides by the centerline
   s_frame_max, so one physical lap measures 343.3222 m against s_frame_max 343.3222 -> exactly
   1.000000 laps. Replace it with what is actually true: info['progress'] and the PROGRESS reward
   are denominated in centerline arclength, so a lap always accrues 343.3 m even when the raceline
   odometer reads 338.1. NEW: pasted output, verified this session — `reset(seed=42)` gives
   frenet_pose (0.2630, 0.8086, -0.0008), and across 50 seeds ey spans only [0.808, 0.8086], so
   the offset is structural, not seed noise.

3. **Reading (s, ey, ephi)** `[code]`
   *Reader can:* The reader can write a lateral controller with the correct sign and knows the one
   geometry that makes the projection lock onto the wrong stretch.
   *Source:* Definitions and the +ey-is-LEFT sign convention from :97-101; the always-centerline
   fact from :103-105 merged in (it currently restates :88). The two conversion calls fold up from
   :112-124, rewritten as a doctest round-trip that prints a number rather than assigning to
   unused names. NEW from CLAUDE.md: the Frenet local search window is ±~4.8 m
   (`frenet_search_range = 10`, a plain Track attribute with no EnvConfig field), so an agent that
   translates further in one step silently locks onto the wrong stretch — the trade-off that buys
   the per-step cost.

4. **Where the map comes from** `[code]`
   *Reader can:* The reader can point the env at a bare name, a directory or a prebuilt Track, and
   knows which of the three costs 190x more per environment.
   *Source:* Dispatch from :11-15; download endpoint, tar hardening and the editable-install
   constraint from :20-24 and :42; the custom-directory path from :68-75; the centerline=None
   crash from :61-63, moved here (it is a map-authoring failure, not a Frenet one) and kept as the
   page's second warning. NEW from CLAUDE.md, two corrections the current page needs:
   `from_track_path` accepts only the legacy `{stem}_map.yaml`, so the :74 recipe cannot load any
   shipped map; and passing a Track instance as map_name cuts gym.make from 227 ms to 1.2 ms
   because the LiDAR EDT is cached on the Track — the highest-leverage knob in the repo, and
   exactly the alternatives-and-trade-offs beat an explanation page owes.

5. **What the occupancy image encodes** `[code]`
   *Reader can:* The reader can author a map image whose walls the distance transform actually
   finds, instead of one that reads inside-out.
   *Source:* resolution / origin / FLIP_TOP_BOTTOM from :51-53; the reference-line CSV column
   conventions from :59. :55-57 IS INVERTED AND MUST BE CORRECTED (ISSUES_PLAN #1): the true
   polarity is `<=128 -> 0.0` = OBSTACLE and `>128 -> 255.0` = FREE, since get_dt is the EDT to
   the nearest zero. NEW: pasted proof, verified this session — `np.unique(occupancy_map)` is
   `[0., 255.]`, only 0.82% of Spielberg's pixels are 0, and the pixel under `raceline[0]` reads
   255.0. Keep the surviving half of :57: negate/occupied_thresh/free_thresh are parsed into
   TrackSpec and never read, so every shipped map's declared 0.45 is silently overridden by the
   hard-coded 128.

6. **Synthetic tracks are always closed** `[code]`
   *Reader can:* The reader stops trying to build an open straight and designs a deliberate
   circuit, or leaves knowing no open-path mode exists.
   *Source:* from_refline description from :129 and the circle example from :131-159, trimmed (the
   EnvConfig block keeps only what differs). PROTECTED :163 (force-closed loop, phantom return
   leg) survives near-verbatim and stays immediately after the block that provokes it. NEW: pasted
   proof, verified this session — `Track.from_refline(x=np.linspace(0,10,50), y=np.zeros(50))`
   yields 51 points and `s_frame_max = 20.0`, so the '10 m straight' is a closed 20 m path. Last
   because it is the sharpest trap and the only escape hatch for a reader with no map at all.

   **Cut:**
   - :4 — the 509-character unwrapped prose table of contents; it is both the machine-template
   opener and the source-hygiene outlier. The whole file rewraps at 88 columns.
   - :6 — the two-link navigation sentence ('selected by EnvConfig's map_name ... see
   configuration ... see quickstart'); one clause carries it.
   - :8-9 — the 'How a map is resolved' H2 as the page's first section; resolving a map before the
   reader knows what a track holds is the current spine's inversion.
   - :26-38 — the six-line 'downloaded on first use' EnvConfig block; it constructs an env, resets
   and closes without printing anything, violating the every-config-example-prints-a-number rule.
   Its one fact (first use downloads) becomes a clause.
   - :40-42 — the note wrapper around the download prerequisites; the FileNotFoundError text and
   the network requirement move into section 4's prose (admonition budget: 1 note, 2 warnings for
   the page).
   - :92, the '~1.5% lap-length error' clause only — verified false; see section 2's sources. The
   ey half of :92 is protected and survives.
   - :103-105 — the 'always centerline-based' note; it restates :88, which section 2 now owns.
   - :107-108 — the 'Converting coordinates directly' H3 and its 'You can transform points
   yourself' lead-in; the calls fold into section 3 as a doctest.
   - :124 — 'Pass use_raceline=True to either method'; kept as a clause, but the standalone
   sentence goes, since the sim never passes it.
   - :165 — the update_map / configure sentence; one clause plus a :doc:`configuration` link.
   - :167-173 — the 'See also' closer (house rule 7).

### `docs/sim2real.rst` — Modelling the sim-to-real gap

**Mode:** explanation · **Target:** ~150 lines

**Job:** Which hardware imperfections can this simulator reproduce, where does each one enter the control loop, and how often is it redrawn?

**Opening lines (draft):**

> A policy trained here fails on hardware for three separable reasons: the command the actuator
> executes is not the command the policy sent, the car executing it is not the car the policy was
> tuned on, and the measurement fed back is not the truth. Each stage can be corrupted
> independently, and each corruption is redrawn either every step or once per episode.

1. **Three stages, two cadences** `[code]`
   *Reader can:* The reader can locate any noise knob on a three-by-two grid and predict whether
   it varies within an episode or only between episodes — which is the distinction the old page
   buried in a parenthetical.
   *Source:* NEW organising table, assembled from rewards_and_rl.rst:204-222 (ControlConfig
   fields), :247-259 (LiDAR fields) and :156-164 (DR prose). This table is the replacement for the
   three identically-shaped one-section-per-config-class blocks; the axes are loop position
   (command / plant / sensing) and redraw cadence (per step / per episode), which is what makes
   range_bias_std and noise_std different knobs rather than near-duplicates. Code is the cadence
   proof: reset twice at seed 42 -> mu=1.0717 m=3.4389 both times, seed 7 -> mu=1.0551 m=3.8972.

2. **The command the actuator executes** `[code]`
   *Reader can:* The reader can make the actuator disagree with the policy in both amplitude and
   time, and knows which of the two is applied first.
   *Source:* rewards_and_rl.rst:202-238. Keeps the noise-before-the-delay-buffers ordering at
   :236-238 — the only mechanism fact on the page. Promotes steer_delay_steps to a first-class row
   alongside throttle_delay_steps; today it appears only as an aside at :222. The 'byte-identical
   to before' sentence at :206 is replaced by 'All four default to 0.'

3. **The car the command acts on** `[code]`
   *Reader can:* The reader can train across a distribution of vehicles instead of one point
   estimate, and knows which array is the ground truth the physics kernels read.
   *Source:* rewards_and_rl.rst:153-199. Keeps sim.params_array as the observable (:195-199) and
   the actuation-limits warning (:189-193), which is honest until ISSUES_PLAN #6 lands
   widest_params. The field-name note at :181-187 folds into the code read-out (m, not mass; h,
   not h_cg).

4. **The measurement the policy reads back** `[code]`
   *Reader can:* The reader can degrade perception in amplitude, dropout, calibration and time
   without changing when a crash fires.
   *Source:* rewards_and_rl.rst:240-273 (dropout_prob and range_bias_std) plus :275-279 (PROTECTED
   — noise reaches the observed scan only, collisions use the clean scan), MERGED with :311-326
   (ObservationDelayWrapper), which moves here from its orphaned position beside
   SingleAgentWrapper: sensing lag is the exact analogue of throttle_delay_steps and belongs on
   the sensing stage. Two measured observables: dropout_prob=0.02 clamps 31 of 1080 beams to
   range_max while collision stays 0.0; with delay_steps=3 the policy at step 6 reads
   pose_x=-0.0466, the true value from step 3.

5. **Pinning the whole loop**
   *Reader can:* The reader can A/B two noise settings and trust that the difference came from the
   setting and not the RNG.
   *Source:* rewards_and_rl.rst:393-405, compressed to roughly five lines: reset(seed=...) fixes
   the DR draw, the command noise, the per-step dropout and the per-episode range bias;
   EnvConfig.seed is not that knob. One clause + :doc:`reproducibility` link, which is the
   protected canonical home (reproducibility.rst:58, :127, :178).

   **Cut:**
   - rewards_and_rl.rst:10-25 (scope boundary), :27-151 (all reward material), :288-310
   (SingleAgentWrapper), :328-391 (the full RL example) — all belong to docs/rl.rst.
   - rewards_and_rl.rst:206 — 'All default to 0, so the simulation is **byte-identical to before**
   unless you set them.' Release-note voice, banned by rule 12 and named explicitly in
   ISSUES_PLAN.
   - rewards_and_rl.rst:281-286 — the field_of_view with_updates no-op note. Registered to
   configuration.rst:399.
   - rewards_and_rl.rst:181-187 — the standalone 'field names are the actual VehicleParameters
   fields' note. Becomes the read-out of the DR code block instead of a third admonition.
   - The three parallel H2s 'Domain randomization' / 'Sim2real: actuation realism' / 'Sim2real:
   richer LiDAR noise' — each opened with the same beat (name the config class, state the
   defaults, show a table, show a cfg=EnvConfig block that prints nothing). That shared rhythm is
   the machine tell; the loop-position spine replaces it and none of the four sections can now be
   swapped without breaking the traversal of the signal.

### `docs/reproducibility.rst` — Reproducibility

**Mode:** explanation · **Target:** ~145 lines

**Job:** What must I pin so an episode replays bit-for-bit, and what silently unpins it?

**Opening lines (draft):**

> .. seealso:: :class:`~f1tenth_gym.envs.f110_env.F110Env` — ``reset``, ``step``;
> :class:`~f1tenth_gym.envs.env_config.DomainRandomizationConfig`. One call fixes an entire
> episode. ``reset(seed=...)`` seeds ``np_random``, and the spawn pose, LiDAR noise, actuator
> noise and domain-randomization draw all descend from it.

1. **Pin these four things** `[code]`
   *Reader can:* The reader leaves the first screen with a template whose assert passes, and knows
   which four inputs it pinned.
   *Source:* The inversion the plan asks for: the deliverable from :185-193 and the proof from
   :100-133 fuse into one opening section, so the checklist is the code's read-out rather than a
   terminal bullet list. PROTECTED :127 (the twin `rollout(seed=42)` calls and the array_equal
   assert) survives near-verbatim, with the horizon shortened to 40 steps so the run does not
   terminate on a wall mid-proof. NEW: pasted output, verified this session — `A == B` is True
   over 40 steps, while `rollout(seed=7)` diverges from `rollout(seed=42)` by 0.7726 m at the same
   index. The read-out names 0.7726 as the observable that the seed, not the action sequence, is
   doing the work.

2. **Why one seed is enough**
   *Reader can:* The reader can stop hunting for a second seed, because they can name every stream
   and the order in which it is drawn.
   *Source:* The derivation chain from :48-66, kept but reordered to the actual consumption order
   in reset(): spawn pose, then the conditional domain-randomization draw, then the noise seed.
   PROTECTED :58 (control_rng at `noise_seed + 2**20`, and *why* — far enough from the per-agent
   scan seeds `noise_seed + idx` that the streams cannot collide) survives verbatim. Compress
   :17-30 to two sentences: the transition is a pure function of state, action and the seeded
   draws, and wall-clock pacing touches only the renderer. Keep :32-37's float32 point as the
   page's single note, since it bounds the claim — deterministic, but stored at float32, and
   portable only across matching NumPy builds.

3. **``EnvConfig.seed`` is not a seed** `[code]`
   *Reader can:* The reader stops setting a field that reaches nothing observable, and can
   reproduce the two-line proof themselves.
   *Source:* Canonical home per the register. Replaces :74-87, whose claim is too soft — it says
   EnvConfig.seed is the 'base/default seed the simulator falls back to', but gymnasium requires a
   reset() before the first step() and reset() unconditionally draws a fresh noise_seed from
   np_random, so the config value is always overwritten first. NEW: pasted proof, verified this
   session, both directions — with `EnvConfig.seed=12345`, three bare `reset()` calls give pose X
   = -0.4304, -0.2372, -0.4304; with config seeds 1 and 999999 but `reset(seed=42)`, the scan
   after one step is byte-identical. The :80-87 warning box is dropped: a section named after the
   trap does not also need one.

4. **What an unseeded reset actually varies** `[code]`
   *Reader can:* The reader can choose deliberate per-episode variation instead of inheriting
   accidental variation, knowing its measured extent.
   *Source:* Rebuilds :163-180. PROTECTED :178 survives as a measured Spielberg spawn span, BUT
   ITS NUMBERS ARE STALE AND MUST BE CORRECTED — the page says x spans -1.6…-2.4 and y
   -1.26…-1.46; measured over 50 seeds this session, x spans [-0.8167, -0.0441] and y [-1.0562,
   -0.8492]. NEW, the mechanism that makes the window that narrow: GridResetFn masks the first
   `int(start_width / (raceline.length / raceline.n))` waypoints — 5 of Spielberg's 1692, about
   1.00 m — and draws one via rng.choice, which is also why ey stays pinned at 0.808 across all 50
   seeds. Keep :90-92's `reset(seed=episode_index)` recipe here, where it is the answer to the
   section rather than a trailing suggestion.

5. **What silently shifts the stream**
   *Reader can:* The reader knows an A/B comparison across two configs is not paired even at the
   same seed, and can spot the three switches that unpair it.
   *Source:* Entirely NEW, from CLAUDE.md's 'Seeding & reproducibility' section — the page's
   sharpest content and currently absent. The three draws happen in a fixed order but draw 2 is
   conditional, so enabling domain randomization shifts everything downstream even with a
   degenerate range that changes no physics (measured max abs scan difference 0.045); passing
   `options['poses']` skips draw 1 entirely (0.051); and `range_bias_std` is drawn from the same
   per-agent scan RNG that then feeds per-step noise, so two configs differing only in that field
   produce different noise rather than 'the same noise plus a bias'. Carries the page's one
   warning: mechanism first, imperative second. Last because it defeats everything the four
   preceding sections established.

   **Cut:**
   - :4-8 — the prose table of contents ('This page explains what the seed actually controls, how
   reset seeding follows the gymnasium contract, and how to write experiments ...').
   - :10-12 — the two-link 'If you have not yet built an environment' navigation paragraph; it
   delays code past 100 words for no fact.
   - :14-15 and :18-21 — the 'The determinism model' H2 and its no-thread / no-async / no-wall-
   clock bullets; compressed to two sentences inside section 2.
   - :39-40 — the 'What reset(seed=...) seeds' H2 heading; its content becomes section 2, whose
   title states the conclusion rather than the topic.
   - :42-44 — 'reset(seed=...) is the gymnasium way to seed, and it is the seed that matters'
   followed by a restatement of the same claim; the opener now carries it once.
   - :94-99 — the 'Reproducing two identical rollouts' H2 and its 'The following script ...' lead-
   in, which narrates what the code says (house rule 8).
   - :139-144 — the note that DIRECT has no pose_x; canonical home is observations.rst, so one
   clause plus a link, or nothing, since the code already reads std_state[0].
   - :146-151 — the steering-is-column-0 warning; canonical home is actions.rst per the register,
   and this is one of six copies. The inline code comment carries it.
   - :153-158 — the note that the first scan after reset is zeros; canonical home is
   quickstart.rst:92, so one clause plus a link.
   - :160-161 and :182-183 — the 'Unseeded episodes vary' and 'Checklist for a reproducible
   experiment' headings as separate terminal sections; the checklist moves to position 1 and the
   unseeded material becomes section 4.
   - :165-174 — the unseeded code block; it prints nothing and its two comments assert 'generally
   differ' without showing it. Replaced by the 50-seed span measurement.
   - :195 — the 'See also :doc:`quickstart`, :doc:`observations`, and :doc:`actions`' closer
   (house rule 7).


---

## Reference

### `docs/configuration.rst` — Configuration

**Mode:** reference · **Target:** ~400 lines

**Job:** What every EnvConfig field is called, what it defaults to, and what it rejects.

**Opening lines (draft):**

> ``F110Env`` takes exactly one argument — a frozen ``EnvConfig`` — and a dict, ``None`` or loose
> keyword arguments raise ``TypeError``. Both the type check and every default are readable off
> the config object before an environment exists:

1. **Top-level fields**
   *Reader can:* The reader can name any of the 17 fields, its default and which of the nine are
   themselves config objects, without reading the rest of the page.
   *Source:* configuration.rst:83-154 (17-row table kept, lead-in :86-87 and trailing :155-157
   dropped); ``seed`` row rewritten from CLAUDE.md:231+264 (today's row claims the seed feeds the
   noise RNGs — it reaches nothing observable); ``collision_check`` row gains the raw-int footgun,
   NEW from CLAUDE.md:449

2. **Deriving a config** `[code]`
   *Reader can:* The reader can produce any variant of any field, including nested ones, without
   mutating anything — the only operation the nine section tables below are reachable through.
   *Source:* configuration.rst:21-54, compressed from 34 lines to ~18; keeps the nested
   ``with_updates`` block at :47-54; adds the deep-import rule (``from f1tenth_gym import
   EnvConfig`` raises ``ImportError``) from CLAUDE.md:52, replacing the vague :155-157

3. **When validation runs** `[code]`
   *Reader can:* The reader can predict where a bad config surfaces — construction, ``gym.make``,
   or neither — and stops treating a constructed EnvConfig as a validated environment.
   *Source:* NEW from CLAUDE.md:256 (three tiers, only the first at construction time); absorbs
   the scattered validation prose at configuration.rst:37-42; the protected timestep-multiple text
   at :253-258 moves here verbatim as the worked tier-2 observable (``timestep=0.03`` constructs
   fine, ``gym.make`` raises). The SimulationConfig ``timestep`` row below then carries one clause
   + a back-link, not a second statement.

4. **``ControlConfig``** `[code]`
   *Reader can:* The reader can set what each action column means and add actuator lag or command
   noise.
   *Source:* configuration.rst:159-193 (table kept); NEW from CLAUDE.md:246 — no int coercion,
   ``steer_delay_steps=2.7`` is stored verbatim, which becomes the section's printed observable

5. **``SimulationConfig``** `[code]`
   *Reader can:* The reader can set the physics clock, integrator, dynamics model and lap rule,
   and knows which two of those coerce each other.
   *Source:* configuration.rst:207-251; the :246-251 note corrected per CLAUDE.md:247
   (SimulationConfig alone does *not* enforce FRENET⇒frenet; only ``with_updates`` and
   ``EnvConfig`` do); ``loop_counter`` row gains TOGGLE's real behaviour from CLAUDE.md:451 — zero
   laps forever, and with ``max_laps=1`` the episode never terminates either

6. **``ObservationConfig``** `[code]`
   *Reader can:* The reader can pick an observation preset or declare a custom field tuple, and
   knows which combination raises.
   *Source:* configuration.rst:260-286; NEW from CLAUDE.md:248 — no tuple coercion, a list is
   accepted and makes the config unhashable; :288-289 reduced to one clause + :doc:`observations`

7. **``ResetConfig``** `[code]`
   *Reader can:* The reader can control spawn placement and spacing, and knows which reset knob is
   unreachable from here.
   *Source:* configuration.rst:291-327; NEW from CLAUDE.md:469 (``start_width`` is reachable from
   ``make_reset_fn`` but has no ``ResetConfig`` field; a small value raises ``ValueError: a cannot
   be empty``) and CLAUDE.md:217 (``shuffle`` permutes rows only — the pose set is identical).
   Printed observable: ``ResetConfig(min_dist=2.0).reset_kwargs()``.

8. **``LiDARConfig``** `[code]`
   *Reader can:* The reader can set beam count, range and noise, and knows the one field
   ``with_updates`` cannot change.
   *Source:* configuration.rst:337-416, including the protected :397-416 warning verbatim
   (canonical home). The warning's code block becomes a doctest printing the verified
   contradiction: ``field_of_view`` 6.283 against ``angle_min/max`` still ±2.3561945.

9. **``RenderConfig``**
   *Reader can:* The reader can size the window, pace the sim and pick a camera, and knows which
   three knobs do nothing.
   *Source:* configuration.rst:418-463 — canonical home for this table, ``rendering.rst`` links
   here. NEW adjacent clauses from CLAUDE.md:455 (``focus_on=None`` parks the camera at the world
   origin, it is not a map view) and CLAUDE.md:456 (``show_wheels``/``car_thickness`` are dead —
   frames are byte-identical).

10. **``TerminationConfig``** `[code]`
   *Reader can:* The reader can bound episode length and choose whose collision ends it.
   *Source:* configuration.rst:465-487 (table); :489-495 note rewritten to drop the release-note
   voice ("is no longer hardcoded ``False``"); NEW from CLAUDE.md:471 — ``max_laps`` termination
   watches only the ego even under ``collision_agents="any"``

11. **``RewardConfig``** `[code]`
   *Reader can:* The reader can name every reward field and its validation rule; the arithmetic
   lives in :doc:`rl`.
   *Source:* configuration.rst:497-533 (table kept, semantics prose at :534-536 reduced to one
   clause + :doc:`rl`); NEW from CLAUDE.md:252 — the three weights are unvalidated and a
   ``reward_fn`` set with SURVIVAL/PROGRESS is silently ignored

12. **``DomainRandomizationConfig``** `[code]`
   *Reader can:* The reader can declare randomization ranges in the right units and knows which
   parameter names are silently inert.
   *Source:* configuration.rst:538-577 (table + the :561-570 example, extended to print a number);
   NEW from CLAUDE.md:348 (under KS/ST only indices 0-17 reach the kernels, so
   ``param_ranges={"K_zt": ...}`` is a no-op) and CLAUDE.md:450 (``param_ranges`` is a plain dict
   on a module-level default instance — mutating it bypasses validation and is process-global).
   The page's single ``.. note::`` sits here.

13. **Reconfiguring a live environment** `[code]`
   *Reader can:* The reader can swap a whole config on a running env and knows what that silently
   destroys.
   *Source:* configuration.rst:62-81, extended to print ``env.unwrapped.sim.params_array[0]``
   before and after so the swap produces a number; NEW from CLAUDE.md:453 — ``configure()``
   rebuilds the renderer with an empty callback list and leaks the GL context when
   ``render_enabled`` goes ``True``→``False``. Last because it is the only section that
   presupposes two configs.

   **Cut:**
   - configuration.rst:607-617 — the 'See also' closer re-lists seven targets already linked
   inline on a page carrying 25 :doc: references (house rule 7)
   - configuration.rst:579-605 — the terminal 'A worked example' prints nothing (violates rule 4)
   and duplicates the opener; the realistic training config belongs in the new docs/rl.rst, which
   leads with a runnable program
   - configuration.rst:21-22 heading 'The frozen-dataclass philosophy' and the :35 'Two properties
   follow from this design' frame — preamble scaffolding; the two properties survive as the
   content of two sections, the framing does not
   - configuration.rst:200-206 — the steering-is-column-0 warning; canonical home is
   docs/actions.rst, which is the only copy explaining why a swap cannot be caught. One clause +
   :doc:`actions` remains in the ControlConfig lead-in
   - configuration.rst:288-289 — 'the trap that DIRECT does not contain pose_x'; canonical home is
   docs/observations.rst, above its preset table
   - configuration.rst:329-336 — the note that ey is non-zero at spawn; canonical home is
   docs/tracks.rst, promoted there to its own section
   - configuration.rst:96-100 — the claim that EnvConfig.seed 'feeds the sim's noise RNGs'; it is
   false (CLAUDE.md:264, reset() unconditionally overwrites it) and the seeding contract's
   canonical home is docs/reproducibility.rst
   - configuration.rst:56-60 — the standalone max_laps note; the fact moves into the opener read-
   out and the SimulationConfig max_laps row
   - configuration.rst:155-157 — 'All the enum defaults come from f1tenth_gym.envs'; nothing is
   re-exported from the package roots, so the sentence misleads (CLAUDE.md:52)
   - configuration.rst:196-198 — 'byte-identical to a noiseless, lag-free actuator' (release-note
   voice, banned by rule 12 of the house style)
   - configuration.rst:491-495 — 'truncated is no longer hardcoded False' (repo archaeology; state
   what the field does)
   - All ten list-tables re-cut as simple RST tables where the notes column fits inside 88
   columns; the field content survives, roughly 150 lines of directive scaffolding does not

### `docs/observations.rst` — Observations

**Mode:** reference · **Target:** ~200 lines

**Job:** Which fields each observation preset returns, in what shape and dtype, and what bounds the space declares.

**Opening lines (draft):**

> An observation is a ``dict[agent_id -> dict[field -> ndarray]]``, and the default ``DIRECT``
> preset carries no ``pose_x`` — position arrives packed inside ``std_state``. Both facts are
> visible in the first observation ``reset`` returns:

1. **Presets and their fields** `[code]`
   *Reader can:* The reader can pick the preset that names the fields they want, instead of
   discovering by KeyError which ones it omits.
   *Source:* observations.rst:52-82 (six-row table); the :86-93 warning dissolves — its content is
   the page opener and the DIRECT row. Absorbs the FEATURES section :182-206 into the FEATURES row
   plus one doctest printing the real key list. Absorbs two of the four terminal warnings as
   DIRECT-row clauses: :249-254 (frenet_pose dropped when compute_frenet_frame=False) and :262-265
   (scan is shape (0,) with the LiDAR off). NEW from CLAUDE.md:203 — FRENET_DYNAMIC_STATE contains
   no Frenet field; the name refers to splitting velocity into body-frame vx/vy.

2. **Base fields** `[code]`
   *Reader can:* The reader can decode every field read straight from the simulator buffers,
   including the two whose value at reset is not what the name suggests.
   *Source:* observations.rst:102-143 (eight-row table, index-6-is-beta note at :119-121 kept).
   The ``scan`` row absorbs the fourth terminal warning :267-270 as one clause + :doc:`quickstart`
   (canonical home). The ``frenet_pose`` row absorbs :256-260 as one clause + :doc:`tracks`
   (canonical home). Doctest prints the four shapes — (1080,) (7,) (7,) () — and the count of non-
   zero scan entries at reset.

3. **Derived fields** `[code]`
   *Reader can:* The reader can read named scalars and knows which three are identically zero
   under KS.
   *Source:* observations.rst:145-173 (nine-row table with the vx/vy formulas at :148-149). The
   :175-180 pose-frame note collapses to one clause on the pose_x/pose_y row + :doc:`dynamics`
   (canonical home). Doctest under KINEMATIC_STATE prints linear_vel_x = 1.8179682 after 50 steps
   of a 2 m/s command — the observable is that the P controller has not converged, not that the
   field exists.

4. **Dtype and shape contract** `[code]`
   *Reader can:* The reader can store an observation in a replay buffer and index it in numba
   without either aliasing the live buffer or hitting a dtype promotion.
   *Source:* observations.rst:22-25 moved down out of the opener; NEW from CLAUDE.md:205 (0-d
   float32 ndarrays, never Python floats, load-bearing for the numba np.dot in waypoint_follow.py)
   and CLAUDE.md:207 (scan is copied, not aliased — state.scans is overwritten in place every
   step). Placed after the vocabulary because the contract is meaningless until the reader knows
   which fields it governs, and needed immediately before the flatten code in the next section.

5. **The observation space** `[code]`
   *Reader can:* The reader can flatten and normalise the space, index the flat vector correctly,
   and knows the one config change that invalidates the declared bounds.
   *Source:* observations.rst:213-242, with the :218 hedge ('All bounds are finite and roughly
   physical') replaced by the measured answer from CLAUDE.md:209 — delta ±0.4509 and linear_vel_x
   [-5.0951, 20.0951] carry one integrator step of actuator overshoot. Doctest prints the verified
   Dict repr. NEW from CLAUDE.md:215 — observe() order is preset order but the space is sorted
   lexicographically, so KINEMATIC_STATE flattens to [delta, linear_vel_x, pose_theta, pose_x,
   pose_y] and agent_10 sorts before agent_2. NEW from CLAUDE.md:442 — update_params and every
   domain-randomization reset leave observation_space stale; the page's single warning sits here.

6. **The info dict** `[code]`
   *Reader can:* The reader can read info without a first-frame KeyError and knows that its clock
   and the observation's disagree.
   *Source:* NEW from CLAUDE.md:444 (reset returns 3 keys, step returns 5 — verified: reset
   ['lap_counts','lap_times','sim_time'], step adds 'collisions','progress') and CLAUDE.md:443
   (obs['sim_time'] lags info['sim_time'] by exactly one timestep because observe() runs before
   self.sim_time is refreshed). Last because info is the other half of the step return and only
   matters once the obs vocabulary is settled.

   **Cut:**
   - observations.rst:244-271 — the entire 'Gotchas' heading and its four consecutive .. warning::
   directives with no prose between them; each caveat is redistributed to the table row that
   provokes it (house rule 5, and the Batch F automatic-reject rule)
   - observations.rst:6-8 — the prose table of contents ('This page documents the six presets, the
   field vocabulary...'), which restates the sidebar
   - observations.rst:10-12 — the two forward :doc: pointers before any content; they belong
   inline where the reader needs them
   - observations.rst:84 — 'The default config uses DIRECT.', a bare restatement of the table row
   above it
   - observations.rst:182-188 — the standalone 'Custom feature subsets (FEATURES)' heading and its
   18-line block that prints only sorted(keys); folds into the FEATURES table row plus one doctest
   line
   - observations.rst:208-211 — the note that features may only be set with type=FEATURES; the
   validation rule's canonical home is docs/configuration.rst, one clause remains on the FEATURES
   row
   - observations.rst:175-180 — the KS/ST pose-frame note; canonical home is docs/dynamics.rst, at
   the point of model choice
   - observations.rst:224-228 — 'Because the space is finite (not the old blanket ±1e30)' — repo
   archaeology (house rule 12); the finiteness survives, the comparison to a removed
   implementation does not
   - The bold spans at observations.rst:22 and :218 — the page carries 2 bold runs in 270 lines
   today and must not gain any as it shrinks to 200

### `docs/actions.rst` — Actions

**Mode:** reference · **Target:** ~130 lines

**Job:** What the two action columns mean under each mode, and what the environment does with a value it never validates.

**Opening lines (draft):**

> Every ``step`` consumes one ``(num_agents, 2)`` float32 array with steering in column 0, and the
> simulator checks only its shape. An action outside ``action_space`` is executed rather than
> rejected or clipped:

1. **Column meanings by mode** `[code]`
   *Reader can:* The reader can write a correct command pair under any of the four mode
   combinations and read its bounds off the env rather than hardcoding them.
   *Source:* Fuses today's twin sections into one four-row table keyed by (column, mode):
   actions.rst:44-56 (longitudinal) + :80-92 (steering), inheriting both bounds columns. The two
   hardcoded-number notes at :66-72 and :106-110 collapse into that bounds column and into one
   doctest printing the verified ACCL+STEERING_SPEED box, Box([[-3.2 -9.51]], [[3.2 9.51]], (1,
   2), float32). Absorbs :139-161 ('Reading the action space') and its action_space.sample() line,
   and :163-168 (bounds identical across rows). Carries the page's first warning immediately after
   the table: a transposed pair whose two values both lie in range is indistinguishable from a
   correct one, because both columns are float32 with overlapping valid ranges and only the shape
   is checked (CLAUDE.md:184) — the canonical explanation of why the swap cannot be caught.

2. **What each mode does to the command** `[code]`
   *Reader can:* The reader can predict the actuator response — a rate that slams to its limit,
   and a gain that jumps 5x the instant speed leaves zero — instead of assuming two PID
   controllers.
   *Source:* actions.rst:58-64 (pid_accl, a real P controller with four gain quadrants) and
   :103-104 (STEERING_SPEED identity); the protected :94-101 pid_steer bang-bang warning kept
   verbatim as the page's second warning. NEW from CLAUDE.md:448 — pid_accl's quadrant test is
   ``current_speed > 0.0``, so pid_accl(5, 0.0) = 4.755 against pid_accl(5, 0.001) = 23.77, and
   every reset spawns at v=0, so the first step of every episode launches on the reverse branch.
   That doctest is what makes 'four gain quadrants' observable instead of asserted. Ordered second
   because the transform cannot be interpreted before the reader knows which column and mode it
   reads.

3. **Multi-agent action arrays** `[code]`
   *Reader can:* The reader can drive N agents, knows a row maps to an agent index, and knows the
   wrapper that removes the leading axis does not clip.
   *Source:* Collapses actions.rst:198-230 to the array literal alone plus a doctest printing
   action_space.shape (2, 2) and its two identical rows — house rule 9, since :198-230 repeats
   :170-196 verbatim but for num_agents=2. Absorbs :232-237 as one clause: SingleAgentWrapper
   reshapes (2,) to (1,2) and does not clip (CLAUDE.md:355), linking to :doc:`rl`.

   **Cut:**
   - actions.rst:239-247 — the 'See also' closer, four links already present inline (house rule 7)
   - actions.rst:4-10 — the prose table of contents ('This page explains the fixed shape of that
   array, what each column means, the two longitudinal and two steering interpretation modes...')
   plus the two forward :doc: pointers; the Batch F diagnosis names this exact sentence pattern as
   tell #1
   - actions.rst:30-36 — the standalone steering-is-column-0 warning; its claim is promoted to the
   page's opening thesis and its mechanism sharpened into the one warning under the mode table, so
   the box itself is redundant
   - actions.rst:112-137 — the 'Selecting the modes' section; the nested with_updates pattern's
   canonical home is docs/configuration.rst, and section 1's doctest already constructs
   ControlConfig inline. One clause + :doc:`configuration` survives
   - actions.rst:134-137 — the sim2real knobs paragraph; ControlConfig's noise and delay fields
   belong to docs/configuration.rst (fields) and the new docs/sim2real.rst (semantics)
   - actions.rst:170-196 — the standalone single-agent example; the opener's loop already runs 100
   steps and prints a speed, so a second silent loop adds nothing
   - actions.rst:15-21 — the bulleted restatement of the two columns immediately after the
   sentence that states them
   - actions.rst:22-28 — the orphan two-line code block with no printed output (violates rule 4);
   its content is the opener's array literal

### `docs/api/index.rst` — API Reference

**Mode:** reference · **Target:** ~175 lines

**Job:** Which public symbol lives on which import path, and what each one is in one line.

**Opening lines (draft):**

> Every public symbol is documented on the package path it imports from, so ``from
> f1tenth_gym.envs.lidar import LiDARConfig`` and :class:`~f1tenth_gym.envs.lidar.LiDARConfig`
> name one object: >>> import f1tenth_gym >>> import gymnasium as gym >>>
> gym.spec("f1tenth-v0").entry_point 'f1tenth_gym.envs:F110Env' >>> from f1tenth_gym.envs.lidar
> import LiDARConfig >>> LiDARConfig().num_beams 1080 Importing ``f1tenth_gym`` contributes the
> ``f1tenth-v0`` registration and nothing else; the 1080-beam default and every symbol below live
> under ``f1tenth_gym.envs``.

1. **Environment and configuration**
   *Reader can:* After this group the reader can construct an env and change any top-level
   default, because F110Env cannot be built without an EnvConfig instance and the two are
   inseparable.
   *Source:* autosummary of f1tenth_gym.envs.f110_env.F110Env; envs.env_config.{EnvConfig,
   SimulationConfig, ControlConfig, ObservationConfig, ResetConfig, TerminationConfig,
   LoopCounterMode}. Inherits the intent of today's :8-15 heading, discards its automodule bodies.
   RenderConfig/RewardConfig/DomainRandomizationConfig deliberately move to their subsystem groups
   so each symbol owns exactly one autosummary stub.

2. **Observations**
   *Reader can:* After this group the reader can name every field the env returns and pick or
   build a preset, which is required before an action can be chosen.
   *Source:* autosummary of f1tenth_gym.envs.observation.{ObservationType, ALL_FEATURES,
   observation_factory, FullObservation, Observation, scan_space}. Replaces the observation.full
   module path at :36 with the package path every guide example already imports from (nine :doc:
   refs in docs/*.rst point at envs.observation).

3. **Actions**
   *Reader can:* After this group the reader can build the (num_agents, 2) array step consumes and
   look up which transform sets its bounds.
   *Source:* autosummary of f1tenth_gym.envs.action.{SteerActionType, LongitudinalActionType,
   get_action_space, from_single_to_multi_action_space, accl_action, speed_action,
   steering_angle_action, steering_speed_action}. Inherits :31-34. The four transform functions
   are added because actions.rst:64 and :104 already :func: them and today resolve to nothing.

4. **Vehicle dynamics and integration**
   *Reader can:* After this group the reader can swap the model, override a vehicle parameter, and
   change the integration scheme — the first three things a config points at.
   *Source:* autosummary of f1tenth_gym.envs.dynamic_models.{DynamicModel, VehicleParameters,
   F1TENTH_VEHICLE_PARAMETERS, F1FIFTH_VEHICLE_PARAMETERS, FULLSCALE_VEHICLE_PARAMETERS,
   pid_steer, pid_accl}; envs.integrators.{IntegratorType, integrator_from_type}. Inherits :26-29
   and :24. pid_steer/pid_accl are documented at the package path (their __all__ home), which
   requires migrating actions.rst:59 and :97 off the dynamic_models.utils path — verified that
   function xrefs, unlike class xrefs, get no :canonical: alias.

5. **Track, reference lines and spawning**
   *Reader can:* After this group the reader can load or synthesize a track, convert to Frenet,
   and choose where cars are placed on it — reset functions bind to track.raceline, so they are
   the same subject.
   *Source:* autosummary of f1tenth_gym.envs.track.{Track, TrackSpec, Raceline, find_track_dir};
   envs.track.cubic_spline.CubicSplineND; envs.reset.{ResetStrategy, ResetFn, make_reset_fn,
   GridResetFn, AllTrackResetFn, AllMapResetFn}. Merges today's :47-59. CubicSplineND stays on its
   module path because envs.track does not re-export it (verified AttributeError). Requires
   migrating the three method xrefs at tracks.rst:13, :14, :129 off track.track.Track.

6. **LiDAR and collision**
   *Reader can:* After this group the reader can change what the car sees and read the exact
   function that decides a crash, the last subsystem the step loop touches.
   *Source:* autosummary of f1tenth_gym.envs.lidar.{LiDARConfig, ScanSimulator2D, check_collision,
   ray_cast}; envs.collision_models.{CollisionCheckMode, collision, collision_multiple,
   get_vertices}. Inherits :38-45. Documenting the lidar package rather than lidar.laser_models
   structurally drops ScanTests, main and the nine internal ray kernels — no :exclude-members:
   needed, verified they are absent from the package namespace.

7. **RL and sim2real**
   *Reader can:* After this group the reader can wrap the env for a single-agent learner, swap the
   reward, and randomize the vehicle per episode.
   *Source:* autosummary of f1tenth_gym.envs.wrappers.{SingleAgentWrapper,
   ObservationDelayWrapper}; envs.env_config.{RewardConfig, RewardMode,
   DomainRandomizationConfig}. NEW group; wrappers previously sat at :15 under "Environment &
   configuration" and the three RL config classes were undocumented. Mirrors the split of
   rewards_and_rl.rst into docs/rl.rst and docs/sim2real.rst.

8. **Rendering**
   *Reader can:* After this group the reader can subclass a renderer or attach an overlay
   callback, which only matters once something is already moving.
   *Source:* autosummary of f1tenth_gym.envs.env_config.RenderConfig; envs.f110_env.RenderClock;
   envs.rendering.{make_renderer, EnvRenderer, ObjectRenderer, make_lidar_scan_callback}. Inherits
   :61-66 but on the package path, so the rendering.callbacks xref in rendering.rst migrates.
   RenderClock is added because RenderConfig's docstring points at it by name. The RenderConfig
   field table stays in docs/configuration.rst; this is the symbol entry only.

9. **Simulation internals** `[code]`
   *Reader can:* After this group the reader can read the struct-of-arrays buffers directly, the
   escape hatch for when the config surface has run out.
   *Source:* autosummary of f1tenth_gym.envs.simulator.F110Simulator; envs.state.SimulationState.
   Inherits :17-22 minus integrators, which move to the dynamics group. Carries the page's second
   doctest because env.unwrapped.sim is the access path and it is documented nowhere else:
   reset(seed=42) then sim.state.poses.shape -> (1, 3) and sim.params_array[0] -> 1.0489
   (verified). ScanCache stays undocumented.

   **Cut:**
   - docs/api/index.rst:4-6 — "this page is the exhaustive symbol reference" is false (#23) and
   "For a task-oriented introduction start with :doc:`../configuration`" is a See-also closer
   hoisted to the top; both violate house rules 1 and 7. Replaced by the import-path doctest.
   - docs/api/index.rst:11-66 — all 18 bare `.. automodule::` directives. They produce zero one-
   line summaries (the defect #23 names) and publish every module member indiscriminately.
   Replaced by nine curated `.. autosummary:: :toctree: generated :nosignatures:` tables over 63
   symbols.
   - docs/api/index.rst:43 — `automodule:: f1tenth_gym.envs.lidar.laser_models`. This is the
   vector that publishes ScanTests(unittest.TestCase) (laser_models.py:588) and module-level
   main(); it also publishes get_dt, xy_2_rc, distance_transform, trace_ray, get_scan, cross,
   are_collinear, get_range and get_blocked_view_indices, none of which the guide teaches. Not
   replaced.
   - docs/api/index.rst:41, :36, :50, :52, :64, :66 — the six module paths (lidar.config,
   observation.full, track.track, track.raceline, rendering.renderer, rendering.callbacks) that
   differ from the paths examples/ and docs/ import from. Replaced by the four package paths
   envs.lidar, envs.observation, envs.track, envs.rendering. Verified: for classes autodoc emits
   :canonical:, so both the package and module path resolve and no existing class xref breaks.
   - docs/conf.py:43 — `"undoc-members": True`. 119 of 272 public symbols have no docstring; the
   flag renders them as bare signatures that make the tree look complete when it is not. Off
   globally, with no per-directive re-enable: verified that VehicleParameters keeps its full
   88-field list anyway via autodoc_typehints="description" on the dataclass __init__, so the one
   symbol at risk does not regress.
   - Never-documented modules, excluded by curation rather than by directive:
   dynamic_models.kinematic, .single_track, .multi_body, .multi_body.tire_model, .utils internals
   (upper_accel_limit, accl_constraints, steering_constraint); reset.utils, reset.masked_reset,
   reset.map_reset module paths; rendering.pyqtgl_objects, rendering.rendering_pyqtgl;
   track.utils.nearest_point_on_trajectory; simulator.ScanCache; observation.base module path;
   f1tenth_gym.envs.f110_env.RenderClock is the sole f110_env symbol besides F110Env kept.
