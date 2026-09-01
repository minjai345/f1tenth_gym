JAX training and throughput measurements
========================================

The functional JAX simulator's historical Phase 6 migration used two separate
forms of evidence.  A complete native PPO job demonstrated that rollout, reset,
terminal targets and an optimizer could remain in one JAX program.  A rollout
harness compared the functional batch with the mutable Gymnasium reference on
equivalent physics.  The native PPO CPU gate and the full-rollout CPU/RTX 3080
gates were rerun successfully immediately before their removal; they are not
shipped interfaces or current commands.  Neither result predicts the
Python-facing ``JaxF110Env`` adapter, which transfers observations to NumPy on
every step.

Measurement protocol
--------------------

The measurements below were taken on 2026-08-31 with JAX 0.11.1, float32
production state and Python 3.13.11.  The CPU is an Intel Core i9-11980HK with
8 cores, 16 threads and 62 GiB of RAM.  The accelerator is an NVIDIA GeForce
RTX 3080 Laptop GPU with 16 GiB; CPU and GPU results are reported separately.

The rollout harness constructed the same track, poses, controls, actions,
model, integrator and episode settings for both implementations.
State-only work uses a wall-free synthetic circuit because sensing and contact
are disabled.  LiDAR, contact and full transitions use a deterministic annular
road whose sub-pixel extraction produces 154 wall segments.  LiDAR starts on
the centerline, so beams intersect real walls.  Contact and full workloads
place every body 4 cm into the outer wall; resting slop keeps every agent in
contact for every measured transition.

The schema required zero colliding-agent steps for state/LiDAR and exactly
``batch * agents * rollout_length`` for contact/full.  It rejected an empty or
partly inactive contact workload.  Both implementations consumed every numeric
and boolean element of every transition in a checksum.  Construction and reset
were outside the steady timer, JAX compilation was separate, and every warm-up
and measured call was synchronized.  Mutable environments reset before each
run.  Each row is the median of three runs after one warm-up:

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 42

   * - Workload
     - Rollout
     - Track points
     - Initial condition
   * - State
     - 96 steps
     - 96
     - centerline, walls disabled
   * - LiDAR, 64 beams
     - 96 steps
     - 96
     - centerline on the annular road
   * - LiDAR, 1080 beams
     - 64 steps
     - 96
     - centerline on the annular road
   * - Contact
     - 64 steps
     - 96
     - persistent shallow wall contact
   * - Full
     - 32 steps
     - 96
     - persistent contact plus 1080 beams

CPU runs forced the CPU JAX platform and compared both implementations.  GPU
runs exercised only the functional JAX implementation and used one batch size
per process so the allocator's process-lifetime peak was not inherited from a
previous compilation.  Indexed rows used translated equal-shape maps, while
the reference-table experiments varied their track-point count.  The rates are
measurements on this machine, not API guarantees.

Native PPO measurement
----------------------

The validation policy observes steering angle and speed, emits a joint latent
Gaussian action followed by ``tanh``, and tracks a 3 m/s target with direct
steering-rate and acceleration control.  It is deliberately a small validation
task rather than a racing benchmark.  The recorded job used 64 environments,
64 steps per rollout and 24 updates, or 98,304 environment transitions.

On the reviewed CPU, deterministic fixed-key evaluation changed as follows:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Measurement
     - Before
     - After
     - Required
   * - Mean bounded objective
     - 0.000000
     - 0.924272
     - improvement at least 0.10
   * - Mean speed
     - 0.000000 m/s
     - 2.964633 m/s
     - target 3 m/s

The first rollout/update compilation and execution took 1.62 s.  Later updates
sustained about 49,229 environment-steps/s including PPO optimization, and
every parameter, optimizer, loss and reported metric remained finite.  The
migration-time mathematical checks separately verified that natural
termination zero-bootstraps, timeout truncation bootstraps the terminal
transition observation, and both stop GAE recurrence across auto-reset.

The same fixed job passed on the RTX 3080.  Its objective improved from
0.000000 to 0.929017, final mean speed was 2.881180 m/s, the first compiled
update took 5.66 s and later updates sustained about 29,163
environment-steps/s.  This small policy/update program is faster on the
reviewed CPU; the result is a correctness gate, not evidence that every native
training shape should be placed on the GPU.

The separate Gymnasium ecosystem gate also passed in an ephemeral environment:
Stable-Baselines3 2.9.0 accepted the wrapped ``JaxF110Env``, and SBX 0.28.0
completed one eight-step PPO update and returned a valid action and transition
with JAX 0.11.1.  The trainer stack is deliberately absent from the package
dependencies and committed lock.

CPU rollout results
-------------------

Euler integration uses a 0.01 s step and Frenet projection remains enabled.
Rates are environment-steps/s; two-agent rows have twice the listed rate in
agent-steps/s.  A dash means the large-batch mutable comparison was not run.

.. list-table::
   :header-rows: 1
   :widths: 31 7 15 15 12 12

   * - Workload
     - Batch
     - JAX
     - Mutable
     - Speedup
     - Compile
   * - KS state, 1 agent, P96
     - 1
     - 128,679
     - 10,101
     - 12.7x
     - 0.417 s
   * - KS state, 1 agent, P96
     - 16
     - 800,971
     - 10,127
     - 79.1x
     - 0.420 s
   * - KS state, 1 agent, P96
     - 64
     - 687,920
     - 10,148
     - 67.8x
     - 0.450 s
   * - KS state, 1 agent, P96
     - 256
     - 677,320
     - —
     - —
     - 0.621 s
   * - KS state, 1 agent, P96
     - 1024
     - 762,746
     - —
     - —
     - 0.683 s
   * - KS state, 2 agents, P96
     - 64
     - 274,967
     - 6,691
     - 41.1x
     - 0.437 s
   * - KS LiDAR, 64 beams
     - 64
     - 128,426
     - 5,144
     - 25.0x
     - 0.775 s
   * - KS LiDAR, 1080 beams
     - 1
     - 3,679
     - 3,634
     - 1.0x
     - 0.628 s
   * - KS LiDAR, 1080 beams
     - 16
     - 9,166
     - 3,061
     - 3.0x
     - 0.894 s
   * - KS LiDAR, 1080 beams
     - 64
     - 10,715
     - 3,583
     - 3.0x
     - 0.762 s
   * - ST persistent contact, 2 agents
     - 1
     - 12,748
     - 2,124
     - 6.0x
     - 1.406 s
   * - ST persistent contact, 2 agents
     - 16
     - 31,127
     - 2,086
     - 14.9x
     - 1.517 s
   * - ST persistent contact, 2 agents
     - 48
     - 36,038
     - 2,074
     - 17.4x
     - 1.770 s
   * - ST persistent contact, 2 agents
     - 64
     - 38,001
     - 2,081
     - 18.3x
     - 1.518 s
   * - ST persistent contact, 2 agents
     - 96
     - 38,564
     - 1,990
     - 19.4x
     - 1.439 s
   * - ST persistent contact, 2 agents
     - 256
     - 16,793
     - 1,405
     - 11.9x
     - 1.460 s
   * - ST full, 2 agents, 1080 beams
     - 1
     - 3,561
     - 1,192
     - 3.0x
     - 1.620 s
   * - ST full, 2 agents, 1080 beams
     - 8
     - 4,817
     - 987
     - 4.9x
     - 1.578 s
   * - ST full, 2 agents, 1080 beams
     - 16
     - 4,493
     - 1,136
     - 4.0x
     - 1.770 s
   * - ST full, 2 agents, 1080 beams
     - 64
     - 3,832
     - 970
     - 4.0x
     - 1.957 s
   * - KS state, 4 indexed maps, P96
     - 64
     - 607,768
     - 10,225
     - 59.4x
     - 0.506 s
   * - KS state, 1 agent, P384
     - 16
     - 177,606
     - 9,481
     - 18.7x
     - 0.452 s

State-only KS peaks at batch 16 on this CPU.  Persistent contact peaks at batch
96, while the combined transition peaks at batch 8; larger batches can reduce
CPU throughput even though they expose more parallel rows.  Increasing the
reference table from P96 to P384 cuts batch-16 state throughput from 800,971 to
177,606 steps/s.  These results justify exact-shape buckets and measured batch
sizes, not one global default.

The resident P96 state table occupies 30,924 bytes.  The annular LiDAR,
contact and full tables occupy 64,066, 109,786 and 136,961 bytes respectively.
Four indexed P96 maps occupy exactly 123,696 bytes; P384 occupies 132,348
bytes.  The CPU backend exposes no allocator peak, so JSON reports it as
unavailable instead of substituting process RSS or an array-size estimate.

GPU rollout results
-------------------

The RTX 3080 was measured with JAX x64 disabled and ``/dev/nvidia*`` visible.
An execution sandbox can hide those device nodes even when the host GPU and
driver are healthy.  ``Peak`` is the allocator's process-lifetime high-water
mark for an isolated batch, not the live size of one transition.

.. list-table::
   :header-rows: 1
   :widths: 31 7 15 15 12 12

   * - Workload
     - Batch
     - Env steps/s
     - Agent steps/s
     - Peak
     - Compile
   * - KS state, 1 agent, P96
     - 1
     - 2,931
     - 2,931
     - 0.1 MiB
     - 0.786 s
   * - KS state, 1 agent, P96
     - 64
     - 125,432
     - 125,432
     - 96.1 MiB
     - 1.200 s
   * - KS state, 1 agent, P96
     - 256
     - 613,162
     - 613,162
     - 96.4 MiB
     - 1.597 s
   * - KS state, 1 agent, P96
     - 1024
     - 2,034,346
     - 2,034,346
     - 97.2 MiB
     - 1.111 s
   * - KS LiDAR, 64 beams
     - 64
     - 89,887
     - 89,887
     - 144.5 MiB
     - 2.076 s
   * - KS LiDAR, 1080 beams
     - 1
     - 1,945
     - 1,945
     - 112.1 MiB
     - 1.450 s
   * - KS LiDAR, 1080 beams
     - 16
     - 20,900
     - 20,900
     - 144.6 MiB
     - 1.620 s
   * - KS LiDAR, 1080 beams
     - 64
     - 67,296
     - 67,296
     - 146.2 MiB
     - 2.188 s
   * - ST persistent contact, 2 agents
     - 1
     - 695
     - 1,390
     - 80.2 MiB
     - 2.324 s
   * - ST persistent contact, 2 agents
     - 16
     - 10,771
     - 21,541
     - 80.2 MiB
     - 2.510 s
   * - ST persistent contact, 2 agents
     - 48
     - 29,903
     - 59,807
     - 96.2 MiB
     - 2.563 s
   * - ST persistent contact, 2 agents
     - 64
     - 40,902
     - 81,804
     - 96.3 MiB
     - 2.258 s
   * - ST persistent contact, 2 agents
     - 96
     - 59,674
     - 119,349
     - 96.4 MiB
     - 2.531 s
   * - ST persistent contact, 2 agents
     - 256
     - 158,576
     - 317,153
     - 96.7 MiB
     - 5.697 s
   * - ST full, 2 agents, 1080 beams
     - 1
     - 655
     - 1,310
     - 112.3 MiB
     - 3.367 s
   * - ST full, 2 agents, 1080 beams
     - 8
     - 4,766
     - 9,531
     - 144.7 MiB
     - 5.595 s
   * - ST full, 2 agents, 1080 beams
     - 16
     - 9,856
     - 19,713
     - 145.2 MiB
     - 3.741 s
   * - ST full, 2 agents, 1080 beams
     - 64
     - 34,302
     - 68,605
     - 148.3 MiB
     - 4.242 s
   * - KS state, 4 indexed maps, P96
     - 64
     - 141,197
     - 141,197
     - 96.3 MiB
     - 1.324 s
   * - KS state, 4 indexed maps, P96
     - 256
     - 358,392
     - 358,392
     - 96.9 MiB
     - 1.994 s

Placement depends on the whole workload.  State-only CPU remains faster through
batch 256, while the GPU is 2.67x faster at batch 1024.  With 64 beams the CPU
still leads at batch 64.  With 1080 beams the CPU leads at batch 1, but the GPU
is 2.28x faster at batch 16 and 6.28x faster at batch 64.  The full transition
is effectively tied at batch 8, then favors the GPU by 2.19x at batch 16 and
8.95x at batch 64.

The largest isolated allocator peak was 148.3 MiB, with no OOM on the 16 GiB
device.  That figure covers the simulator benchmark only; a training program
must also budget policy, optimizer and rollout storage.

Contact placement decision
--------------------------

Persistent shallow contact makes the solver live for every agent-step.  The
final ST/two-agent comparison locates the hardware crossover rather than
extrapolating the old contact microbenchmark:

.. list-table::
   :header-rows: 1
   :widths: 18 27 27 28

   * - Batch
     - CPU env steps/s
     - GPU env steps/s
     - GPU / CPU
   * - 1
     - 12,748
     - 695
     - 0.05x
   * - 8
     - 27,207
     - 5,428
     - 0.20x
   * - 16
     - 31,127
     - 10,771
     - 0.35x
   * - 32
     - 35,342
     - 21,321
     - 0.60x
   * - 48
     - 36,038
     - 29,903
     - 0.83x
   * - 64
     - 38,001
     - 40,902
     - 1.08x
   * - 96
     - 38,564
     - 59,674
     - 1.55x
   * - 128
     - 26,686
     - 80,490
     - 3.02x
   * - 256
     - 16,793
     - 158,576
     - 9.44x

The crossover on this machine lies between batches 48 and 64, or 96 and 128
simultaneously contacting bodies.  The public mutable and single-environment
defaults therefore remain on CPU.  Wide native batches may explicitly select
the GPU, but LiDAR, policy networks, optimizer state and different contact
tables can move the crossover; the repository does not encode 64 as a universal
threshold.
