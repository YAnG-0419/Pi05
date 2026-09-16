# FR3 + Wuji GR00T deployment

完整的工控机单机部署步骤见 [MIGRATION.md](MIGRATION.md)。

This directory adds GR00T as a separate inference backend. It does not replace
or edit `deploy/fr3_wuji_fast`, so the existing Pi05 commands and checkpoint
remain unchanged.

## Process boundary

There are two processes with separate Python environments:

1. `serve.sh` runs the GR00T model in the Isaac-GR00T Python 3.12 environment.
   It has GPU/checkpoint access and no robot access.
2. `run.sh` runs observation, admission and 100 Hz device control in the existing
   Pi05 environment. It talks to the model server over a bounded HTTP/NPZ API.

The wire format is fixed to 16 x 54 absolute joint targets at 30 Hz. The GR00T
joint order is converted inside the model process to the existing physical order:

`left arm 7, left hand 20, right arm 7, right hand 20`

The initial implementation is deliberately serial: it executes all 16 policy
steps, holds the final target while requesting the next fresh observation, then
smoothly starts the next chunk. It does not introduce unverified receding-horizon
blending into the Pi05 controller.

## 1. Verify the checkpoint without GPU or hardware

```bash
GROOT_ROOT=/home/descfly/workspace/Isaac-GR00T \
bash deploy/fr3_wuji_groot/serve.sh \
  --checkpoint /home/descfly/workspace/Isaac-GR00T/outputs/dual_franka_wuji/newdata_v1_scratch30k_20260916/best_checkpoint \
  --check-only
```

For the currently verified checkpoint, the complete model manifest is:

```text
cf9fd5ab49b8dab8f27bccd02bfe280f8cd0914247c3a5551754adcefd66889c
```

This fingerprint covers configuration, embodiment, processor, statistics,
inference recipe, reload verification and every safetensors file. Re-run the
check and use the newly printed value whenever any checkpoint file changes.

## 2. Preview the robot-side launch without sockets or devices

```bash
bash deploy/fr3_wuji_groot/run.sh --check \
  --output /home/descfly/workspace/Pi05/logs/weight_motion_eval/groot-check
```

The default mode is also `--check`. This creates only runtime and command-preview
files. It does not connect to the model, open device sockets or launch a driver.

## 3. Start and verify the inference service

In the remote GPU terminal:

```bash
GROOT_ROOT=/home/descfly/workspace/Isaac-GR00T \
bash deploy/fr3_wuji_groot/serve.sh \
  --checkpoint /home/descfly/workspace/Isaac-GR00T/outputs/dual_franka_wuji/newdata_v1_scratch30k_20260916/best_checkpoint \
  --host 127.0.0.1 --port 8766
```

Wait for `GR00T_POLICY_READY`. In another terminal, validate the exact warmed-up
model without touching hardware:

```bash
bash deploy/fr3_wuji_groot/run.sh --server-check \
  --server http://127.0.0.1:8766 \
  --expected-model-sha256 cf9fd5ab49b8dab8f27bccd02bfe280f8cd0914247c3a5551754adcefd66889c
```

For a remote inference host, bind the server to an appropriate private interface
and pass that host to `--server`. Do not expose this unauthenticated first-stage
service to a public network.

## 4. Supervised one-chunk hardware trial

Only after the server check, hardware preflight and physical stop preparation:

```bash
bash deploy/fr3_wuji_groot/run.sh --execute --supervised-trial \
  --server http://127.0.0.1:8766 \
  --expected-model-sha256 cf9fd5ab49b8dab8f27bccd02bfe280f8cd0914247c3a5551754adcefd66889c \
  --rounds 1 --start-cameras --finish-policy disable
```

Real execution cannot start without both `--execute` and either
`--supervised-trial` or an existing `--qualification` file. The first trial
defaults to one 16-step chunk and disables output after measured settling.

## Existing Pi05 path

The old entry points are unchanged. For example:

```bash
bash deploy/fr3_wuji_fast/run.sh --check
```

Keep the robot runtime native on the real-time host. The GR00T model service is
the container boundary: it needs only GPU, read-only model mounts and its HTTP
port, while hardware ownership stays in Pi05. Use `MIGRATION.md` to build,
export and install the dedicated inference image.
