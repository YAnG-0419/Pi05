# Wuji upstream reference

This is an FR3/Wuji adapter, not an unmodified Wuji release or a new model.

- Repository: https://github.com/wuji-technology/wuji-openpi
- Reviewed local checkout commit: `be6b32540546fb4fdd274f5cc86a7f9a67148a4c`
- Source: `packages/openpi-client/src/openpi_client/rtg_action_broker.py`
- Reviewed source SHA-256: `67df69f19acfb7644f1350ce73196237742aa3e032fb39fb813a984ee80e27df`
- License: Apache-2.0; the complete upstream license is in `UPSTREAM_LICENSE`.

The cubic Hermite calculation in `timeline.smooth_prefix` adapts the upstream
`cubic_smooth_prefix` function. The orchestration is implemented here so the
existing FR3/Wuji device process retains its ownership, deadlines and gateway.
No runtime dependency on the separate wuji-openpi checkout is added.

Preserved concepts: 50-step policy chunk, 30 Hz consumption, half-chunk asynchronous
inference, latency compensation, three-node cubic prefix guidance; serial mode
is also available. RTG may replace a chunk before all 50 nodes are consumed.

Intentional adaptations:

- The first target uses the existing controlled acquisition, once per session.
- A separate process interpolates 30 Hz nodes linearly into 100 Hz device frames.
- The RTG offset uses actual elapsed monotonic time since request submission,
  rounded up to the next policy node; it is not an inference-frequency setting.
- Cubic guidance begins at the actual splice offset and anchors the current
  commanded position. The upstream cubic broker smooths the prefix before
  skipping latency steps, which can skip the smoothed part altogether.
- First and spliced arm targets are projected onto the session-specific 1.0 rad/s
  reachable set on the 30 Hz grid (shared slow callers keep their 0.7 rad/s
  default). The original prediction remains in the run record.
- A missing model/observation has a bounded 1.5-second wait from the trigger.
  Predictions must also pass the original 70 ms observation-send budget and
  1 s observation-to-admission deadline (checked again at live splice).
- While a late inference is pending, only the remainder of the accepted chunk
  and then its final pose are issued. There is no extrapolation or stale replay.
- The bridge receives one stream identity with monotonically increasing frame
  sequences. Each new raw prediction has its own recorded request ID and hash;
  the 100 Hz controller validates it before changing the active nodes. The
  existing bridge's plan hash identifies this stream, not a precomputed spline.
- The existing slider hand projection, rate limiter, contact handling and
  device operating parameters are inherited; actual hand motion can lag targets.

This adaptation does not implement QP-RTG or model-side real-time chunking (RTC).
