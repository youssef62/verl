"""Async poller for vLLM Prometheus /metrics endpoint.

Polls each vLLM replica's HTTP /metrics endpoint on a configurable interval,
aggregates results across replicas, and logs to wandb on its own time-based
x-axis (``vllm/poll_elapsed_s``) — decoupled from training step cadence.

The x-axis is registered via ``wandb.define_metric`` so all ``vllm/*`` keys
appear in their own section with wall-clock time, matching wandb's built-in
System metrics panel behaviour.

Usage in a trainer::

    from verl.experimental.fully_async_policy.vllm_metrics_poller import VllmMetricsPoller

    poller = VllmMetricsPoller(server_addresses=["10.0.0.1:8080"], interval=5.0)
    poller.start()
    ...
    poller.stop()

Requires ``actor_rollout_ref.rollout.disable_log_stats=False`` in the run script
so that vLLM actually collects and serves stats at /metrics.
"""

import asyncio
import logging
import threading
import time

logger = logging.getLogger(__name__)


# Prometheus metric names exposed by vLLM v1 that we want to track.
# Verified against vllm/v1/metrics/loggers.py::PrometheusStatLogger.
# Extend this set to pull in additional metrics.
_WATCH_METRICS: set[str] = {
    # Scheduler state (gauges)
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",                  # was gpu_cache_usage_perc in vLLM v0
    # Counters
    "vllm:num_preemptions",                       # was num_preemptions_total in vLLM v0
    "vllm:generation_tokens",                     # was num_generation_tokens_total in vLLM v0
    "vllm:prompt_tokens",                         # was num_prompt_tokens_total in vLLM v0
    # Histogram _sum/_count (used to compute means)
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:inter_token_latency_seconds_sum",       # was time_per_output_token_seconds in vLLM v0
    "vllm:inter_token_latency_seconds_count",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
}

# wandb key prefix for all vLLM metrics
_WANDB_PREFIX = "vllm"

# The custom x-axis metric used for define_metric
_STEP_METRIC = "vllm/poll_elapsed_s"


def parse_prometheus_text(text: str, watch: set[str] | None = None) -> dict[str, float]:
    """Parse Prometheus text exposition format into a flat ``{name: value}`` dict.

    Values are **summed** across all label combinations (e.g. per-LoRA adapter,
    per-model-name). This gives you engine-wide totals, which is appropriate
    for queue depths, cache usage, and token counters.

    Args:
        text: raw response body from ``GET /metrics``
        watch: set of Prometheus metric names to extract; defaults to
               ``_WATCH_METRICS``. Pass a custom set to extract other metrics.

    Returns:
        dict mapping Prometheus metric name → aggregated float value.
    """
    if watch is None:
        watch = _WATCH_METRICS

    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Prometheus exposition format:
        #   metric_name{label="v",...} value [unix_timestamp_ms]
        #   metric_name value [unix_timestamp_ms]
        # Split into at most 3 whitespace-separated tokens.
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue

        # parts[0] = "metric_name" or "metric_name{...}"
        name = parts[0].split("{", 1)[0]
        if name not in watch:
            continue

        try:
            value = float(parts[1])
        except ValueError:
            continue

        # Sum across label combinations (per-LoRA, per-model, etc.)
        result[name] = result.get(name, 0.0) + value

    return result


def _to_wandb_key(prom_name: str) -> str:
    """Convert a Prometheus metric name to a wandb key."""
    # "vllm:num_requests_running" → "vllm/num_requests_running"
    return prom_name.replace("vllm:", f"{_WANDB_PREFIX}/", 1)


def _compute_derived(raw: dict[str, float]) -> dict[str, float]:
    """Compute derived metrics from raw scraped values.

    Currently computes:
    - ``vllm/mean_ttft_ms``       – mean time-to-first-token in ms
    - ``vllm/mean_itl_ms``        – mean inter-token latency in ms
    - ``vllm/mean_e2e_latency_ms``– mean end-to-end request latency in ms
    """
    derived: dict[str, float] = {}

    ttft_sum = raw.get("vllm:time_to_first_token_seconds_sum", 0.0)
    ttft_count = raw.get("vllm:time_to_first_token_seconds_count", 0.0)
    if ttft_count > 0:
        derived[f"{_WANDB_PREFIX}/mean_ttft_ms"] = (ttft_sum / ttft_count) * 1000.0

    # vLLM v1: inter_token_latency_seconds (renamed from time_per_output_token_seconds)
    itl_sum = raw.get("vllm:inter_token_latency_seconds_sum", 0.0)
    itl_count = raw.get("vllm:inter_token_latency_seconds_count", 0.0)
    if itl_count > 0:
        derived[f"{_WANDB_PREFIX}/mean_itl_ms"] = (itl_sum / itl_count) * 1000.0

    e2e_sum = raw.get("vllm:e2e_request_latency_seconds_sum", 0.0)
    e2e_count = raw.get("vllm:e2e_request_latency_seconds_count", 0.0)
    if e2e_count > 0:
        derived[f"{_WANDB_PREFIX}/mean_e2e_latency_ms"] = (e2e_sum / e2e_count) * 1000.0

    return derived


class VllmMetricsPoller:
    """Polls vLLM ``/metrics`` on a dedicated background thread and logs to wandb.

    Runs in its **own OS thread with its own asyncio event loop**, completely
    independent of the trainer's event loop. This is necessary because the
    trainer blocks its event loop with synchronous ``ray.get()`` calls and
    ``time.sleep()`` during compute — a plain ``asyncio.Task`` would be starved
    for the duration of every training step.

    Uses ``wandb.define_metric`` to register all ``vllm/*`` keys against a
    dedicated ``vllm/poll_elapsed_s`` x-axis, so metrics appear at true
    wall-clock resolution (one data point per ``interval`` seconds), mirroring
    how wandb's built-in System metrics panel works.

    Args:
        server_addresses: list of ``"host:port"`` strings, one per vLLM replica.
        interval: poll interval in seconds (default 5 s).
        extra_watch: optional additional Prometheus metric names to scrape on
                     top of ``_WATCH_METRICS``.
    """

    def __init__(
        self,
        server_addresses: list[str],
        interval: float = 5.0,
        extra_watch: set[str] | None = None,
    ):
        self.server_addresses = list(server_addresses)
        self.interval = interval
        self.watch = _WATCH_METRICS | (extra_watch or set())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0

    def start(self) -> None:
        """Spawn the background polling thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("[VllmMetricsPoller] already running, ignoring start()")
            return

        self._start_time = time.monotonic()
        self._stop_event.clear()

        # Register vllm/* with a dedicated time x-axis — must be called from
        # the process that owns the wandb run (this method is called on the
        # trainer actor, which is where wandb.init() was called).
        try:
            import wandb
            if wandb.run is not None:
                wandb.define_metric(_STEP_METRIC)
                wandb.define_metric(f"{_WANDB_PREFIX}/*", step_metric=_STEP_METRIC)
        except Exception:
            pass

        self._thread = threading.Thread(
            target=self._thread_main,
            name="vllm_metrics_poller",
            daemon=True,  # dies automatically when the main process exits
        )
        self._thread.start()
        print(
            f"[VllmMetricsPoller] Started. Polling {self.server_addresses} "
            f"every {self.interval}s on background thread (x-axis: {_STEP_METRIC})"
        )

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)
            print("[VllmMetricsPoller] Stopped.")

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _thread_main(self) -> None:
        """Entry point for the background thread — runs its own event loop."""
        try:
            asyncio.run(self._loop())
        except Exception as exc:
            logger.debug(f"[VllmMetricsPoller] Thread exited with error: {exc}")

    async def _loop(self) -> None:
        try:
            import aiohttp
        except ImportError:
            print("[VllmMetricsPoller] aiohttp not available — poller disabled.")
            return

        async with aiohttp.ClientSession() as session:
            while not self._stop_event.is_set():
                try:
                    await self._poll_once(session)
                except Exception as exc:
                    logger.debug(f"[VllmMetricsPoller] Poll error: {exc}")
                await asyncio.sleep(self.interval)

    async def _poll_once(self, session) -> None:
        import aiohttp

        per_replica: list[dict[str, float]] = []
        for addr in self.server_addresses:
            url = f"http://{addr}/metrics"
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        raw = parse_prometheus_text(await resp.text(), watch=self.watch)
                        per_replica.append(raw)
                    else:
                        logger.debug(f"[VllmMetricsPoller] {url} returned {resp.status}")
            except Exception as exc:
                logger.debug(f"[VllmMetricsPoller] Failed to reach {url}: {exc}")

        if not per_replica:
            return

        # Average scalar metrics across replicas
        all_prom_names: set[str] = set().union(*per_replica)
        averaged: dict[str, float] = {}
        for prom_name in all_prom_names:
            values = [r[prom_name] for r in per_replica if prom_name in r]
            averaged[prom_name] = sum(values) / len(values)

        # Build wandb dict: renamed keys + derived metrics + x-axis value
        out: dict[str, float] = {_to_wandb_key(k): v for k, v in averaged.items()}
        out.update(_compute_derived(averaged))
        out[_STEP_METRIC] = time.monotonic() - self._start_time

        # wandb.log is thread-safe; calling without step= commits each poll
        # as its own row on the wall-clock x-axis.
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(out)
                return
        except Exception:
            pass

        logger.debug(f"[VllmMetricsPoller] wandb not active, dropping poll: {out}")
