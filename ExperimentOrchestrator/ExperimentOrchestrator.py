#!/usr/bin/env python3
"""
start_experiment.py - VIBE experiment startup orchestrator.

Launches the experiment stack for a given subject, gating each launch on the
appearance of the previous app's LSL outlet, then drives start/shutdown via
LSL marker streams.

Stream topology
---------------
The orchestrator owns one OUTLET and two INLETS, modeled as a small class
hierarchy:

    Stream (base: name + StreamInfo construction)
      |
      +-- OutletStream      publishes the orchestrator's own commands
      |     OrchestratorOutlet   -> BACKEND_STREAM_NAME_ORCHESTRATOR
      |
      +-- InletStream       background reader thread + process_event() hook
            GameInlet        -> BACKEND_STREAM_NAME_GAME    (game_finished, *_ack)
            LoggerInlet      -> BACKEND_STREAM_NAME_LOGGER  (logger_ready)

Each inlet processes a DIFFERENT set of events by overriding process_event().
Because every stream has a distinct name, there is no name-collision ambiguity:
each inlet resolves to exactly one outlet.

Sequence:
    1. dsi2lsl.exe          -> wait until its LSL outlet is resolvable
                               (skipped when --use_eeg_sim is passed)
    2. MyoDetector.py       -> wait until its LSL outlet is resolvable
    3. game                 -> wait until its LSL outlet is resolvable
    4. LSLLiveViewLogger.py -> wait until it publishes logger_ready
                               (started with "use_eeg_sim" when --use_eeg_sim
                               is passed)
    5. publish game_start on the orchestrator outlet (until acked by the game)
    6. on shutdown -> gracefully terminate all

Usage:
    python start_experiment.py SUBJECT_ID [--use_eeg_sim]
"""

import argparse
import json
import logging
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import psutil
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BACKEND_STREAM_NAME_GAME = "Experiment.VIBE.Backend.Game"
BACKEND_STREAM_NAME_LOGGER = "Experiment.VIBE.Backend.Logger"
BACKEND_STREAM_NAME_ORCHESTRATOR = "Experiment.VIBE.Backend.Orchestrator"

# Control messages.
MSG_START = "game_start"
MSG_GAME_FINISHED = "game_finished"
MSG_SHUTDOWN = "shutdown"
# Readiness message the LSLLiveViewLogger publishes on its stream.
MSG_READY = "logger_ready"

MSG_ACK_SUFFIX = "_ack"

# How long to wait for an app's LSL outlet to appear before giving up (seconds).
OUTLET_RESOLVE_TIMEOUT = 60.0
# Graceful shutdown grace period before force-killing a child (seconds).
TERMINATE_GRACE = 10.0

# Pre-launch checklist. Every item must be confirmed with "yes" before launch.
CHECKLIST = [
    "EEG calibrated?",
    "Eye tracking calibrated?",
]


def build_default_config(subject_id: str, use_eeg_sim: bool = False) -> dict:
    """
    Returns the launch plan. Each app entry:
        cmd          : argv list passed to subprocess.Popen
        label        : human-readable name for logging
        lsl_stream   : name of the LSL outlet to wait for (None = don't wait,
                       just confirm the process is alive)

    use_eeg_sim: when True, dsi2lsl is not launched (the logger simulates EEG
    instead), and LSLLiveViewLogger is started with the "use_eeg_sim" argument.
    """
    logger_cmd = [
        "uv", "run", "--directory", "..\\LSLService",
        "LSLLiveViewLogger.py",
        subject_id,
    ]
    if use_eeg_sim:
        logger_cmd.append("use_eeg_sim")

    apps = []
    if not use_eeg_sim:
        apps.append({
            "label": "dsi2lsl",
            "cmd": [
                "..\\dsi2lsl-win\\dsi2lsl.exe",
                "--port=COM15",
                "--lsl-stream-name=Experiment.VIBE.EEG",
                "--reference=Pz"
            ],
            "lsl_stream": "Experiment.VIBE.EEG",
        })
    apps.append({
        "label": "game",
        "cmd": [
            "..\\Experiment\\dist\\Experiment.exe",
        ],
        "lsl_stream": "Experiment.VIBE.GameEventStream",
        "skip_launch": True,
    })
    apps.append({
        "label": "LSLLiveViewLogger",
        "cmd": logger_cmd,
        # Logger publishes READY on its backend stream when up.
        "lsl_stream": None,
        "wait_ready": True,
    })

    return {"apps": apps}


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
log = logging.getLogger("vibe")


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    logging.getLogger("pylsl").setLevel(logging.ERROR)


# --------------------------------------------------------------------------- #
# Launched-process wrapper
# --------------------------------------------------------------------------- #
@dataclass
class LaunchedApp:
    label: str
    cmd: list
    lsl_stream: Optional[str]
    wait_ready: bool = False
    skip_launch: bool = False
    proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        log.info("[startup] launching %s: %s", self.label, " ".join(self.cmd))

        if self.skip_launch:
            return

        try:
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"cannot launch {self.label!r}: {exc}. Check the path in CONFIG."
            ) from exc

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def terminate(self, grace: float = TERMINATE_GRACE) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        log.info("[shutdown] terminating %s (pid %s)", self.label, self.proc.pid)

        try:
            parent = psutil.Process(self.proc.pid)
        except psutil.NoSuchProcess:
            return

        # Capture descendants BEFORE killing the parent.
        procs = parent.children(recursive=True) + [parent]

        for p in procs:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass

        gone, alive = psutil.wait_procs(procs, timeout=grace)
        for p in alive:
            log.info("[shutdown] %s pid %s did not exit; killing", self.label, p.pid)
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        self.proc.wait()


# --------------------------------------------------------------------------- #
# Stream hierarchy
# --------------------------------------------------------------------------- #
class Stream:
    """
    Base for any LSL marker stream owned by the orchestrator. Holds the stream name
    and knows how to build a standard 1-channel string marker StreamInfo.

    Subclasses are either an OutletStream (we publish) or an InletStream (we
    read). The base intentionally does NOT open any LSL endpoint; that is the
    responsibility of the concrete subclass, so the two directions stay clearly
    separated.
    """

    def __init__(self, stream_name: str) -> None:
        self.stream_name = stream_name

    def _build_info(self, source_id: str) -> StreamInfo:
        return StreamInfo(
            name=self.stream_name,
            type="Markers",
            channel_count=1,
            nominal_srate=0,        # irregular marker stream
            channel_format="string",
            source_id=source_id,
        )

    def close(self) -> None:
        """Release resources. Overridden where there is something to release."""


class OutletStream(Stream):
    """A stream this process publishes to."""

    def __init__(self, stream_name: str, source_id: str = "vibe_startup_script") -> None:
        super().__init__(stream_name)
        self.outlet = StreamOutlet(self._build_info(source_id))

    def publish(self, message: str) -> None:
        log.info("[%s >] %s", self.stream_name, message)
        self.outlet.push_sample([message])


class InletStream(Stream):
    """
    A stream this process reads from, on a background thread.

    Resolution: because each inlet has a unique name, resolve_byprop returns the
    single matching outlet and we attach to it. The reader calls process_event()
    for every token received; subclasses override that method to handle their
    own distinct events.
    """

    def __init__(self, stream_name: str, resolve_timeout: float = 10.0) -> None:
        super().__init__(stream_name)
        self.resolve_timeout = resolve_timeout
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._inlet: Optional[StreamInlet] = None

    def start_reader(self) -> None:
        self._reader.start()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.stop()

    def _read_loop(self) -> None:
        results = resolve_byprop(
            "name", self.stream_name, minimum=1, timeout=self.resolve_timeout
        )
        if not results:
            log.info("[%s] could not resolve inlet; reader inactive", self.stream_name)
            return
        self._inlet = StreamInlet(results[0])
        log.info("[%s] reader attached", self.stream_name)

        while not self._stop.is_set():
            sample, _ts = self._inlet.pull_sample(timeout=0.5)
            if sample is None:
                continue
            message = sample[0]
            log.info("[%s <] %s", self.stream_name, message)
            self.process_event(message.strip())

    # --- hook for subclasses ------------------------------------------------ #
    def process_event(self, token: str) -> None:
        """Handle one received token. Overridden by each concrete inlet."""
        raise NotImplementedError


class GameInlet(InletStream):
    """
    Reads the game's backend stream. Responsible for command acknowledgements
    (e.g. game_start_ack) and the game_finished signal.
    """

    def __init__(self, stream_name: str = BACKEND_STREAM_NAME_GAME) -> None:
        super().__init__(stream_name)
        self.acks: set[str] = set()
        self.finished_event = threading.Event()

    def process_event(self, token: str) -> None:
        if token.endswith(MSG_ACK_SUFFIX):
            self.acks.add(token)
        elif token == MSG_GAME_FINISHED:
            self.finished_event.set()

    def is_acked(self, message: str) -> bool:
        return f"{message}{MSG_ACK_SUFFIX}" in self.acks


class LoggerInlet(InletStream):
    """
    Reads the logger's backend stream. Its only job is to detect the logger's
    readiness signal.
    """

    def __init__(self, stream_name: str = BACKEND_STREAM_NAME_LOGGER) -> None:
        super().__init__(stream_name, resolve_timeout=30)
        self.ready_event = threading.Event()

    def process_event(self, token: str) -> None:
        if token == MSG_READY:
            self.ready_event.set()


# --------------------------------------------------------------------------- #
# Orchestration helpers
# --------------------------------------------------------------------------- #
def wait_for_outlet(stream_name: str, timeout: float) -> bool:
    """Block until an LSL outlet with the given name is resolvable."""
    log.info("[startup] waiting for LSL outlet %r ...", stream_name)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = resolve_byprop("name", stream_name, minimum=1, timeout=1.0)
        if found:
            log.info("[startup] outlet %r is live", stream_name)
            return True
    return False


def run_checklist(items: list) -> bool:
    """
    Prompt the operator with each checklist item. Requires an explicit "yes"
    for every item. Returns False if any item is not confirmed, or on
    EOF/interrupt, so the caller can abort before launching.
    """
    log.info("[checklist] pre-launch checklist - answer 'yes' to each")
    for item in items:
        try:
            answer = input(f"  {item} [yes/no]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            log.info("[checklist] aborted by operator")
            return False
        if answer != "yes":
            log.info("[checklist] %r not confirmed (got %r); aborting", item, answer)
            return False
    log.info("[checklist] all items confirmed")
    return True


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class Orchestrator:
    def __init__(self, subject_id: str, config: dict) -> None:
        self.subject_id = subject_id
        self.apps = [
            LaunchedApp(
                label=a["label"],
                cmd=a["cmd"],
                lsl_stream=a.get("lsl_stream"),
                wait_ready=a.get("wait_ready", False),
                skip_launch=a.get("skip_launch", False),
            )
            for a in config["apps"]
        ]

        # One outlet (the orchestrator's own commands)
        self.orchestrator = OutletStream(BACKEND_STREAM_NAME_ORCHESTRATOR)
        self.game_inlet = None
        self.logger_inlet = None
        self.inlets: list[InletStream] = []

        self.shutdown_event = threading.Event()
        self._shutting_down = False

    def run(self) -> int:
        log.info("[startup] VIBE experiment start, subject=%s", self.subject_id)

        if not run_checklist(CHECKLIST):
            log.info("[startup] checklist not passed; nothing launched")
            return 1

        try:
            self._launch_sequence()
        except RuntimeError as exc:
            log.info("[startup] aborting: %s", exc)
            self.shutdown()
            return 1


        self.game_inlet = GameInlet()
        self.game_inlet.start_reader()
        self.inlets: list[InletStream] = [self.game_inlet, self.logger_inlet]

        # Step 5: start the game; re-publish until the game acknowledges.
        while not self.game_inlet.is_acked(MSG_START) and not self.shutdown_event.is_set():
            self.orchestrator.publish(MSG_START)
            time.sleep(2)

        log.info("[startup] all apps up; experiment running. Waiting for shutdown.")

        # Step 6: wait for shutdown (local signal, game finished, or dead child).
        self._wait_until_shutdown()
        self.shutdown()
        return 0

    def _launch_sequence(self) -> None:
        for app in self.apps:
            app.start()
            time.sleep(0.5)  # let the process get past immediate-crash territory
            if not app.is_alive() and not app.skip_launch:
                raise RuntimeError(f"{app.label} exited immediately on launch")

            if app.lsl_stream:
                if not wait_for_outlet(app.lsl_stream, OUTLET_RESOLVE_TIMEOUT):
                    raise RuntimeError(
                        f"timed out waiting for {app.label} outlet "
                        f"{app.lsl_stream!r}"
                    )
            elif app.wait_ready:
                log.info("[startup] waiting for %s to publish %r ...",app.label, MSG_READY)

                self.logger_inlet = LoggerInlet()
                self.logger_inlet.start_reader()
                self.inlets.append(self.logger_inlet)

                if not self.logger_inlet.ready_event.wait(timeout=OUTLET_RESOLVE_TIMEOUT):
                    raise RuntimeError(
                        f"timed out waiting for {app.label} to publish {MSG_READY!r}"
                    )
                log.info("[startup] %s reported %s", app.label, MSG_READY)
            else:
                log.info("[startup] %s running (no outlet to await)", app.label)

    def _wait_until_shutdown(self) -> None:
        while not self.shutdown_event.is_set():
            if self.game_inlet.finished_event.is_set():
                log.info("[startup] game reported %s", MSG_GAME_FINISHED)
                return
            dead = [a.label for a in self.apps if not a.skip_launch and not a.is_alive()]
            if dead:
                log.info("[startup] app(s) exited unexpectedly: %s", ", ".join(dead))
                return
            time.sleep(0.5)

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        log.info("[shutdown] shutting down experiment")

        # Tell downstream apps to stop, then close our streams.
        self.orchestrator.publish(MSG_SHUTDOWN)
        for inlet in self.inlets:
            inlet.close()

        # Terminate in reverse launch order (last up, first down).
        for app in reversed(self.apps):
            app.terminate()
        log.info("[shutdown] all processes terminated")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VIBE experiment startup orchestrator")
    p.add_argument("subject_id", help="subject identifier")
    p.add_argument("--config", help="optional JSON config file overriding the launch plan")
    p.add_argument(
        "--use_eeg_sim",
        action="store_true",
        help="simulate EEG instead of launching dsi2lsl; passed through to LSLLiveViewLogger",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    else:
        config = build_default_config(args.subject_id, use_eeg_sim=args.use_eeg_sim)

    orch = Orchestrator(args.subject_id, config)

    # Allow Ctrl+C as an operator escape hatch in addition to a backend shutdown.
    def _signal_handler(_signum, _frame):
        log.info("[shutdown] local interrupt received")
        orch.shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    return orch.run()


if __name__ == "__main__":
    sys.exit(main())