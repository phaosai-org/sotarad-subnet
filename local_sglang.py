"""
Start and stop a local SGLang OpenAI-compatible server via subprocess.

Uses ``python -m sglang.launch_server`` (same as the CLI) so no in-process
import of ``sglang`` is required. The validator polls until HTTP is ready, then
POSTs to ``{base}/v1/chat/completions`` like Chutes.

Requires ``sglang`` installed in the same Python environment as the validator.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from subprocess import Popen
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class SglangSubprocessServer:
    """Manage ``sglang.launch_server`` as a child process (new session / process group)."""

    def __init__(
        self,
        hf_repo: str,
        revision: str,
        host: str,
        port: int,
        *,
        extra_argv: Optional[list[str]] = None,
        startup_timeout_s: float = 600.0,
    ) -> None:
        self.hf_repo = hf_repo
        self.revision = revision
        self.host = host
        self.port = int(port)
        self.extra_argv = list(extra_argv or [])
        self.startup_timeout_s = float(startup_timeout_s)
        self._proc: Optional[Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("SGLang server already started")
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.hf_repo,
            "--revision",
            self.revision,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ] + self.extra_argv
        logger.info("Starting SGLang subprocess: %s", " ".join(cmd))
        # New session so we can signal the whole process group on shutdown.
        self._proc = Popen(
            cmd,
            stdin=Popen.DEVNULL,
            stdout=None,
            stderr=None,
            start_new_session=True,
        )

    async def wait_until_ready(self, session: aiohttp.ClientSession) -> bool:
        deadline = time.monotonic() + self.startup_timeout_s
        base = self.base_url.rstrip("/")
        probe_urls = [f"{base}/health", f"{base}/v1/models"]
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                logger.error(
                    "SGLang process exited early with code %s",
                    self._proc.returncode,
                )
                return False
            for url in probe_urls:
                try:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status < 500:
                            logger.info("SGLang server ready at %s", base)
                            return True
                except Exception:
                    pass
            await asyncio.sleep(2.0)
        logger.error("SGLang server did not become ready within %.0fs", self.startup_timeout_s)
        return False

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except ProcessLookupError:
                return
        try:
            proc.wait(timeout=60)
        except Exception:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        logger.info("SGLang subprocess stopped")
