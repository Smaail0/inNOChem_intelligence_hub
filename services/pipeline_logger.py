"""Reusable structured pipeline logging and timing helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass(frozen=True)
class PipelineEvent:
    stage: str
    event_type: str
    timestamp: str
    message: str
    duration_seconds: float | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PipelineLogger:
    """Collect structured pipeline events and emit standard Python logs."""

    def __init__(self, sink: Callable[[PipelineEvent], None] | None = None) -> None:
        self.events: list[PipelineEvent] = []
        self._active_stages: dict[str, tuple[float, str]] = {}
        self._sink = sink
        self._logger = logging.getLogger("innochem.pipeline")

    def start_stage(
        self,
        stage: str,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        timestamp = _now()
        self._active_stages[stage] = (time.perf_counter(), timestamp)
        self._record(
            PipelineEvent(
                stage=stage,
                event_type="start",
                timestamp=timestamp,
                message=message or f"{stage} started",
                metadata=metadata or {},
            )
        )

    def end_stage(
        self,
        stage: str,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        timestamp = _now()
        start = self._active_stages.pop(stage, None)
        duration = round(time.perf_counter() - start[0], 3) if start else None
        self._record(
            PipelineEvent(
                stage=stage,
                event_type="end",
                timestamp=timestamp,
                message=message or f"{stage} completed",
                duration_seconds=duration,
                metadata=metadata or {},
            )
        )

    def progress(
        self,
        stage: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._record(
            PipelineEvent(
                stage=stage,
                event_type="progress",
                timestamp=_now(),
                message=message,
                metadata=metadata or {},
            )
        )

    def message(
        self,
        stage: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._record(
            PipelineEvent(
                stage=stage,
                event_type="message",
                timestamp=_now(),
                message=message,
                metadata=metadata or {},
            )
        )

    def timing_summary(self) -> dict[str, float]:
        summary = {}
        for event in self.events:
            if event.event_type == "end" and event.duration_seconds is not None:
                summary[event.stage] = round(
                    summary.get(event.stage, 0.0) + event.duration_seconds,
                    3,
                )

        end_events = [event for event in self.events if event.event_type == "end"]
        if end_events and self.events:
            first_start = next((event for event in self.events if event.event_type == "start"), None)
            if first_start is not None:
                start_time = datetime.fromisoformat(first_start.timestamp)
                end_time = datetime.fromisoformat(end_events[-1].timestamp)
                summary["Total"] = round((end_time - start_time).total_seconds(), 3)

        return summary

    def _record(self, event: PipelineEvent) -> None:
        self.events.append(event)
        self._logger.info("%s", event.to_dict())
        if self._sink is not None:
            self._sink(event)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
