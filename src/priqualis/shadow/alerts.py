"""Anomaly Detection for Priqualis."""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class Alert(BaseModel):
    alert_id: str
    alert_type: Literal["zscore", "trend", "threshold"]
    severity: Literal["info", "warning", "critical"]
    rule_id: str
    message: str
    current_value: int
    threshold: float
    z_score: float | None = None
    detected_at: datetime = Field(default_factory=datetime.now)

@dataclass(slots=True, frozen=True)
class AlertConfig:
    zscore_threshold: float = 2.0
    min_history_days: int = 7
    critical_zscore: float = 3.0
    enable_trend_detection: bool = True

DEFAULT_ALERT_CONFIG = AlertConfig()

class AnomalyDetector:
    """Detects anomalies using Z-score and trend analysis."""

    def __init__(self, config: AlertConfig | None = None, history: dict[str, list[int]] | None = None):
        self.config = config or DEFAULT_ALERT_CONFIG
        self._history = history or {}

    def record_batch(self, rule_counts: dict[str, int]) -> None:
        for r, c in rule_counts.items():
            self._history.setdefault(r, []).append(c)
            if len(self._history[r]) > 90: self._history[r] = self._history[r][-90:]

    def detect_zscore(self, rule_id: str, count: int) -> Alert | None:
        hist = self._history.get(rule_id, [])
        if len(hist) < self.config.min_history_days: return None
        
        mean = sum(hist) / len(hist)
        std = (sum((x - mean) ** 2 for x in hist) / len(hist)) ** 0.5
        
        if std == 0: 
            if count <= mean:
                return None
            return Alert(
                alert_id=f"ALERT_{rule_id}",
                alert_type="zscore",
                severity="critical",
                rule_id=rule_id,
                message="Zero variance exceeded",
                current_value=count,
                threshold=mean,
                z_score=float("inf")
            )
        
        z = (count - mean) / std
        if z < self.config.zscore_threshold: return None
        
        sev = "critical" if z >= self.config.critical_zscore else "warning"
        return Alert(
            alert_id=f"AL_{rule_id}_{datetime.now().strftime('%f')}",
            alert_type="zscore",
            severity=sev,
            rule_id=rule_id,
            message=f"Z-score {z:.1f} > {mean:.0f}",
            current_value=count,
            threshold=self.config.zscore_threshold,
            z_score=z
        )

    def check_batch(self, rule_counts: dict[str, int], record: bool = True) -> list[Alert]:
        alerts = []
        for r, c in rule_counts.items():
            if a := self.detect_zscore(r, c): alerts.append(a)
        if record: self.record_batch(rule_counts)
        return alerts

class AlertManager:
    """Manages alerts."""
    def __init__(self): self._alerts: list[Alert] = []
    def add_alerts(self, alerts: list[Alert]): self._alerts.extend(alerts)
    def get_alerts(self, severity: str | None = None, days: int = 7) -> list[Alert]:
        cutoff = datetime.now() - timedelta(days=days)
        res = [a for a in self._alerts if a.detected_at >= cutoff]
        return [a for a in res if a.severity == severity] if severity else res
