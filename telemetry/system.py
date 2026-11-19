"""Métricas de sistema del nodo (D0 Tabla 8; RNF-13).

Dos semánticas de CPU, ambas registradas con nombres distintos:
  * `cpu_percent_legacy()` — la del código base (`metrics_utils.get_cpu_percent`): CPU de TODO el sistema
    durante 0,2 s, muestreada antes de recibir el modelo y después de evaluar (nunca durante el
    entrenamiento). Sirve para reproducir la Tabla 8 de D0.
  * `ProcessCpu` — CPU del propio proceso durante la ronda: Δ(tiempo de CPU user+system) / Δ(tiempo de pared),
    en % de un núcleo (puede superar 100 en varios núcleos) y normalizada al total de núcleos.
RAM: RSS del proceso en MB (igual que `get_ram_mb`). Temperatura y frecuencia: None si no hay sensor (p. ej. Windows).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

_THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
_SENSOR_PRIORITY = ("cpu_thermal", "coretemp", "k10temp", "cpu-thermal", "soc_thermal", "acpitz")


def cpu_percent_legacy(interval: float = 0.2) -> float:
    return float(psutil.cpu_percent(interval=interval))


def ram_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def temperature_c() -> float | None:
    sensors = getattr(psutil, "sensors_temperatures", None)
    if sensors is not None:
        try:
            temps = sensors() or {}
            for name in _SENSOR_PRIORITY:
                if temps.get(name):
                    return round(float(temps[name][0].current), 2)
            for entries in temps.values():
                if entries:
                    return round(float(entries[0].current), 2)
        except Exception:
            pass
    try:
        return round(int(_THERMAL_ZONE.read_text().strip()) / 1000.0, 2)   # Raspberry Pi sin psutil sensors
    except (OSError, ValueError):
        return None


def declared_telemetry() -> dict[str, float | None]:
    """Valores AUTODECLARADOS que un nodo SD envía en `CLAIM` para `SD-score` (D3 §2.1.3).

    `cpu_percent` sin intervalo (desde la llamada anterior: no bloquea), RAM disponible del sistema y
    temperatura. Son falsificables por el propio nodo (R-02): pesan menos que las variables observadas.
    """
    return {"cpu_percent": float(psutil.cpu_percent(interval=None)),
            "ram_available_mb": round(psutil.virtual_memory().available / (1024 * 1024), 1),
            "temperature_c": temperature_c()}


def cpu_freq_mhz() -> float | None:
    try:
        f = psutil.cpu_freq()
        return round(float(f.current), 1) if f and f.current > 0 else None
    except Exception:
        return None


@dataclass
class NicCounters:
    bytes_sent: int
    bytes_recv: int

    @classmethod
    def snapshot(cls, interface: str | None = None) -> "NicCounters":
        if interface:
            c = psutil.net_io_counters(pernic=True).get(interface)
            if c is None:
                raise ValueError(f"Interfaz de red desconocida: {interface}")
        else:
            c = psutil.net_io_counters()
        return cls(int(c.bytes_sent), int(c.bytes_recv))

    def delta(self, later: "NicCounters") -> tuple[int, int]:
        return later.bytes_sent - self.bytes_sent, later.bytes_recv - self.bytes_recv


class ProcessCpu:
    """Uso de CPU del proceso entre `start()` y `stop()`."""

    def __init__(self) -> None:
        self._proc = psutil.Process(os.getpid())
        self._cores = psutil.cpu_count() or 1

    def start(self) -> None:
        t = self._proc.cpu_times()
        self._cpu0, self._wall0 = t.user + t.system, time.perf_counter()

    def stop(self) -> dict[str, float]:
        t = self._proc.cpu_times()
        cpu = (t.user + t.system) - self._cpu0
        wall = max(time.perf_counter() - self._wall0, 1e-9)
        return {"cpu_process_s": cpu,
                "cpu_process_percent_core": 100.0 * cpu / wall,
                "cpu_process_percent_total": 100.0 * cpu / wall / self._cores}
