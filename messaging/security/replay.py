"""Ventana anti-replay por (emisor, época) (04 ADR-05, RF-20).

Guarda la mayor `seq` aceptada y un mapa de bits de las últimas `W`. Rechaza duplicados y mensajes más
antiguos que la ventana. Tolera huecos (mensajes perdidos por el broker) y reordenaciones dentro de la
ventana. Se consulta DESPUÉS de verificar el AEAD (un atacante no puede desplazar la ventana con
mensajes falsos) y se marca solo si el mensaje se acepta.
"""
from __future__ import annotations


class ReplayWindow:
    def __init__(self, size: int = 64) -> None:
        if size < 1:
            raise ValueError("La ventana debe tener al menos 1 posición")
        self.size = size
        self.highest = 0          # seq empieza en 1
        self._bits = 0            # bit i ⇒ (highest - i) ya visto

    def check(self, seq: int) -> str | None:
        """None si `seq` es nuevo y está dentro de la ventana; si no, el motivo del rechazo."""
        if seq < 1:
            return "seq fuera de rango"
        if seq > self.highest:
            return None
        offset = self.highest - seq
        if offset >= self.size:
            return "seq demasiado antiguo (fuera de la ventana)"
        if self._bits >> offset & 1:
            return "seq duplicado (replay)"
        return None

    def mark(self, seq: int) -> None:
        if seq > self.highest:
            shift = seq - self.highest
            self._bits = (((self._bits << shift) | 1) & ((1 << self.size) - 1)) if shift < self.size else 1
            self.highest = seq
        else:
            self._bits |= 1 << (self.highest - seq)
