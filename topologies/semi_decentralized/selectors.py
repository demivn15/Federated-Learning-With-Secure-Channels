"""Selectores del agregador de la semi-descentralizada (RF-04; 04 ADR-08). Funciones puras y deterministas.

Todos reciben la lista de candidatos (el coordinador la fija y la envía en `ELECTION`) y devuelven el
ganador. Cada Client Node los calcula por su cuenta y lo anuncia en `CLAIM`; el coordinador calcula lo mismo
y verifica que todos coinciden (D0 §3.3.3: semilla compartida; el coordinador solo registra; I-26).

  * `lottery_d0`      — referencia: U ~ uniforme en [0,1) derivada de (semilla, ronda, intento); con k
                        candidatos ordenados, ganador = candidatos[⌊U·k⌋] (k = 5 ⇒ los 5 subintervalos de D0).
                        U = primeros 64 bits de SHA-256("lottery_d0|semilla|ronda|intento") / 2^64: idéntica en
                        cualquier plataforma y versión de Python. La semilla la fija el experimentador en el
                        manifiesto ⇒ los nodos pueden predecirla pero no manipularla (W-17; la VRF es la
                        variante del ataque F5).
  * `lottery_legacy`  — la del código P3 (`server.py`): `randint(1, 100)` por candidato y gana el máximo
                        (el primero en caso de empate). Aquí con un generador sembrado para que sea verificable.
  * `round_robin`     — alternativa citada en D0 §3.3.3: candidatos[(ronda − 1 + intento − 1) mod k].
  * `sd_score_hybrid` — `SD-score` de D3 §2.1.3 en versión híbrida (02 §3.2, R-02): gana la mayor puntuación
                        (`hybrid_scores`); los empates (p. ej. sin historial) se resuelven con `lottery_d0`.
"""
from __future__ import annotations

import hashlib
import random
from typing import Mapping, Sequence

SELECTORS = ("lottery_d0", "lottery_legacy", "round_robin", "sd_score_hybrid")
_TIE = 1e-9


def uniform_draw(seed: int, round_: int, attempt: int, tag: str = "lottery_d0") -> float:
    digest = hashlib.sha256(f"{tag}|{seed}|{round_}|{attempt}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64


def _sorted(candidates: Sequence[str]) -> list[str]:
    c = sorted(candidates)
    if not c:
        raise ValueError("Sin candidatos")
    if len(set(c)) != len(c):
        raise ValueError("Candidatos repetidos")
    return c


def lottery_d0(candidates: Sequence[str], seed: int, round_: int, attempt: int = 1) -> str:
    c = _sorted(candidates)
    return c[int(uniform_draw(seed, round_, attempt) * len(c))]


def lottery_legacy(candidates: Sequence[str], seed: int, round_: int, attempt: int = 1) -> str:
    c = _sorted(candidates)
    rng = random.Random(f"lottery_legacy|{seed}|{round_}|{attempt}")
    draws = {n: rng.randint(1, 100) for n in c}
    return max(draws, key=draws.get)


def round_robin(candidates: Sequence[str], seed: int, round_: int, attempt: int = 1) -> str:
    c = _sorted(candidates)
    return c[(round_ - 1 + attempt - 1) % len(c)]


def score_winner(candidates: Sequence[str], scores: Mapping[str, float], seed: int, round_: int,
                 attempt: int = 1) -> str:
    c = _sorted(candidates)
    missing = [n for n in c if n not in scores]
    if missing:
        raise ValueError(f"Faltan puntuaciones de {missing}")
    best = max(scores[n] for n in c)
    tied = [n for n in c if scores[n] >= best - _TIE]
    return tied[0] if len(tied) == 1 else lottery_d0(tied, seed, round_, attempt)


def select(selector: str, candidates: Sequence[str], seed: int, round_: int, attempt: int = 1,
           scores: Mapping[str, float] | None = None) -> str:
    if selector == "lottery_d0":
        return lottery_d0(candidates, seed, round_, attempt)
    if selector == "lottery_legacy":
        return lottery_legacy(candidates, seed, round_, attempt)
    if selector == "round_robin":
        return round_robin(candidates, seed, round_, attempt)
    if selector == "sd_score_hybrid":
        if scores is None:
            raise ValueError("sd_score_hybrid necesita las puntuaciones de ELECTION")
        return score_winner(candidates, scores, seed, round_, attempt)
    raise ValueError(f"Selector desconocido: {selector!r} (opciones: {SELECTORS})")


# ---------------------------------------------------------------- SD-score híbrido
def _minmax(values: Mapping[str, float]) -> dict[str, float]:
    lo, hi = min(values.values()), max(values.values())
    return {k: 0.0 if hi - lo < _TIE else (v - lo) / (hi - lo) for k, v in values.items()}


def hybrid_scores(candidates: Sequence[str], observed: Mapping[str, Mapping[str, float | None]],
                  declared: Mapping[str, Mapping[str, float | None]],
                  weights: Mapping[str, float]) -> dict[str, float]:
    """Score_i = w_cpu·CPU_libre + w_ram·RAM_libre − w_lat·retraso − w_rtt·RTT − w_temp·temperatura.

    Cada término se normaliza a [0, 1] entre los candidatos (min–max). Variables:
      * OBSERVADAS por otros (no falsificables por el propio nodo; R-02): `lateness_s` = retraso medio de
        llegada de sus modelos medido por los agregadores; `rtt_s` = respuesta a `ELECTION` medida por el
        coordinador. Pesan más (pesos en la configuración).
      * AUTODECLARADAS (en `CLAIM`; firmadas en el perfil S2, E10): `cpu_percent`, `ram_available_mb`,
        `temperature_c`.
    Un valor ausente (nodo sin historial, sin sensor) toma la media de los demás candidatos: ni premia ni
    penaliza. Sin datos en absoluto, todos empatan y decide `lottery_d0`.
    """
    c = _sorted(candidates)

    def column(src: Mapping[str, Mapping[str, float | None]], key: str) -> dict[str, float]:
        vals = {n: src.get(n, {}).get(key) for n in c}
        known = [v for v in vals.values() if v is not None]
        fill = sum(known) / len(known) if known else 0.0
        return _minmax({n: fill if v is None else float(v) for n, v in vals.items()})

    lat, rtt = column(observed, "lateness_s"), column(observed, "rtt_s")
    cpu, ram, temp = (column(declared, "cpu_percent"), column(declared, "ram_available_mb"),
                      column(declared, "temperature_c"))
    w = {k: float(weights.get(k, 0.0)) for k in ("lateness", "rtt", "cpu_idle", "ram_free", "temperature")}
    return {n: round(w["cpu_idle"] * (1 - cpu[n]) + w["ram_free"] * ram[n] - w["lateness"] * lat[n]
                     - w["rtt"] * rtt[n] - w["temperature"] * temp[n], 9) for n in c}
