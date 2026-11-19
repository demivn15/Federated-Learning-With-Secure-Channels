"""Estudio del coste de la seguridad: S0 (sin seguridad) vs S1 (AMQPS) vs S2 (AMQPS + E2E) — ADR-10 estudio 3,
RNF-05. Primera versión en Docker, antes de repetirlo en las Raspberry Pi (E11/E13).

    python -m experiments.security_study run     [--seeds 42 43 44] [--quick] [--cpus 1]
    python -m experiments.security_study analyze

`run` ejecuta topología × perfil × semilla con los nodos en contenedores (un factor: el perfil; todo lo demás
igual) y es reanudable: una corrida que ya tiene `run_summary.json` no se repite. El orden de los perfiles
rota con la semilla para no confundir el perfil con la deriva térmica o de carga del PC.
`analyze` resume cada métrica por (topología, perfil): media e IC bootstrap del 95 % entre semillas, y el
sobrecoste PAREADO frente a S0 (misma topología y semilla), en %.

Salidas en `results/studies/security_docker/`: `runs/` (corridas), `runs.csv` (una fila por corrida),
`summary.csv` y `report.md`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import PROJECT_ROOT  # noqa: E402

STUDY = "security_cost_docker"
OUT = PROJECT_ROOT / "results" / "studies" / "security_docker"
TOPOLOGIES = ("centralized", "hierarchical", "semi_decentralized", "p2p")
PROFILES = ("S0", "S1", "S2")
# Versión rápida (decisión del usuario, 2026-11-19): 1 época y un nº FIJO de rondas en todas las topologías
# (patience alto ⇒ la parada temprana no cambia la cantidad de trabajo entre perfiles).
QUICK = {
    "rounds": {"centralized": 6, "hierarchical": 6},
    "epochs": 1,
    "overrides": {
        "topologies.semi_decentralized.rounds.patience": 1000,
        "topologies.semi_decentralized.rounds.max_global_rounds": 3,
        "topologies.p2p.rounds.patience": 1000,
        "topologies.p2p.rounds.max_global_rounds": 6,
    },
}

# métrica → (descripción, unidad, "menor es mejor")
METRICS: dict[str, tuple[str, str]] = {
    "wall_s": ("Duración del experimento (contenedor más largo)", "s"),
    "lead_round_s": ("Tiempo por ronda del nodo que dirige (servidor/raíz/coordinador; P2P: media de pares)", "s"),
    "trainer_round_s": ("Tiempo por ronda de los nodos que entrenan", "s"),
    "trainer_train_s": ("Tiempo de entrenamiento por ronda", "s"),
    "publish_confirm_s": ("Publicación hasta la confirmación del broker, por fila", "s"),
    "crypto_s_per_row": ("Cifrado + descifrado E2E por nodo y ronda", "s"),
    "handshake_s_total": ("Handshakes E2E (suma de todos los nodos)", "s"),
    "handshakes": ("Handshakes E2E completados", "n"),
    "payload_bytes_per_round": ("Bytes de trama enviados por ronda (todos los nodos)", "B"),
    "nic_bytes_per_round": ("Bytes en la interfaz de red por ronda (TX+RX, todos los nodos)", "B"),
    "cpu_core_pct": ("CPU del proceso (% de un núcleo), nodos que entrenan", "%"),
    "ram_mb": ("RAM RSS al final de la ronda, nodos que entrenan", "MB"),
    "e_comm_j": ("E_comm (p·T_comm de la subida) por nodo y ronda", "J"),
    "e_comp_j": ("E_comp (Eq. 4) por nodo y ronda", "J"),
    "net_f1": ("F1 macro de red (media de las últimas 5 rondas)", ""),
    "net_recall": ("Recall macro de red (media de las últimas 5 rondas)", ""),
    "rejected": ("Mensajes rechazados (validación + canal seguro)", "n"),
}


def _mean(s: pd.Series) -> float | None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else None


def summarize_run(run_dir: Path, durations: dict[str, float] | None = None) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    k = pd.read_csv(run_dir / "kpis_all.csv")
    lead = k[k["node_id"] == "infra"]
    trainers = k[k["node_id"] != "infra"]
    if lead.empty:                                   # P2P: no hay nodo que dirija
        lead = trainers
    rounds = int(k["round"].max())
    num = lambda c: pd.to_numeric(k[c], errors="coerce").fillna(0)   # noqa: E731
    ev = json.loads((run_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    model = ev.get("global") or ev.get("mixed") or {}
    q = model.get("mean_last_5", {})
    return {
        "experiment_id": manifest["experiment_id"], "topology": manifest["topology"],
        "profile": manifest["security_profile"], "seed": manifest["seed"], "rounds": rounds,
        "wall_s": max(durations.values()) if durations else None,
        "lead_round_s": _mean(lead["t_round_s"]),
        "trainer_round_s": _mean(trainers["t_round_s"]),
        "trainer_train_s": _mean(trainers["t_train_s"]),
        "publish_confirm_s": _mean(k["t_publish_confirm_s"]),
        "crypto_s_per_row": float((num("t_encrypt_s") + num("t_decrypt_s")).mean()),
        "handshake_s_total": float(num("t_handshake_s").sum()),
        "handshakes": int(num("handshakes").sum()),
        "payload_bytes_per_round": float(num("payload_bytes_tx").sum() / rounds),
        "nic_bytes_per_round": float((num("nic_bytes_tx") + num("nic_bytes_rx")).sum() / rounds),
        "cpu_core_pct": _mean(trainers["cpu_process_percent_core"]),
        "ram_mb": _mean(trainers["ram_rss_mb_after"]),
        "e_comm_j": _mean(trainers["e_comm_transfer_j"]),
        "e_comp_j": _mean(trainers["e_comp_j"]),
        "net_f1": q.get("net_f1"), "net_recall": q.get("net_recall"),
        "rejected": int(num("messages_rejected").sum() + num("sec_rejected").sum()),
    }


# ---------------------------------------------------------------------------- campaña
def run_campaign(seeds, topologies, profiles, quick: bool, cpus: float | None, out: Path, timeout_s: float) -> None:
    from experiments.docker_launcher import DockerOptions, wait_broker_healthy
    from scripts.run_experiment import run_experiment

    runs_root = out / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    opts = DockerOptions(cpus=cpus, threads=1)
    for i, seed in enumerate(seeds):
        order = list(profiles[i % len(profiles):]) + list(profiles[:i % len(profiles)])
        for topology in topologies:
            for profile in order:
                exp_id = f"sec-{topology[:4]}-{profile}-s{seed}"
                run_dir = runs_root / exp_id
                if (run_dir / "run_summary.json").exists():
                    continue
                if run_dir.exists():
                    import shutil
                    shutil.rmtree(run_dir)               # corrida incompleta: se repite
                wait_broker_healthy(opts)
                print(f"[{STUDY}] {exp_id} …", flush=True)
                out_run = run_experiment(
                    topology, "fedavg", seed=seed, experiment_id=exp_id, runs_root=runs_root, study=STUDY,
                    rounds=QUICK["rounds"].get(topology) if quick else None, epochs=QUICK["epochs"] if quick else None,
                    threads=1, launcher="docker", docker_opts=opts, timeout_s=timeout_s,
                    config_overrides=QUICK["overrides"] if quick else None, security_profile=profile)
                res = out_run["result"]
                if not res.ok:
                    print(f"[{STUDY}] {exp_id} FALLÓ: {res.exit_codes} {res.timed_out}", flush=True)
                    continue
                summary = summarize_run(run_dir, res.durations_s)
                (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(f"[{STUDY}] {exp_id}: ronda {summary['lead_round_s']:.2f} s, F1 {summary['net_f1']:.4f}",
                      flush=True)


# ---------------------------------------------------------------------------- análisis
def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze(out: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "runs").glob("*/run_summary.json"))]
    if not rows:
        raise SystemExit("No hay corridas resumidas")
    runs = pd.DataFrame(rows).sort_values(["topology", "profile", "seed"])
    runs.to_csv(out / "runs.csv", index=False)
    records = []
    for (topo, prof), g in runs.groupby(["topology", "profile"]):
        base = runs[(runs["topology"] == topo) & (runs["profile"] == "S0")].set_index("seed")
        for m in METRICS:
            v = pd.to_numeric(g[m], errors="coerce").to_numpy(dtype=float)
            lo, hi = bootstrap_ci(v)
            rec = {"topology": topo, "profile": prof, "metric": m, "n": int((~np.isnan(v)).sum()),
                   "mean": float(np.nanmean(v)) if (~np.isnan(v)).any() else np.nan, "ci_low": lo, "ci_high": hi}
            if prof != "S0" and not base.empty:
                paired = []
                for _, r in g.iterrows():
                    b = base[m].get(r["seed"]) if r["seed"] in base.index else None
                    if b is not None and r[m] is not None and pd.notna(b) and pd.notna(r[m]) and b != 0:
                        paired.append((float(r[m]) - float(b)) / abs(float(b)) * 100)
                p = np.array(paired, dtype=float)
                rec["overhead_pct"] = float(p.mean()) if len(p) else np.nan
                rec["overhead_ci_low"], rec["overhead_ci_high"] = bootstrap_ci(p) if len(p) else (np.nan, np.nan)
            records.append(rec)
    summary = pd.DataFrame(records)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "report.md").write_text(render_report(runs, summary), encoding="utf-8")
    return summary


def _fmt(x: float | None, unit: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if unit == "B":
        return f"{x / 1024:,.1f} KiB"
    if unit in ("s",):
        return f"{x * 1000:.1f} ms" if abs(x) < 1 else f"{x:.2f} s"
    if unit == "J":
        return f"{x:.4g} J"
    if unit == "":
        return f"{x:.4f}"
    return f"{x:,.1f} {unit}".strip()


def render_report(runs: pd.DataFrame, summary: pd.DataFrame) -> str:
    seeds = sorted(runs["seed"].unique())
    lines = [f"# Coste de la seguridad en Docker: S0 vs S1 vs S2 ({STUDY})", "",
             f"{len(runs)} corridas · semillas {', '.join(map(str, seeds))} · 1 CPU por nodo · "
             "media (IC bootstrap 95 % entre semillas) · Δ% = sobrecoste pareado frente a S0 con la misma semilla.",
             ""]
    for topo in TOPOLOGIES:
        s = summary[summary["topology"] == topo]
        if s.empty:
            continue
        lines += [f"## {topo}", "", "| Métrica | S0 | S1 | Δ% S1 | S2 | Δ% S2 |", "|---|---|---|---|---|---|"]
        for m, (desc, unit) in METRICS.items():
            cell = {}
            for prof in PROFILES:
                r = s[(s["profile"] == prof) & (s["metric"] == m)]
                if r.empty:
                    cell[prof], cell[prof + "d"] = "—", ""
                    continue
                r = r.iloc[0]
                ci = "" if np.isnan(r["ci_low"]) else f" [{_fmt(r['ci_low'], unit)}, {_fmt(r['ci_high'], unit)}]"
                cell[prof] = _fmt(r["mean"], unit) + ci
                d = r.get("overhead_pct", np.nan)
                cell[prof + "d"] = "" if prof == "S0" or pd.isna(d) else f"{d:+.1f} %"
            lines.append(f"| {desc} | {cell['S0']} | {cell['S1']} | {cell['S1d']} | {cell['S2']} | {cell['S2d']} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    r.add_argument("--topologies", nargs="+", default=list(TOPOLOGIES))
    r.add_argument("--profiles", nargs="+", default=list(PROFILES))
    r.add_argument("--quick", action="store_true")
    r.add_argument("--cpus", type=float, default=1.0)
    r.add_argument("--timeout", type=float, default=1800)
    r.add_argument("--out", default=str(OUT))
    a = sub.add_parser("analyze")
    a.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")          # consola de Windows (cp1252)
    out = Path(args.out)
    if args.cmd == "run":
        run_campaign(args.seeds, args.topologies, args.profiles, args.quick, args.cpus, out, args.timeout)
        analyze(out)
    else:
        analyze(out)
    print((out / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
