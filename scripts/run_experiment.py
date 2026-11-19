"""Ejecuta un experimento completo: manifiesto → broker → nodos → recogida → evaluación → limpieza.

    python scripts/run_experiment.py --topology centralized --method fedavg [--seed 42]
           [--rounds N] [--epochs E] [--threads 2] [--variant model_120_84 ...] [--id nombre]
           [--security S0|S1|S2] [--launcher local|docker] [--cpus 1.5 --memory 2g]

`--launcher docker` lanza un contenedor `fl-node` por nodo en la red del broker (E9/E10; ver
`experiments/docker_launcher.py`); `local`, un proceso por nodo en este PC. `--security` fija el perfil
(ADR-05): S1/S2 copian la CA del broker a `<run>/security/` y S2 genera además las identidades de los nodos.

`--rounds` y `--epochs` sobrescriben la referencia (pruebas rápidas); sin ellos se usa la referencia de D0
(centralizada: 12 rondas, 3 épocas). La corrida queda en `results/runs/<id>/`.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from broker.provision_broker import admin_from_env  # noqa: E402
from core.config import PROJECT_ROOT, load_config  # noqa: E402
from experiments.collect import collect_kpis  # noqa: E402
from experiments.evaluate_run import evaluate_run  # noqa: E402
from experiments.collect import CollectError  # noqa: E402
from experiments.manifest import build_manifest  # noqa: E402
from experiments.runner import RUNS_ROOT, launch_local, prepare_run, teardown  # noqa: E402
from messaging.security import PROFILES, uses_e2e, uses_tls  # noqa: E402
from scripts.provision_identities import TLS_DIR, provision_node_identities  # noqa: E402


def make_config(variants, rounds=None, epochs=None, topology="centralized", overrides=None) -> dict:
    cfg = load_config(legacy_variants=variants)
    cfg = copy.deepcopy(cfg)
    if rounds is not None:
        cfg["topologies"][topology]["rounds"] = {"policy": "fixed", "value": int(rounds)}
    if epochs is not None:
        cfg["training"]["local_epochs"] = int(epochs)
    for path, value in (overrides or {}).items():
        node = cfg
        *parents, last = path.split(".")
        for p in parents:
            node = node[p]
        node[last] = value
    return cfg


def participants(cfg: dict, topology: str) -> list[str]:
    """Nodos que se aprovisionan y lanzan. En P2P la 6.ª Pi no tiene rol: solo el broker (04 §4.4)."""
    if topology == "p2p":
        return sorted(cfg["topologies"]["p2p"]["peers"])
    return list(cfg["platform"]["node_assignment"])


def run_experiment(topology: str, method: str, *, seed: int = 42, rounds: int | None = None,
                   epochs: int | None = None, threads: int | None = None, variants=(), experiment_id=None,
                   processed_dir: Path = PROJECT_ROOT / "data" / "processed", runs_root: Path = RUNS_ROOT,
                   timeout_s: float = 3600, command_overrides: dict | None = None,
                   config_overrides: dict | None = None, study: str | None = None,
                   security_profile: str = "S0", launcher: str = "local", docker_opts=None,
                   faults=()) -> dict:
    if security_profile not in PROFILES:
        raise ValueError(f"Perfil de seguridad desconocido: {security_profile}")
    cfg = make_config(list(variants), rounds, epochs, topology, config_overrides)
    cfg["data"]["seed"] = seed
    cfg["communication"]["security_profile"] = security_profile
    exp_id = experiment_id or f"{topology[:4]}-{method}-s{seed}-{datetime.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:4]}"
    extra = {"launcher": launcher}
    if launcher == "docker":
        from experiments.docker_launcher import DockerOptions
        docker_opts = docker_opts or DockerOptions(threads=threads)
        extra.update({"image": docker_opts.image, "cpus": docker_opts.cpus, "memory": docker_opts.memory})
    manifest = build_manifest(cfg, experiment_id=exp_id, topology=topology, method=method, seed=seed,
                              processed_dir=processed_dir, study=study, security_profile=security_profile,
                              extra=extra)
    nodes = participants(cfg, topology)
    admin = admin_from_env()
    run_dir = prepare_run(manifest, nodes, runs_root=runs_root, broker_admin=admin)
    if uses_tls(security_profile):
        sec_dir = run_dir / "security"
        sec_dir.mkdir(exist_ok=True)
        ca = TLS_DIR / "ca.pem"
        if not ca.exists():
            raise FileNotFoundError(f"Falta {ca}: python scripts/provision_identities.py tls")
        (sec_dir / "ca.pem").write_bytes(ca.read_bytes())
        if uses_e2e(security_profile):
            provision_node_identities(sec_dir, nodes)
    command_overrides = command_overrides or {}

    def command(rd: Path, node: str) -> list[str]:
        if node in command_overrides:
            return command_overrides[node](rd, node)
        cmd = [sys.executable, "-m", "node.main", "--run-dir", str(rd), "--node", node,
               "--processed", str(processed_dir)]
        return cmd + (["--threads", str(threads)] if threads else [])

    try:
        # El servidor arranca primero en la lista para que su cola esté consumiéndose cuanto antes
        order = (["infra"] if "infra" in nodes else []) + [n for n in nodes if n != "infra"]
        if launcher == "docker":
            from experiments.docker_launcher import launch_docker
            result = launch_docker(run_dir, exp_id, order, timeout_s, processed_dir=processed_dir,
                                   opts=docker_opts, tls=uses_tls(security_profile), faults=faults)
        elif launcher == "local":
            if faults:
                raise ValueError("La inyección de fallos requiere --launcher docker")
            result = launch_local(run_dir, order, timeout_s=timeout_s, command=command)
        else:
            raise ValueError(f"Lanzador desconocido: {launcher}")
    finally:
        teardown(manifest, admin)
    out = {"run_dir": run_dir, "result": result}
    lead_ok = result.exit_codes.get("infra") == 0 if "infra" in nodes else any(
        c == 0 for c in result.exit_codes.values())
    if lead_ok:
        out["kpis"] = collect_kpis(run_dir)
        out["evaluation"] = evaluate_run(run_dir, processed_dir)
    else:
        try:                                    # corridas interrumpidas (E9): los KPIs parciales sirven igual
            out["kpis"] = collect_kpis(run_dir)
        except CollectError:
            pass
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topology", default="centralized")
    ap.add_argument("--method", required=True, choices=["fedavg", "fedprox", "fednova"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--id")
    ap.add_argument("--study")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--security", default="S0", choices=list(PROFILES))
    ap.add_argument("--launcher", default="local", choices=["local", "docker"])
    ap.add_argument("--cpus", type=float, help="docker: límite de CPU por nodo")
    ap.add_argument("--memory", help="docker: límite de memoria por nodo (p. ej. 2g)")
    args = ap.parse_args(argv)
    docker_opts = None
    if args.launcher == "docker":
        from experiments.docker_launcher import DockerOptions
        docker_opts = DockerOptions(cpus=args.cpus, memory=args.memory, threads=args.threads)
    out = run_experiment(args.topology, args.method, seed=args.seed, rounds=args.rounds, epochs=args.epochs,
                         threads=args.threads, variants=args.variant, experiment_id=args.id,
                         timeout_s=args.timeout, study=args.study, security_profile=args.security,
                         launcher=args.launcher, docker_opts=docker_opts)
    res = out["result"]
    print(f"Corrida: {out['run_dir']}")
    print(f"Códigos de salida: {res.exit_codes} · duración del servidor {res.durations_s.get('infra', 0):.1f} s")
    if "evaluation" in out:
        print(json.dumps(out["evaluation"]["summary"], indent=2))
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
