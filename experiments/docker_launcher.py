"""Lanzador de nodos en contenedores Docker (E9, E10): un contenedor por nodo en la red del broker.

Equivale a `experiments.runner.launch_local`, pero cada nodo es un contenedor `fl-node` con:
  * el código del proyecto montado en `/app` en solo lectura (el hash de código del manifiesto es el del PC);
  * la carpeta de la corrida en `/fl/run` (KPIs, checkpoints, eventos, credenciales, identidades);
  * los datos procesados en `/fl/data` en solo lectura;
  * la red `fl-net` (llega al broker como `fl-rabbitmq`) y `NET_ADMIN` (para `tc netem`);
  * límites opcionales de CPU y memoria (`--cpus`, `--memory`) para aproximar una Pi.
Opcionalmente ejecuta un plan de fallos (`experiments/faults.py`) mientras corren los contenedores.
Al terminar guarda `docker logs --timestamps` en `logs/<nodo>.log` y elimina los contenedores.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from core.config import PROJECT_ROOT
from experiments.faults import Fault, FaultInjector
from experiments.runner import RunResult

RUN_MOUNT, DATA_MOUNT = "/fl/run", "/fl/data"
EntryFactory = Callable[[list[str]], list[str]]         # args de node.main → argv completo de python


@dataclass
class DockerOptions:
    image: str = "fl-node:dev"
    network: str = "fl-net"
    broker_container: str = "fl-rabbitmq"
    broker_host: str = "fl-rabbitmq"
    amqp_port: int = 5672
    amqps_port: int = 5671
    cpus: float | None = None
    memory: str | None = None
    threads: int | None = None
    entries: dict[str, EntryFactory] = field(default_factory=dict)


def container_name(experiment_id: str, node_id: str) -> str:
    return f"fl-{experiment_id}-{node_id}".lower()[:120]


def _docker(*args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def _logs(container: str) -> str:
    r = _docker("logs", container, timeout=30)
    return r.stdout + r.stderr


def node_args(node_id: str, opts: DockerOptions, tls: bool) -> list[str]:
    args = ["--run-dir", RUN_MOUNT, "--node", node_id, "--processed", DATA_MOUNT,
            "--amqp-host", opts.broker_host, "--amqp-port", str(opts.amqps_port if tls else opts.amqp_port)]
    return args + (["--threads", str(opts.threads)] if opts.threads else [])


def run_command(run_dir: Path, processed_dir: Path, experiment_id: str, node_id: str, opts: DockerOptions,
                tls: bool) -> list[str]:
    entry = opts.entries.get(node_id, lambda a: ["-m", "node.main", *a])
    cmd = ["docker", "run", "-d", "--name", container_name(experiment_id, node_id), "--hostname", node_id,
           "--network", opts.network, "--cap-add", "NET_ADMIN",
           "--label", f"fl.experiment={experiment_id}", "--label", f"fl.node={node_id}",
           "-v", f"{PROJECT_ROOT}:/app:ro", "-v", f"{Path(run_dir).resolve()}:{RUN_MOUNT}",
           "-v", f"{Path(processed_dir).resolve()}:{DATA_MOUNT}:ro"]
    if opts.cpus:
        cmd += ["--cpus", str(opts.cpus)]
    if opts.memory:
        cmd += ["--memory", opts.memory]
    return cmd + [opts.image, *entry(node_args(node_id, opts, tls))]


def check_environment(opts: DockerOptions) -> None:
    """Falla pronto si falta la imagen, la red o el broker."""
    if _docker("image", "inspect", opts.image).returncode != 0:
        raise RuntimeError(f"Falta la imagen {opts.image}: docker build -f docker/node.Dockerfile -t {opts.image} docker")
    if _docker("network", "inspect", opts.network).returncode != 0:
        raise RuntimeError(f"Falta la red {opts.network}: docker compose -f docker/compose.yaml up -d")
    state = _docker("inspect", "-f", "{{.State.Running}}", opts.broker_container).stdout.strip()
    if state != "true":
        raise RuntimeError(f"El broker {opts.broker_container} no está en marcha")


def wait_broker_healthy(opts: DockerOptions, timeout_s: float = 180) -> bool:
    """Espera a que el broker vuelva a estar sano (p. ej. tras un fallo inyectado sobre él)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = _docker("inspect", "-f", "{{.State.Running}} {{.State.Health.Status}}", opts.broker_container)
        if r.stdout.strip() == "true healthy":
            return True
        if r.stdout.startswith("false"):
            _docker("start", opts.broker_container)
        time.sleep(2)
    return False


def launch_docker(run_dir: str | Path, experiment_id: str, node_ids: Sequence[str], timeout_s: float, *,
                  processed_dir: str | Path, opts: DockerOptions = DockerOptions(), tls: bool = False,
                  faults: Sequence[Fault] = ()) -> RunResult:
    run_dir = Path(run_dir)
    check_environment(opts)
    names = {n: container_name(experiment_id, n) for n in node_ids}
    waiters: dict[str, subprocess.Popen] = {}
    t0: dict[str, float] = {}
    injector = None
    try:
        for n in node_ids:
            _docker("rm", "-f", names[n])
            res = subprocess.run(run_command(run_dir, Path(processed_dir), experiment_id, n, opts, tls),
                                 capture_output=True, text=True, timeout=120)
            if res.returncode != 0:
                raise RuntimeError(f"No se pudo lanzar {n}: {res.stderr.strip()}")
            t0[n] = time.monotonic()
            waiters[n] = subprocess.Popen(["docker", "wait", names[n]], stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, text=True)
        if faults:
            targets = {**names, "broker": opts.broker_container}
            injector = FaultInjector(list(faults), targets, opts.network, run_dir / "faults.jsonl",
                                     lambda node: _logs(names[node]))
            injector.start()
        deadline = time.monotonic() + timeout_s
        durations: dict[str, float] = {}
        codes: dict[str, int | None] = {}
        while time.monotonic() < deadline and len(durations) < len(waiters):
            for n, w in waiters.items():
                if n not in durations and w.poll() is not None:
                    durations[n] = time.monotonic() - t0[n]
                    out = (w.stdout.read() if w.stdout else "").strip()
                    codes[n] = int(out) if out.lstrip("-").isdigit() else None
            time.sleep(0.2)
        timed_out = [n for n in waiters if n not in durations]
        for n in timed_out:
            _docker("kill", names[n])
            waiters[n].wait(timeout=60)
            durations[n] = time.monotonic() - t0[n]
            codes[n] = None
        return RunResult(run_dir, codes, durations, timed_out)
    finally:
        if injector is not None:
            injector.stop()
            if any(f.target == "broker" for f in faults):
                wait_broker_healthy(opts)          # la limpieza del vhost usa la API del broker
        for n, name in names.items():
            logs = _docker("logs", "--timestamps", name, timeout=60)
            if logs.returncode == 0:
                (run_dir / "logs" / f"{n}.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
            _docker("rm", "-f", name)
        for w in waiters.values():
            if w.poll() is None:
                w.kill()
