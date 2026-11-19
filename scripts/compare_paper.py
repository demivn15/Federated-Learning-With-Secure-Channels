"""Compara corridas de réplica con las Tablas 7–9 del paper (D0).

    python scripts/compare_paper.py results/runs/ref-c-fedavg-s42 results/runs/ref-c-fedprox-s42 ...

Para cada corrida informa:
  * Rendimiento (media de las últimas 5 rondas): alcance de red (protocolo de D0 §3.7) y métrica heredada
    (media local previa a la agregación), frente a la Tabla 7. SD † se compara con el alcance del agregador.
  * Hardware de los Client Nodes (últimas 5 rondas): CPU heredada, RAM y temperatura, frente a la Tabla 8.
  * Straggler medido en el servidor frente a la Tabla 9.
  * Comunicación por ronda en dos definiciones (D0 no la define, I-25).
  * E_cum: Eq. 4 a la frecuencia nominal de la Pi (1,5 GHz) + E_comm de subida (I-32).
La tolerancia de reproducción es ±0,03 en F1 (docs/reproducibility.md §2); los valores fuera de ella se marcan.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.manifest import load_manifest  # noqa: E402
from experiments.paper_values import TABLE7, TABLE8, TABLE9  # noqa: E402
from telemetry.energy import e_comp  # noqa: E402

F1_TOL = 0.03
NOMINAL_HZ = 1.5e9


def compare(run_dir: Path) -> dict:
    m = load_manifest(run_dir)
    cfg, topo, method = m["config"], m["topology"], m["method"]
    summ = json.loads((run_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    k = pd.read_csv(run_dir / "kpis_all.csv")
    # Nodos que entrenan y nodo que mide el straggler, según la topología (C: server; H: root; SD: aggregator)
    clients = k[k["role"].isin(["client", "leaf", "cluster_aggregator", "worker", "peer"])]
    server = k[k["role"].isin(["server", "root", "aggregator"])]
    last5 = sorted(k["round"].unique())[-5:]
    cl5 = clients[clients["round"].isin(last5)]
    paper7 = TABLE7.get((topo, method), {})
    net = summ.get("global", summ.get("mixed", {})).get("mean_last_5", {})
    legacy = summ.get("legacy_local", {}).get("mean_last_5", {})

    ec = cfg["energy"]["e_comp"]
    feats = cfg["data"]["preprocessing"]["expected_n_features"]
    e_nominal = sum(e_comp(int(r.n_samples), feats, NOMINAL_HZ, ec["alpha"], ec["c_cycles_per_bit"])
                    for r in clients.itertuples()) + clients["e_comm_transfer_j"].fillna(0).sum()
    rounds = k["round"].nunique()
    return {
        "run": run_dir.name, "topology": topo, "method": method, "rounds": rounds,
        "f1_net": net.get("net_f1"), "acc_net": net.get("net_accuracy"), "rec_net": net.get("net_recall"),
        "f1_legacy": legacy.get("legacy_local_f1"), "acc_legacy": legacy.get("legacy_local_accuracy"),
        "rec_legacy": legacy.get("legacy_local_recall"),
        "f1_paper": paper7.get("f1"), "acc_paper": paper7.get("accuracy"), "rec_paper": paper7.get("recall"),
        "cpu_legacy": cl5["cpu_percent_legacy_before"].mean(), "cpu_process_core": cl5["cpu_process_percent_core"].mean(),
        "ram_mb": cl5["ram_rss_mb_after"].mean(), "temp_c": cl5["temperature_c"].mean(),
        "cpu_paper": TABLE8.get(topo, {}).get("cpu_percent"), "ram_paper": TABLE8.get(topo, {}).get("ram_mb"),
        "straggler_mean": server["straggler_s"].mean(), "straggler_std": server["straggler_s"].std(),
        "straggler_paper": TABLE9.get((topo, method), {}).get("mean"),
        "comm_upload_mb": clients.groupby("round")["payload_bytes_tx"].sum().mean() / 1e6,
        "comm_up_down_mb": (clients.groupby("round")["payload_bytes_tx"].sum()
                            + clients.groupby("round")["payload_bytes_rx"].sum()).mean() / 1e6,
        "comm_paper_mb": paper7.get("comm_mb"),
        "e_cum_nominal_j": e_nominal, "e_cum_paper_j": paper7.get("e_cum_j"),
    }


def fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", help="CSV de salida")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")          # la consola de Windows usa cp1252 (sin Δ ni ✗)
    rows = [compare(Path(r)) for r in args.runs]
    print(f"{'corrida':<22}{'F1 red':>8}{'F1 hered.':>10}{'F1 paper':>9}{'Δ red':>8}{'Acc red':>9}{'Acc pap.':>9}"
          f"{'Rec red':>9}{'Rec pap.':>9}")
    for r in rows:
        d = None if r["f1_paper"] is None else r["f1_net"] - r["f1_paper"]
        flag = "" if d is None or abs(d) <= F1_TOL else " ✗"
        print(f"{r['run']:<22}{fmt(r['f1_net']):>8}{fmt(r['f1_legacy']):>10}{fmt(r['f1_paper']):>9}"
              f"{(fmt(d) + flag) if d is not None else '—':>8}{fmt(r['acc_net']):>9}{fmt(r['acc_paper']):>9}"
              f"{fmt(r['rec_net']):>9}{fmt(r['rec_paper']):>9}")
    print(f"\n{'corrida':<22}{'CPU her.':>9}{'CPU proc':>9}{'CPU pap.':>9}{'RAM':>8}{'RAM pap.':>9}"
          f"{'Strag.':>8}{'Str.pap':>8}{'MB sub.':>8}{'MB s+b':>8}{'MB pap.':>8}{'Ecum J':>8}{'Ecum pap':>9}")
    for r in rows:
        print(f"{r['run']:<22}{fmt(r['cpu_legacy'],1):>9}{fmt(r['cpu_process_core'],0):>9}{fmt(r['cpu_paper'],1):>9}"
              f"{fmt(r['ram_mb'],0):>8}{fmt(r['ram_paper'],0):>9}{fmt(r['straggler_mean'],2):>8}"
              f"{fmt(r['straggler_paper'],2):>8}{fmt(r['comm_upload_mb'],2):>8}{fmt(r['comm_up_down_mb'],2):>8}"
              f"{fmt(r['comm_paper_mb'],2):>8}{fmt(r['e_cum_nominal_j'],1):>8}{fmt(r['e_cum_paper_j'],2):>9}")
    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
