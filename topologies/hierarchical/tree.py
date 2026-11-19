"""Árbol de la topología jerárquica (RF-02; D0 §3.3.4; 04 ADR-09).

Referencia D0: raíz en la 6.ª Pi; clúster A = {CN1*, CN2, CN3} y clúster B = {CN4*, CN5}; los marcados con
* entrenan y agregan su clúster ("dual role"). `clusters: flat` (variante `hierarchical_flat`) hace de
cada Client Node un clúster de un solo miembro: la raíz agrega los 5 modelos directamente, que es la
topología centralizada (oráculo de equivalencia de 04 §7).

Redistribución del modelo global (D0 no la especifica, 01 §1.15.7):
  * `via_cluster_aggregators`: raíz → agregadores de clúster → miembros (el árbol también hacia abajo);
  * `direct`: raíz → todos los Client Nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

REDISTRIBUTIONS = ("via_cluster_aggregators", "direct")
ROLES = ("root", "cluster_aggregator", "leaf")


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    aggregator: str
    members: tuple[str, ...]           # incluye al agregador

    @property
    def leaves(self) -> tuple[str, ...]:
        return tuple(m for m in self.members if m != self.aggregator)


@dataclass(frozen=True)
class Tree:
    root: str
    clusters: tuple[Cluster, ...]
    redistribution: str
    aggregators_train: bool
    flat: bool

    def cluster_of(self, node_id: str) -> Cluster:
        for c in self.clusters:
            if node_id in c.members:
                return c
        raise KeyError(f"{node_id} no pertenece a ningún clúster")

    def role_of(self, node_id: str) -> str:
        if node_id == self.root:
            return "root"
        return "cluster_aggregator" if self.cluster_of(node_id).aggregator == node_id else "leaf"

    def parent_of(self, node_id: str) -> str:
        """A quién envía su modelo el nodo (hoja → agregador de su clúster; agregador → raíz)."""
        c = self.cluster_of(node_id)
        return self.root if c.aggregator == node_id else c.aggregator

    def round_start_sender(self, node_id: str) -> str:
        """Quién envía el modelo global a este nodo, según la redistribución."""
        c = self.cluster_of(node_id)
        if self.redistribution == "direct" or c.aggregator == node_id:
            return self.root
        return c.aggregator

    @property
    def aggregators(self) -> tuple[str, ...]:
        return tuple(c.aggregator for c in self.clusters)


def build_tree(topo_cfg: Mapping[str, Any], clients: Iterable[str]) -> Tree:
    clients = sorted(clients)
    root = topo_cfg["root"]
    redistribution = topo_cfg.get("redistribution", "via_cluster_aggregators")
    if redistribution not in REDISTRIBUTIONS:
        raise ValueError(f"Redistribución desconocida: {redistribution!r} (opciones: {REDISTRIBUTIONS})")
    spec = topo_cfg["clusters"]
    if spec == "flat":
        clusters = tuple(Cluster(c, c, (c,)) for c in clients)
        return Tree(root, clusters, redistribution, True, True)
    if not isinstance(spec, Mapping) or not spec:
        raise ValueError("`clusters` debe ser 'flat' o un mapa {id: {aggregator, members}}")

    clusters, seen = [], {}
    for cid in sorted(spec):
        c = spec[cid]
        members = tuple(c["members"])
        agg = c["aggregator"]
        if agg not in members:
            raise ValueError(f"Clúster {cid}: el agregador {agg} debe ser miembro")
        if len(set(members)) != len(members):
            raise ValueError(f"Clúster {cid}: miembros repetidos")
        for m in members:
            if m == root:
                raise ValueError(f"Clúster {cid}: la raíz {root} no puede ser miembro")
            if m in seen:
                raise ValueError(f"{m} está en los clústeres {seen[m]} y {cid}")
            seen[m] = cid
        clusters.append(Cluster(str(cid), agg, members))
    missing, unknown = sorted(set(clients) - set(seen)), sorted(set(seen) - set(clients))
    if missing or unknown:
        raise ValueError(f"Clústeres no cubren los Client Nodes: faltan {missing}, desconocidos {unknown}")
    return Tree(root, tuple(clusters), redistribution, bool(topo_cfg.get("cluster_aggregators_train", True)), False)
