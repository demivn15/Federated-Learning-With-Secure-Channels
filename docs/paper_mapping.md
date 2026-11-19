# Correspondencia paper (D0) → artefactos de la réplica

> Etapa E0. Para cada tabla o figura del paper base indica **qué se mide**, **con qué configuración**, **qué columnas de KPI** la regeneran y **qué limitación** tiene la comparación.
> Configuración de referencia: [`configs/variants/d0_reference.yaml`](../configs/variants/d0_reference.yaml). Análisis: [`analisis_arquitectonico/`](../../analisis_arquitectonico/01_document_analysis.md).

**Estudio "reproducción":** 9 configuraciones (3 topologías × FedAvg/FedProx/FedNova) + MLP centralizado; perfil de seguridad `S0`; semilla 42 (una corrida, como D0) y semillas 42–46 para los IC.

---

## Tablas

| Paper | Contenido | Valores de D0 (para la tolerancia) | Configuración / estudio | Columnas KPI o salida | Limitaciones de la comparación |
|---|---|---|---|---|---|
| **Tabla 1** | Partición por hospital | H1 7 739 / 5 657 +; H2 19 642 / 39 +; H3 279 / 167 +; H4 1 191 / 1 185 +; H5 10 232 / 143 +; total 39 083 / 7 191 + | `scripts/prepare_data.py` (sin entrenamiento) | Informe de partición: filas, positivos y % por hospital; hashes | Diferencia de positivos conocida: 7 184 locales vs 7 191 (I-02). Se reporta, no se corrige |
| **Tabla 2** | Configuración de las arquitecturas | C: 12 rondas; SD: 5–9; H: 30 globales; 5 CN + 1 servidor | `topologies.*` en la referencia | Manifiesto de cada corrida | En SD, las rondas son resultado de la parada (I-29) |
| **Tabla 3** | MLP centralizado por split | Test: Acc 0,905 · Prec 0,655 · Rec 0,968 · F1 0,781 | `evaluation.centralized_reference` (fuera de la Pi) | Métricas de train/validación/test | Inconsistente con la Tabla 7 (I-22) ⇒ se compara con ambas y se reporta cuál coincide |
| **Tabla 4** | Rotación de la lotería SD | FedAvg 5 RG (CN5: 2), FedProx 8 RG (CN3: 4), FedNova 5 RG (CN5: 4); CN1 nunca elegido | SD × 3 métodos, `selector: lottery_d0` | `sd_aggregator_id`, `sd_aggregator_changed`, `internal_rounds` | La distribución depende de la semilla de la lotería, que D0 no informa ⇒ se compara la **forma** (uniformidad esperada vs realizada) y la sensibilidad multi-semilla, no la secuencia exacta |
| **Tabla 5** | Métricas por CN en SD | Media macro F1: FedAvg 0,6645 · FedProx 0,6506 · FedNova 0,6677; CN1 y CN3 con recall 0,5000 | SD × 3 | `evaluation.scopes.per_client` | — |
| **Tabla 6** | F1 por CN en la jerárquica (últimas 5 rondas) | CN2 ≈ 0,49 y CN4 ≈ 0,49 en los 3 métodos; CN1 0,891/0,897/0,786 | H × 3 | `per_client` | — |
| **Tabla 7** | Resumen entre arquitecturas | C.FedAvg F1 0,661; C.FedNova 0,646; C.FedProx 0,669; SD† 0,791/0,768/0,775; H.FedAvg 0,789; H.FedNova 0,808; H.FedProx 0,725. Comm/ronda: C 1,69; SD 0,45; H 2,82–3,44 MB. E_cum (J): 6,09/0,90/2,23; 0,74/0,93/2,12; 21,45/15,35/18,53 | Las 9 configuraciones | Métricas con alcance `network` y `aggregator_node_sd` (para SD †); `payload_bytes_per_node`, `nic_bytes_per_node`; `e_cum` | (1) SD † se evalúa en el agregador ⇒ se reportan **las dos** vistas (I-23). (2) "Comm/ronda" no está definida en D0 (I-25) ⇒ se reporta con la definición explícita. (3) El modelo de 256/128/64 cambia `M` respecto del código local. (4) Transporte RabbitMQ vs WebSocket ⇒ la vista de NIC incluye el salto del broker; la vista de payload es la comparable. Además, D0 comprimía con `deflate` (I-30): sus bytes en la red pueden ser ≈ 8 % menores que el payload. (5) Energía de comunicación con `T_comm` medido (Q-14): incluye el broker |
| **Tabla 8** | Hardware medio (CN que entrenan, últimas 5 rondas) | CPU 1,2 / 4,5 / 3,4 %; Temp 43,1 / 44,9 / 47,2 °C; RAM 407,2 / 408,8 / 670,4 MB | Las 9 configuraciones | `cpu_percent`, `temperature_c`, `ram_mb` en los 5 CN | Erlang solo en la 6.ª Pi ⇒ los CN son comparables. La RAM puede variar por la librería AMQP del cliente ⇒ se reporta |
| **Tabla 9** | Straggler (media, σ, mín., cuartiles, máx.) | C: 5,563 / 5,501 / 7,263 s; SD: 5,050 / 8,354 (σ 9,418, máx. 35,317) / 6,323; H: 1,037 / 0,105 / 0,992 | Las 9 configuraciones | `straggler` (medido en el agregador) | El código local medía mal el straggler en P2 (P2-H1); aquí se mide con las horas de llegada |

## Figuras

| Paper | Contenido | Configuración | Salida | Notas |
|---|---|---|---|---|
| **Fig. 1** | Diagramas de las arquitecturas | — | `docs/topologies.md` | Solo descriptiva; la figura no se pudo extraer del PDF |
| **Fig. 2** | Recall global y pérdida por ronda (C 12 rondas; SD; H 30 rondas) | Las 9 configuraciones | Curvas por ronda desde los checkpoints (evaluador fuera de línea) | Mismo alcance de métrica para todas las topologías + vista SD † |
| **Fig. 3** | Energía acumulada por ronda | Las 9 configuraciones | `e_cum` por ronda | Energía analítica (Q-07); variante `measured_tcomm` |
| **Fig. 4** | Varianza del recall entre CN por ronda (H < 0,013 en la ronda 10; C 0,030–0,031 en la ronda 12; SD 0,018–0,027) | Las 9 configuraciones | `recall_variance_across_clients` | — |

## Resultados que **no** forman parte de la réplica (ampliación)

| Artefacto nuevo | Estudio | Relación con D0 |
|---|---|---|
| Tablas y figuras P2P | P2P × 3 métodos; malla / anillo / k aleatorio | 4.ª topología, comparada con las Tablas 7–9 |
| Coste de seguridad (desglose `t_*`) | S0 / S1 / S2 × 4 topologías | Añade la dimensión de seguridad (D3 Tabla E3) |
| Selección SD con salvaguardas | `lottery_d0` / `round_robin` / `sd_score_hybrid` | Responde a la recomendación de D0 §4.3 |
| Fallos inyectados, incluido el broker | 4 topologías | Mide R-05 |
| Informe de seguridad | Suite adversarial (E12) | Nuevo |
| IC multi-semilla | Semillas 42–46 | Cierra la limitación (i) de D0 |
| Diferencia de transporte | Micro-benchmark WebSocket vs AMQP | Explica las diferencias de latencia y bytes con D0 |
