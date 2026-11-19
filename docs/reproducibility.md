# Reproducibilidad — versiones, tolerancias y supuestos

> Etapa E0. Complementa [`configs/variants/d0_reference.yaml`](../configs/variants/d0_reference.yaml) y [`paper_mapping.md`](paper_mapping.md).

---

## 1. Versiones de software

### 1.1 Qué se sabe de la línea base

| Elemento | Evidencia | Conclusión |
|---|---|---|
| Python | D0 §3.3: 3.10. `Dockerfile` de P1: `python:3.10-slim`. Logs de P2 en las Pi: **3.13** (456 apariciones de `python3.13` en `logs/*.log`) | Las Pi tenían 3.13 en las corridas de prueba de P2. La referencia es **3.10** |
| PyTorch | `requirements.txt`: `torch>=2.0.0` (P1, P3); sin versión (P2) | **Versión exacta desconocida** |
| websockets | `>=11.0` (P1, P3); `>=14.0` (P2) | Irrelevante: se sustituye por RabbitMQ |
| numpy / pandas / joblib / psutil | Solo mínimos (`>=`) | **Versiones exactas desconocidas** |
| scikit-learn | Mínimo `>=1.3`; pero `FL_Centralized_Hospital/artifacts/preprocessor_global.joblib` fue **serializado con scikit-learn 1.8.0** (advertencia `InconsistentVersionWarning` al cargarlo con 1.9.1) | Única versión exacta recuperada del proyecto base. Se usará **1.8.0** si hay ruedas para Python 3.10 en aarch64 |

**Conclusión:** el código base no fija versiones, así que **no se pueden recuperar** las que usó el paper. Se fijan nuevas versiones y se declaran. El riesgo es que diferencias de versión afecten ligeramente a los números; queda cubierto por la tolerancia (§2).

### 1.2 Versiones a fijar (pendiente de verificación en las Pi)

| Componente | Requisito | Cómo se decide |
|---|---|---|
| Python | 3.10 | D0 |
| PyTorch | Rueda CPU para aarch64 compatible con Python 3.10 | La que se instale en la Pi; se anota la versión exacta |
| scikit-learn | ≥ 1.3 (mínimo del código) | Debe ser la **misma** versión al generar el preprocesador y al usarlo |
| numpy, pandas, joblib, psutil | Mínimos del código | Se anotan las instaladas |
| Cliente AMQP (`aio-pika`) | Compatible con Python 3.10 | 04 ADR-03 |
| `cryptography`, `jsonschema` | Ruedas aarch64 disponibles | Solo para los perfiles S1/S2 y los contratos |
| RabbitMQ + Erlang | Versión con paquetes disponibles para el SO de la 6.ª Pi; D3 propone RabbitMQ 3.12 | Se verifica en la Pi; Erlang debe ser compatible con esa versión de RabbitMQ |
| SO de las Pi | — | Se anota (D3 sugiere Raspberry Pi OS Lite 64-bit) |

### 1.3 Comandos para registrar el estado actual de cada Pi

Ejecutar en **cada** Pi y guardar la salida en `results/environment/<hostname>.txt`. No modifican nada:

```bash
hostname; cat /etc/os-release | head -4; uname -m
cat /proc/device-tree/model; echo
free -m | head -2
python3 --version; ls /usr/bin/python3.* 2>/dev/null
pip3 list 2>/dev/null | grep -iE "torch|numpy|pandas|scikit|joblib|psutil|websockets|pika|cryptography|jsonschema"
vcgencmd measure_temp 2>/dev/null; vcgencmd get_config arm_freq 2>/dev/null
```

En la 6.ª Pi, además:

```bash
rabbitmqctl version 2>/dev/null || echo "RabbitMQ no instalado"
erl -noshell -eval 'io:format("~s~n",[erlang:system_info(otp_release)]), halt().' 2>/dev/null || echo "Erlang no instalado"
```

Con esa salida se rellena la tabla de §1.2 y se fijan los requisitos de `pyproject.toml`.

---

## 2. Tolerancias de reproducción

D0 es **una sola ejecución** por configuración (D0 §3.7), así que sus cifras traen ruido de ejecución única. Una configuración se considera **reproducida** si cumple:

| Magnitud | Criterio | Fuente |
|---|---|---|
| F1, accuracy, recall (media de las últimas 5 rondas) | El valor de D0 cae dentro del IC del 95 % de las 5 semillas **o** a ±0,03 del valor de la semilla 42 | D3 §9.1 + multi-semilla |
| CPU, RAM, temperatura de los CN que entrenan | ±15 % respecto de la Tabla 8 | D3 §9.1 |
| Bytes de payload por nodo y ronda | Explicables con el tamaño `M` del modelo y el número de mensajes; diferencia declarada frente a "Comm/ronda" de D0 | I-25 (D0 no la define) |
| Straggler | Mismo orden de magnitud y ordenación entre topologías que la Tabla 9; diferencias explicadas con el micro-benchmark de transporte | Transporte distinto (RabbitMQ) |
| Energía analítica | Recalculada con las constantes de D0; se compara la **tendencia** (Fig. 3; FedNova < FedAvg < FedProx dentro de cada arquitectura) | `E_comm` depende del `T_comm` medido (Q-14) |
| Rondas | C = 12 y H = 30 por construcción; SD dentro de 5–9 rondas globales | Tabla 2 + respuesta del usuario |

**Si una configuración no se reproduce:** se repite con las variantes heredadas (`legacy_variants` en la referencia) para ver si la diferencia se explica por el código local (I-24). La discrepancia se documenta, no se ajusta a mano.

---

## 3. Supuestos declarados (confirmar con los autores antes de publicar)

| ID | Supuesto | Dónde se usa |
|---|---|---|
| Q-14 | `E_comm = p · T_comm` medido. La Eq. 5 está implementada pero no se reporta hasta conocer `B`, `h`, `σ²` | `energy.e_comm` |
| Q-15 | *Holdout* = unión de los test (20 %) estratificados por hospital; umbral F1 en la unión de las validaciones | `data.split`, `evaluation` |
| Q-16 | C 12 y H 30 rondas fijas; SD con *patience* 3 (mejora mínima 0,001; máx. 50); Adam persistente por nodo | `topologies.*.rounds`, `training.optimizer_state` |
| — | Semilla de la lotería SD = 42 (D0 no la informa) | `semi_decentralized.selector_seed` |
| — | Redistribución jerárquica a través de los agregadores de clúster (D0 no lo especifica) | `hierarchical.redistribution` |
| — | Épocas, optimizador, *lr*, batch y pérdida tomados del código (D0 no los informa) | `training` |
| — | Preprocesador ajustado con la unión de los train de los 5 hospitales (la cohorte nacional no está disponible) ⇒ **278 features**. Con la cohorte nacional podrían aparecer más categorías de cantón y salir algo más de 278 | `data.preprocessing.fit_on_local` |

## 3.1 Broker de desarrollo y transporte (E3)

**Broker local:**
- `python scripts/broker_env.py` genera las credenciales de administrador en `docker/.env` (no se versiona).
- `docker compose -f docker/compose.yaml up -d` arranca **RabbitMQ 3.13.7 / Erlang 26.2.5.16**.
- Los puertos en el PC son **5673** (AMQP) y **15673** (API de gestión), porque el PC de desarrollo tiene un servicio RabbitMQ nativo de Windows en 5672/15672.
- En las Pi se usan los puertos estándar.

**Aprovisionamiento por experimento:**
- `python broker/provision_broker.py create <id>` crea un vhost propio, exchanges, colas y un usuario por nodo con permisos mínimos.
- `... delete <id>` lo elimina todo.

**Verificado en E3 contra el broker real** (`tests/integration/test_amqp_transport.py`):
- Un nodo no puede leer la cola de otro ni declarar colas o exchanges.
- RabbitMQ rechaza un `user_id` que no sea el del usuario autenticado ⇒ identidad del emisor validada por el broker.
- Cola llena (`reject-publish`) ⇒ el emisor recibe el rechazo.
- Destino inexistente ⇒ error.
- Mensaje de más de 4 MiB ⇒ rechazado por el cliente y por el broker.
- Los mensajes pendientes esperan en la cola durante la desconexión y caducan según el TTL.
- Un mensaje rechazado no se reencola.
- Memoria del broker: 152–165 MiB con 20 modelos encolados (límite 512 MiB).

**Micro-benchmark de transporte** (`scripts/bench_transport.py`; resultado en `results/bench/`):

| Transporte (eco de una trama UPDATE de 453 904 B) | RTT mediana | p95 |
|---|---|---|
| WebSocket con `deflate` (**configuración real de D0**: valor por defecto de `websockets`, que P1–P3 no cambian) | 43,2 ms | 45,3 ms |
| WebSocket sin compresión | 3,6 ms | 25,7 ms |
| AMQP vía RabbitMQ (4 tramos) | 26,0 ms | 29,3 ms |

*PC de desarrollo, n = 100. Comparación sesgada: el broker corre en la VM de Docker Desktop. Los valores que cuentan son los de las Pi (E11).*

**Hallazgo:** el transporte de D0 **comprimía** cada modelo con `deflate`. Ahorra solo un 8 % de bytes (los pesos `float32` apenas se comprimen) y cuesta ≈ 15–19 ms de CPU por compresión en el PC; en la Pi costará más. Implicaciones para comparar con D0:
- Bytes por mensaje: RabbitMQ envía la trama completa, sin compresión.
- Latencia y straggler: D0 incluía el coste de comprimir y descomprimir; RabbitMQ añade el salto del broker.

## 3.2 Telemetría y evaluación (E4)

| KPI | Semántica heredada (para comparar con D0) | Semántica añadida |
|---|---|---|
| CPU | `cpu_percent_legacy_before/after`: CPU de todo el sistema durante 0,2 s, antes de recibir el modelo y después de evaluar (nunca durante el entrenamiento ⇒ valores bajos como los de la Tabla 8) | `cpu_process_*`: CPU del proceso durante la ronda (I-31) |
| RAM | `ram_rss_mb_*`: RSS del proceso (igual que el código base) | — |
| `E_comp` | Eq. 4 con D = muestras de entrenamiento × features × 32 bits, una vez por ronda (verificado frente a `calc_ecomp` de P1) | — |
| `E_comm` | `e_comm_legacy_j` = p · descarga del modelo global (lo que suma el servidor de P1 en `E_cum`). Con AMQP se aproxima por la publicación servidor→broker hacia cada cliente | `e_comm_transfer_j` = p · subida del cliente hasta la confirmación del broker (≈ `send_time` de P1); `e_comm_eq5_j` solo con B, h, σ² confirmados (I-32, Q-14) |
| Straggler | — (el KPI de P2 era inválido, P2-H1) | Última − primera llegada, medido en el agregador con su reloj |

**Reloj:** `time.perf_counter_ns()`. En E4 se detectó que `time.monotonic_ns()` tiene ≈ 15,6 ms de resolución en Windows, demasiado gruesa para medir stragglers de 0–1 s como los de la jerárquica en D0.

**Evaluación fuera de línea** (`python -m experiments.evaluate_run <run_dir>`):
- El evaluador aplica el protocolo de D0 a los checkpoints: umbral F1 en la validación común, métricas en el test común, por hospital, varianza del recall entre hospitales y alcance del agregador en SD.
- Resume con la media de las últimas 5 rondas.
- Verificado: sus métricas coinciden exactamente con las calculadas a mano para un modelo conocido.
- Está en `experiments/` y no en `evaluation/` (como preveía 05), porque lee checkpoints serializados y la regla de capas mantiene `evaluation/` libre de red y de `pickle`.

## 3.3 Primera comparación con D0: topología centralizada (E5, PC de desarrollo)

Configuración de referencia completa (12 rondas, 3 épocas, MLP 256/128/64, partición de la Tabla 1), **una semilla (42)**, 6 procesos en el PC + RabbitMQ en Docker. Comando: `python scripts/compare_paper.py results/runs/ref-c-*-s42`.

| | F1 red | F1 heredada | F1 D0 | Acc red | Acc D0 | RAM CN (MB) | RAM D0 | Straggler (s) | D0 |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg | 0,609 | 0,563 | 0,661 | 0,685 | 0,854 | 407 | 407,2 | 6,84 | 5,56 |
| FedProx | 0,785 | 0,649 | 0,669 | 0,844 | 0,851 | 409 | 407,2 | 8,89 | 7,26 |
| FedNova | 0,710 | 0,634 | 0,646 | 0,778 | 0,848 | 407 | 407,2 | 4,96 | 5,50 |

**Lectura:**
- **RAM:** coincide con D0 (407 MB) ⇒ el modelo de 256/128/64 y la pila son los del paper.
- **Straggler:** mismo orden de magnitud y misma ordenación que D0 (FedProx el más lento, FedNova el más rápido), aunque se midió en el PC.
- **F1/accuracy:** fuera de la tolerancia de ±0,03 en una sola semilla. FedProx coincide en accuracy (0,844 frente a 0,851). FedAvg y FedNova **oscilan entre rondas** (FedAvg 0,50–0,67; sigue mejorando en la ronda 12), así que la media de las últimas 5 rondas depende mucho de la semilla. **No se concluye** hasta la campaña multi-semilla (E13) y hasta confirmar Q-15 (composición del test común) y el ajuste del preprocesador (cohorte nacional).
- **CPU heredada:** 52 % frente a 1,2 % en D0. No es comparable: en el PC los 6 nodos comparten CPU y la muestra es de todo el sistema. En la Pi cada nodo está solo.
- **Comunicación:** 2,27 MB/ronda de subida (5 × 454 KB) frente a 1,69 MB en D0 (I-25: D0 no define la métrica).
- **E_cum:** 45 J frente a 0,90–6,09 J en D0. Ver I-33 / Q-17: la diferencia entre métodos en D0 no la explica la Eq. 4.

## 3.4 Primera comparación con D0: topología jerárquica (E6, PC de desarrollo)

Configuración de referencia completa (30 rondas, 3 épocas, árbol de D0 A = {CN1*, CN2, CN3}, B = {CN4*, CN5}, redistribución vía agregadores de clúster), **una semilla (42)**, 6 procesos en el PC + RabbitMQ en Docker. Corridas: `results/runs/ref-h-*-s42` (≈ 4 min cada una).

| | F1 red | F1 D0 | Acc red | Acc D0 | Rec red | Rec D0 | F1 heredada | RAM CN (MB) | RAM D0 | Straggler raíz / clúster (s) | D0 | MB subida/ronda | D0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FedAvg | 0,717 | 0,789 | 0,804 | 0,924 | 0,755 | 0,753 | 0,521 | 411 | 670,4 | 3,06 / 4,93 | 1,037 | 3,63 | 2,98 |
| FedProx | 0,783 | 0,725 | 0,839 | 0,860 | 0,865 | 0,726 | 0,644 | 413 | 670,4 | 4,08 / 7,20 | 0,992 | 3,63 | 3,44 |
| FedNova | 0,748 | 0,808 | 0,808 | 0,941 | 0,830 | 0,766 | 0,642 | 411 | 670,4 | 2,42 / 4,02 | 0,105 | 3,63 | 2,82 |

**Lectura:**
- **Estancamiento por sesgo de etiqueta (Tabla 6): se reproduce cualitativamente.** F1 en el test de H2 ≈ 0,45–0,46 y de H4 ≈ 0,40–0,47 en los tres métodos (D0: CN2 ≈ 0,49 y CN4 ≈ 0,49); H1 0,70–0,86 (D0: 0,79–0,90).
- **Equidad (Fig. 4):** varianza del recall entre hospitales 0,002–0,006, por debajo del 0,013 de D0 ⇒ misma conclusión (la jerárquica es la más equitativa).
- **F1/accuracy:** fuera de la tolerancia en una semilla, y la ordenación entre métodos difiere (D0: FedNova > FedAvg > FedProx; aquí FedProx > FedNova > FedAvg). Como en la centralizada (§3.3), **no se concluye** hasta la campaña multi-semilla (E13).
- **RAM: 411 MB frente a 670 MB en D0.** Los nodos de esta réplica usan la misma pila en las 4 topologías; los 670 MB de D0 en la jerárquica proceden de la implementación P2 (servidores efímeros, `torch.save`, archivos recibidos por ronda), no de la topología. Es una diferencia de **implementación** (I-24) que esta arquitectura elimina por construcción (ADR-01).
- **Straggler:** mayor que en D0 y medido en dos niveles. En el PC los 6 procesos compiten por la CPU y el agregador de clúster también entrena (su modelo cuenta como una llegada). D0 no dice en qué nivel lo midió ⇒ se reportan ambos; la cifra comparable es la de las Pi (E11).
- **Comunicación:** 3,63 MB/ronda (5 subidas + 3 redistribuciones de los agregadores de clúster), dentro del rango de D0 (2,82–3,44 MB), que también es mayor que el de la centralizada.

## 3.5 Semi-descentralizada y P2P (E7, E8)

Implementadas y verificadas con las pruebas de integración (`tests/integration/test_semi_decentralized_e2e.py`, `test_p2p_e2e.py`). Las corridas de referencia de SD (parada por *patience*, 5 rondas internas) y P2P se ejecutan en E13; su comparación con D0 se añadirá entonces.

## 4. Discrepancias conocidas con D0 (no se corrigen, se reportan)

| ID | Discrepancia |
|---|---|
| I-02 | 7 184 positivos locales vs 7 191 en la Tabla 1 |
| I-22 | MLP centralizado: Tabla 3 ≠ Tabla 7 |
| I-23 | SD † evaluado en el agregador; las demás topologías con alcance de red |
| I-25 | "Comm/ronda" sin definición en D0 |
| Transporte | RabbitMQ (dos saltos) vs WebSocket (uno) |
| I-30 | D0 comprimía cada modelo con `deflate` (−8 % de bytes, coste de CPU); RabbitMQ no comprime |
| I-31 | El "CPU %" de D0 se muestreaba fuera del entrenamiento |
| I-32 | Dos `T_comm` distintos en el código base (descarga en el servidor vs subida en el cliente); la descarga se aproxima con AMQP |
