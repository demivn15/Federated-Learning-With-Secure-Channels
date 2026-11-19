# Federated Learning With Secure Channels

Benchmark de aprendizaje federado multitopología sobre **RabbitMQ**, con **canales seguros** medidos por capas.

Réplica y ampliación del estudio de Castro-Rodriguez et al. (ICI2ST 2026, «D0»): clasificación de mortalidad
prematura por enfermedades no transmisibles con 5 hospitales en dispositivos de borde (5 Raspberry Pi que entrenan
+ 1 de infraestructura).

- **Mismo núcleo FL en 4 topologías:** centralizada, jerárquica, semi-descentralizada y P2P.
- **Comunicación:** un único broker RabbitMQ.
- **3 perfiles de seguridad:** S0 sin seguridad, S1 TLS, S2 TLS + cifrado de extremo a extremo.
- **Mediciones:** KPIs de tiempo, red, CPU, RAM y energía; evaluación fuera de línea con el protocolo de D0.

## Estado

| Etapa | Contenido | Estado |
|---|---|---|
| E0–E4 | Referencia D0, núcleo FL, trama y serialización restringida, transporte AMQP, telemetría y manifiesto | ✅ |
| E5–E8 | Topologías centralizada, jerárquica, semi-descentralizada y P2P | ✅ |
| E9 | Fallos inyectados en contenedores (nodos y broker) | ✅ en Docker |
| E10 | Seguridad S1/S2 + conformidad criptográfica | ✅ en Docker |
| E11 | Validación en las Raspberry Pi y reproducción de D0 | Pendiente |
| E12–E13 | Suite adversarial y campaña experimental | Pendiente |

El plan completo y el estado detallado de cada etapa están en el análisis arquitectónico del proyecto
(`analisis_arquitectonico/06_implementation_roadmap.md`, fuera de este repositorio).

## Topologías

| Topología | Nodo de infraestructura (6.ª Pi) | Client Nodes (CN1–CN5) |
|---|---|---|
| Centralizada | Servidor: agrega (FedAvg, FedProx o FedNova) | Entrenan con su hospital |
| Jerárquica | Raíz | Clúster A = {CN1*, CN2, CN3}, B = {CN4*, CN5}; los marcados con * entrenan y agregan su clúster |
| Semi-descentralizada | Coordinador pasivo (nunca recibe modelos) | Lotería de D0: un CN agrega cada ronda; salvaguardas y `SD-score` opcionales |
| P2P | Solo el broker | DFedAvg con Metropolis ponderado; grafo en malla, anillo o *k* aleatorio |

## Perfiles de seguridad

| Perfil | Qué añade | Protege frente a |
|---|---|---|
| **S0** | Usuario y permisos mínimos por nodo en RabbitMQ (el broker valida `user_id`), *unpickler* restringido, contratos JSON Schema | Reproducción de D0 (sin cifrado) |
| **S1** | AMQPS: TLS nodo↔broker con verificación del certificado del broker | Escuchas y manipulación en la red |
| **S2** | S1 + sobre E2E por par:<br>• *handshake* con identidades ECDSA P-256 y ECDH efímero<br>• claves por mensaje con HKDF-SHA256 **indexado** (perder mensajes no desincroniza)<br>• AES-256-GCM con la cabecera como AAD<br>• ventana anti-replay<br>• nueva época en cada reconexión<br>• telemetría SD firmada | Además, un broker comprometido: no puede leer, alterar ni inventar mensajes |

S2 no oculta los metadatos: quién habla con quién, tamaños y tiempos. Es un diseño criptográfico propio, así que
requiere revisión externa antes de publicar resultados de S2.

## Estructura

```
core/          núcleo FL sin red: datos, modelo, entrenamiento, agregación (Eq. 1–3), parada
messaging/     trama, serialización restringida, contratos, transporte AMQP
  security/    perfil S2: primitivas, identidades, handshake, sobre, anti-replay, canal seguro
topologies/    orquestadores de las 4 topologías (base.py = contexto común del nodo)
node/          punto de entrada de un nodo (python -m node.main)
telemetry/     reloj, KPIs, energía analítica, eventos
experiments/   manifiesto, ejecutor local y en Docker, inyección de fallos, evaluación, estudios
broker/        aprovisionamiento por experimento (vhost, colas, usuarios y permisos)
schemas/       contratos JSON Schema de cada mensaje
configs/       configuración de referencia D0 y variantes heredadas
scripts/       preparar datos, lanzar experimentos, identidades/TLS, benchmarks
docker/        broker RabbitMQ (compose) e imagen de los nodos
docs/          reproducibilidad, mapeo con el paper, serialización, coste de la seguridad
tests/         unidad, arquitectura, equivalencias, conformidad, seguridad, integración
```

## Instalación

Requisitos: Python ≥ 3.10 y Docker Desktop (para el broker y los nodos en contenedores).

```powershell
python -m venv .venv
# Mismas versiones que la imagen de los nodos (el proyecto se ejecuta desde esta carpeta, no se instala como paquete)
.venv\Scripts\python.exe -m pip install -r docker\requirements-node.txt pytest pytest-asyncio

# Datos: reconstruye la partición de D0 a partir de los silos del proyecto jerárquico base
.venv\Scripts\python.exe scripts\prepare_data.py --silo-dir ..\FL_Hierarchical_Hospital\data\silos

# Broker (una vez): credenciales, certificados TLS, imagen de los nodos y arranque
.venv\Scripts\python.exe scripts\broker_env.py
.venv\Scripts\python.exe scripts\provision_identities.py tls
docker build -f docker\node.Dockerfile -t fl-node:dev docker
docker compose -f docker\compose.yaml up -d
```

El broker publica sus puertos solo en `127.0.0.1`:

| Puerto | Uso |
|---|---|
| 5673 | AMQP (S0) |
| 5674 | AMQPS (S1/S2) |
| 15673 | Consola y API de gestión |

No son los puertos estándar para no chocar con un RabbitMQ nativo de Windows.

`docker/.env`, `docker/tls/`, `data/` y `results/` no se versionan: contienen secretos o se regeneran.

## Uso

```powershell
# Un experimento con los nodos en contenedores (un contenedor por nodo en la red fl-net)
.venv\Scripts\python.exe scripts\run_experiment.py --topology centralized --method fedavg `
    --launcher docker --security S2 --rounds 6 --epochs 1 --threads 1 --cpus 1 --id prueba-c-s2

# Lo mismo con un proceso local por nodo
.venv\Scripts\python.exe scripts\run_experiment.py --topology p2p --method fedavg --security S0 --epochs 1
```

| Opción | Valores |
|---|---|
| `--topology` | `centralized`, `hierarchical`, `semi_decentralized`, `p2p` |
| `--method` | `fedavg`, `fedprox`, `fednova` |
| `--security` | `S0`, `S1`, `S2` |
| `--launcher` | `local`, `docker` |
| `--rounds`, `--epochs` | Si se omiten, se usa la referencia de D0 |
| `--cpus`, `--memory` | Límites por contenedor |

Cada corrida queda en `results/runs/<id>/`:

| Archivo | Contenido |
|---|---|
| `manifest.json` | Configuración efectiva, hashes de datos y código, versiones, perfil |
| `kpis_all.csv` | KPIs por nodo y ronda |
| `evaluation_summary.json` | Métricas con el protocolo de D0 |
| `logs/` | Log de cada nodo |
| `nodes/<nodo>/events.jsonl` | Eventos del broker y salida |
| `checkpoints/` | Modelos por ronda |

Cada nodo verifica al arrancar que el código y los datos son los del manifiesto: **no edites el código mientras
corre un experimento**.

## Pruebas

```powershell
.venv\Scripts\python.exe -m pytest -q -m "not broker"                        # unidad, conformidad, canal seguro (~1 min)
.venv\Scripts\python.exe -m pytest -m "broker and not docker"                 # integración con el broker y procesos locales
.venv\Scripts\python.exe -m pytest -m docker tests\integration\test_faults_docker.py            # E9 (~20 min)
.venv\Scripts\python.exe -m pytest -m docker tests\integration\test_security_matrix_docker.py   # E10: 4 topologías × S0/S1/S2
```

| Suite | Qué comprueba |
|---|---|
| `tests/conformance/` | C1–C5: vectores RFC 5869 (HKDF), GCM AES-256, RFC 5903 (ECDH), RFC 6979 (ECDSA); vinculación del transcript; 10⁶ claves sin repetición; pérdidas sin desincronización; replay rechazado al 100 % |
| `tests/security/` | Canal S2 frente a MITM en el *handshake*, replay, manipulación, *downgrade* a texto plano y suplantación; `pickle` malicioso |
| `tests/integration/test_faults_docker.py` | Caída y reinicio del broker en las 4 topologías (con el tiempo de detección), nodo muerto o pausado, partición de red, latencia y pérdida con `netem`, cola llena |

## Resultados en Docker (2026-11-19)

**Coste de la seguridad.** 36 corridas: 4 topologías × S0/S1/S2 × 3 semillas, 1 CPU por nodo, versión rápida
(1 época). Detalle en [`docs/security_cost_docker.md`](docs/security_cost_docker.md).

| | Centralizada | Jerárquica | Semi-desc. | P2P |
|---|---|---|---|---|
| Tiempo por ronda S0 → S2 | 1,63 → 1,61 s | 1,61 → 1,60 s | 7,27 → 7,31 s | 1,46 → 1,47 s |
| Confirmación de publicación S0 / S1 / S2 | 3,1 / 5,0 / 5,0 ms | 3,0 / 5,4 / 5,0 ms | 2,5 / 4,2 / 4,2 ms | 6,4 / 11,4 / 10,6 ms |
| Cifrado + descifrado S2 por nodo y ronda | 1,5 ms | 1,4 ms | 1,2 ms | 3,1 ms |
| Bytes en la red S1 / S2 vs S0 | +0,1 / +0,2 % | +0,1 / +0,2 % | +0,1 / +0,2 % | +0,1 / +0,2 % |
| F1 de red (S0 = S1 = S2) | 0,5483 | 0,5762 | 0,7236 | 0,5487 |

- S1 y S2 no tienen un coste medible por ronda: < 2 %, dentro del ruido; el objetivo de D3 es ≤ 25 %.
- El efecto visible es TLS: unos +2 ms por modelo publicado.
- La calidad del modelo es idéntica en los tres perfiles.
- En las Pi el cifrado será más caro, porque el BCM2711 no tiene extensiones criptográficas ARMv8. Se mide en E11.

**Tolerancia a fallos (E9).**
- **Caída del broker:** los nodos la detectan en unos 50–80 ms y terminan limpiamente (código 6) al vencer
  `loss_grace_s`.
- **Reinicio corto del broker:** se recupera; en S2, con nuevo *handshake*.
- **Partición de red:** se detecta por *heartbeat* (unos 30 s) y el nodo se reincorpora.
- **Fallos de un nodo:** un nodo caído, pausado o con la cola llena no bloquea a los demás más que su timeout.

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/reproducibility.md`](docs/reproducibility.md) | Versiones, tolerancias, supuestos declarados y discrepancias conocidas con D0 |
| [`docs/paper_mapping.md`](docs/paper_mapping.md) | Cada tabla y figura de D0 → su artefacto |
| [`docs/serialization_rationale.md`](docs/serialization_rationale.md) | Por qué `pickle` restringido y no JSON/XML para los parámetros |
| [`docs/security_cost_docker.md`](docs/security_cost_docker.md) | Estudio S0/S1/S2 en contenedores |
