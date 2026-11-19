# Coste de la seguridad S0 / S1 / S2 en Docker (2026-11-19)

Primera medición del estudio 3 de ADR-10 (RNF-05), **en contenedores y antes de las Raspberry Pi**.

- **Comando:** `python -m experiments.security_study run --quick --seeds 42 43 44 --cpus 1`.
- **Salidas:** tablas completas en `results/studies/security_docker/report.md`; datos por corrida en `runs.csv` y `summary.csv`.

## Diseño

| Aspecto | Valor |
|---|---|
| Factor | Perfil de seguridad: **S0** (AMQP en claro) · **S1** (AMQPS: TLS nodo↔broker) · **S2** (S1 + sobre E2E por par: handshake ECDSA/ECDH, HKDF indexado, AES-256-GCM, anti-replay) |
| Fijo | Topología, método FedAvg, datos de D0, 1 hilo de PyTorch y **1 CPU por contenedor** (`--cpus 1`) |
| Topologías | Centralizada, jerárquica, semi-descentralizada y P2P |
| Semillas | 42, 43, 44 (n = 3 por celda): **36 corridas, todas completas** |
| Versión rápida | 1 época; rondas fijas en todas las topologías:<br>• C y H: 6 rondas<br>• SD: 3 rondas externas × 5 internas<br>• P2P: 6 rondas (patience desactivada) |
| Orden | El orden de los perfiles rota con la semilla, para no confundir el perfil con la deriva del PC |
| Estadística | Media e IC bootstrap del 95 % entre semillas.<br>Δ% = sobrecoste **pareado** frente a S0 con la misma topología y semilla. |

## Resultados principales

| Métrica | C | H | SD | P2P | Lectura |
|---|---|---|---|---|---|
| Tiempo por ronda, S2 vs S0 | −1,3 % | −0,8 % | +0,5 % | +0,9 % | **Dentro del ruido** (los IC se solapan). Objetivo de D3: ≤ 25 % por ronda ⇒ se cumple con margen |
| Duración del experimento, S2 vs S0 | −0,3 % | −0,1 % | +0,8 % | +1,9 % | Ídem |
| Publicación hasta la confirmación (S1, S2 vs S0) | +66 %, +64 % | +83 %, +69 % | +68 %, +69 % | +78 %, +66 % | **El único efecto claro es TLS:** ≈ 3 → 5 ms por modelo de 450 KB (+2 ms; P2P: +5 ms). S2 no añade nada sobre S1 |
| Cifrado + descifrado E2E (S2) por nodo y ronda | 1,5 ms | 1,4 ms | 1,2 ms | 3,1 ms | Proporcional al nº de mensajes con modelo por nodo (en P2P, 4 vecinos) |
| Handshakes (S2) | 10 · 51 ms | 16 · 70 ms | 24 · 92 ms | 20 · 6,5 s* | Una vez por par y época, solo en la ronda 1 |
| Bytes de trama, S2 vs S0 | +0,0 % | +0,0 % | +0,0 % | −0,0 % | Cabecera externa + etiqueta GCM ≈ 200 B por mensaje de ≈ 454 KB |
| Bytes en la interfaz (TX+RX), S1 / S2 | +0,1 / +0,2 % | +0,1 / +0,2 % | +0,1 / +0,2 % | +0,1 / +0,2 % | Registros TLS + sobre E2E + mensajes del handshake |
| CPU del proceso (% de núcleo) | ≈ 0 % | ≈ 0 % | −1 % | ≈ 0 % | Domina el entrenamiento |
| RAM RSS, S1 / S2 | +0,6 / +1,5 % | +0,5 / +1,3 % | +0,5 / +1,5 % | +0,6 / +1,3 % | ≈ +3 MB (TLS) y ≈ +7 MB (biblioteca `cryptography` + sesiones) sobre ≈ 461 MB |
| E_comm (p·T_comm de la subida) | +52 / +47 % | +83 / +68 % | +68 / +69 % | +78 / +66 % | Hereda el aumento de T_comm, pero en valor absoluto son **≈ 0,5–2,5 mJ por ronda** frente a E_comp ≈ 3,6 J (< 0,1 % de la energía) |
| F1 / recall de red | idénticos | idénticos | idénticos | idénticos | **La seguridad no cambia lo que se aprende:** mismas métricas con la misma semilla |
| Mensajes rechazados | 0 | 0 | 0 | 0 | Sin falsos rechazos del canal seguro |

\* **P2P:** los 6,5 s de handshake no son coste criptográfico. Los contenedores arrancan escalonados (≈ 0,65 s entre uno y otro) y el handshake de un par espera a que el otro exista.
- CN1, que arranca primero, acumula 2,6 s (lo que tarda CN5 en arrancar).
- CN5, que arranca el último, tarda 10 ms.

En S0, esa misma espera cae en la fase `HELLO` de P2P y no se ve como «handshake». El coste criptográfico real es del orden de 10 ms por nodo (4 pares).

## Conclusiones (Docker, PC de desarrollo)

1. **S1 y S2 no tienen un coste medible por ronda** con modelos de 450 KB y 1 época: < 2 % y dentro del ruido. El criterio de D3 (≤ 25 %) se cumple con un margen de más de un orden de magnitud.
2. **TLS es la parte visible:** +2 ms por publicación (+65–80 % de `t_publish_confirm`). El sobre E2E de S2 no añade latencia de publicación: el cifrado ocurre antes de publicar y cuesta ≈ 0,3–0,8 ms por modelo.
3. **Costes fijos del S2:** ≈ 5 ms por handshake (incluye la ida y vuelta por el broker), una vez por par y época, más ≈ 7 MB de RAM.
4. **Bytes:** el sobrecoste es despreciable (≤ 0,2 % en la interfaz).
5. **Calidad:** idéntica en los tres perfiles. La comparación de seguridad no está confundida con la calidad del modelo.

## Límites y qué cambia en las Raspberry Pi (E11)

- **Versión rápida:** con 3 épocas y 12–30 rondas, el entrenamiento pesa aún más, así que los Δ% por ronda serán todavía menores. Los costes absolutos por mensaje no cambian.
- **CPU:**
  - 1 núcleo del Ryzen 5 7535HS es ≈ 3–4× más rápido que un Cortex-A72.
  - **El BCM2711 de la Pi 4 no tiene las extensiones criptográficas ARMv8**, así que AES-GCM y SHA-256 corren por software.
  - Estimación [INF], sin medir: el cifrado y el handshake podrían costar **≈ 10–30× más en la Pi**: del orden de 10–40 ms por modelo cifrado. Probablemente siga siendo < 5 % de la ronda, pero hay que medirlo (benchmarks de E3 y E10 en la Pi).
- **Red:** en Docker, la red es local y casi sin latencia. En el clúster, TLS añade ida y vuelta al conectar, y los bytes de la interfaz incluyen el salto real por el broker.
- **Energía:** E_comm es analítica (p·T_comm), así que hereda cualquier cambio de T_comm. Su peso absoluto es muy pequeño frente a E_comp.
- **Estadística:** n = 3 da IC anchos. La campaña publicable (E13) usa n ≥ 5 y la configuración completa de D0.
