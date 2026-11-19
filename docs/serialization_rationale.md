# Formato de comunicación: JSON + `pickle`, y por qué

> Responde a la decisión del usuario Q-03: *"usa el formato de comunicación del proyecto del paper base; si es distinto de JSON o XML, justifica por qué se utiliza"* (analisis_arquitectonico/03 §1; ADR-04 en 04).
> Implementación: `messaging/serialization.py`, `messaging/frame.py`, `messaging/contracts.py`, `schemas/`.

## 1. Cuál es el formato del proyecto base

| Parte del mensaje | Proyecto base | Evidencia |
|---|---|---|
| Control y metadatos | **JSON** | P1/P3 envían `register`, `model_update`, `start_round`… como JSON (D1 §6.1) |
| Parámetros del modelo | **`pickle`** de `dict[nombre → Tensor]` | D0 §3.4 ("serialized with pickle/joblib"); `pickle.dumps(params)` en P1 (`server_centralized.py:387`, `worker_centralized.py:328`) y P3 (`agent_csv.py:391`), con el protocolo por defecto (4) |

**Se conservan ambos.** Para el control, el formato ya es JSON y no necesita justificación. Además se valida con JSON Schema, generalizando los esquemas de D4 (`schemas/`). Lo que necesita justificación es `pickle` para los parámetros.

## 2. Por qué `pickle` y no JSON/XML para los parámetros

| Criterio | `pickle` (formato base) | JSON / XML |
|---|---|---|
| **Fidelidad al paper** | Es el formato con el que se midieron los resultados de D0. El tamaño serializado `M` entra en la **Eq. 5** (`T_comm = M / B·log₂(…)`) y en "Comm. per Round" (Tabla 7) | Cambiaría `M` y, con ello, directamente la energía de comunicación y los bytes por ronda: una variable de confusión frente a D0 |
| **Tamaño medido** (modelo de referencia, 112 641 parámetros) | **453 090 B** (+0,56 % sobre los 450 564 B en float32 crudo) | Texto decimal: ≈ 9–12 caracteres por `float32` ⇒ ≈ 1,0–1,4 MB [estimación]. Base64: ≈ 600 KB + la envoltura. XML: más aún por las etiquetas |
| **CPU en la Raspberry Pi** | Copia de memoria | Formatear y analizar 112 641 flotantes por mensaje, por nodo y por ronda |
| **Exactitud** | Bit a bit (verificado en las pruebas) | La ida y vuelta decimal exige cuidado para no perder precisión |
| **Dependencias** | Ninguna nueva | Ninguna, pero requeriría un formato de tensores propio |
| **Riesgo** | **Ejecución de código** al deserializar datos no confiables | Ninguno de ese tipo |

**Conclusión:** `pickle` para los parámetros es la única opción que mantiene comparables los KPIs de comunicación y energía con el paper. Su riesgo de seguridad se mitiga (§3) sin cambiar los bytes en la red.

## 3. Mitigación del riesgo de `pickle`

### 3.1 El riesgo es real (demostrado)
Un payload `pickle` puede invocar cualquier función al deserializarse. Durante E2 se comprobó, en un directorio temporal, que `pickle.loads` sobre un payload malicioso (lo que hacen P1/P3 con los datos de la red) **ejecuta código**: crea un archivo testigo. Era el hallazgo D1 P1-H9; ahora está demostrado.

### 3.2 Riesgo adicional encontrado en E2
`pickle.dumps` de tensores usa solo tres funciones: `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2` y **`torch.storage._load_from_bytes`**. Esta última hace internamente `torch.load(..., weights_only=False)`, es decir, una **deserialización anidada sin restricciones**. Una lista blanca ingenua que la permitiera dejaría pasar un ataque anidado; esto se verificó con una prueba.

### 3.3 Medidas (en el orden de la ruta de recepción)
1. **Límites de tamaño** de la trama, la cabecera y el payload **antes** de procesar nada (`communication.limits`).
2. **Contrato JSON** de la cabecera (`additionalProperties: false`) con el descriptor del payload: formato, protocolo, `nbytes`, `sha256` y la lista de tensores (nombre, forma, dtype).
3. **Integridad:** tamaño y `sha256` del payload = descriptor; tensores anunciados = modelo esperado por el nodo. Todo **antes** de deserializar.
4. **Autenticación del emisor** (perfiles S1/S2, etapa E10): solo una identidad registrada puede entregar un payload.
5. ***Unpickler* restringido:** solo los 3 `globals` anteriores. `_load_from_bytes` se **sustituye** por una versión que usa `torch.load(weights_only=True)`, el cargador restringido de PyTorch. Cualquier otro `global` ⇒ rechazo sin ejecutar nada.
6. **Validación posterior:** dict de str → tensor `float32` denso, finito, sin gradiente, idéntico al descriptor.

Las medidas 1–3, 5 y 6 **no cambian los bytes en la red**: lo que se envía es exactamente `pickle.dumps(dict)`, como en P1/P3 (verificado: bytes idénticos).

### 3.4 Riesgo residual
- En el perfil **S0** (reproducción sin seguridad) no hay autenticación (medida 4). Se apoya en las medidas 1–3, 5 y 6. Solo debe usarse en una red de laboratorio aislada.
- Un nodo registrado y comprometido podría intentar explotar un defecto en `_rebuild_tensor_v2` o en el cargador restringido de PyTorch. Se mitiga fijando y actualizando la versión de PyTorch, y se documenta como riesgo residual en la evaluación de seguridad.

## 4. Pruebas que lo respaldan

| Prueba | Archivo |
|---|---|
| Bytes en la red idénticos a `pickle.dumps(dict, protocol=4)` de P1/P3; carga exacta de bytes producidos así | `tests/unit/test_messaging_format.py` |
| Tamaño del modelo de referencia = 453 090 B | idem |
| Payloads con `exec`, `eval`, `os.system`, `subprocess.call` ⇒ rechazados, sin archivo testigo | `tests/security/test_malicious_payloads.py` |
| Ataque anidado vía `_load_from_bytes` ⇒ rechazado, sin archivo testigo | idem |
| Ataque completo por la ruta de recepción ⇒ rechazado | idem |
| 600 payloads aleatorios o mutados ⇒ solo errores controlados | idem |
| `NaN`/`Inf`, enteros, `requires_grad`, estructuras ajenas ⇒ rechazados | `tests/unit/test_messaging_format.py` |
| `pickle` solo se importa en `messaging/serialization.py` | `tests/architecture/test_layering.py` |
