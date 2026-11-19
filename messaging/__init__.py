"""Pila de mensajería (analisis_arquitectonico/04 §4.2): contratos, serialización, seguridad y transporte.

Regla de dependencias: este paquete NO importa `core`, `topologies` ni `node`; trata los modelos como
`dict[str, torch.Tensor]` con un descriptor.
"""
