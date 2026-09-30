from typing import Dict, Any, List
from langchain_core.tools import tool

# Base de datos simulada en memoria
_DB_PEDIDOS: Dict[int, Dict[str, Any]] = {
    101: {
        "cliente_id": 101,
        "nombre": "Ana Pérez",
        "pedidos": [
            {"pedido_id": "P-101-A", "monto": 4500, "estado": "entregado", "fecha": "2026-02-10"},
            {"pedido_id": "P-101-B", "monto": 8200, "estado": "en camino", "fecha": "2026-03-01"}
        ]
    },
    102: {
        "cliente_id": 102,
        "nombre": "Carlos Gómez",
        "pedidos": [
            {"pedido_id": "P-102-1", "monto": 5000, "estado": "entregado", "fecha": "2026-01-15"},
            {"pedido_id": "P-102-2", "monto": 3500, "estado": "entregado", "fecha": "2026-02-20"},
            {"pedido_id": "P-102-3", "monto": 6000, "estado": "entregado", "fecha": "2026-03-05"}
        ]
    }
}

_DB_DETALLE_PEDIDO: Dict[str, Dict[str, Any]] = {
    "P-102-1": {"items": ["Mouse Logitech", "Pad XL"], "pago": "Tarjeta de Crédito"},
    "P-102-2": {"items": ["Teclado Mecánico"], "pago": "Transferencia"},
    "P-102-3": {"items": ["Monitor 24 pulgadas", "Cable HDMI"], "pago": "Tarjeta de Débito"},
}


@tool
def buscar_pedidos_cliente(cliente_id: int) -> Dict[str, Any]:
    """Busca en la base de datos el historial de pedidos asociados a un cliente específico.

    Args:
        cliente_id: Identificador numérico único del cliente (por ejemplo, 101 o 102).

    Returns:
        Un diccionario con la cantidad de pedidos, el monto total acumulado y la lista
        básica de pedidos con su fecha y estado.
    """
    cliente = _DB_PEDIDOS.get(cliente_id)
    if not cliente:
        return {"error": f"No se encontró el cliente con ID {cliente_id}"}

    pedidos = cliente["pedidos"]
    total = sum(p["monto"] for p in pedidos)
    return {
        "cliente": cliente["nombre"],
        "cantidad_pedidos": len(pedidos),
        "monto_total": total,
        "pedidos": pedidos
    }


@tool
def consultar_detalle_pedido(pedido_id: str) -> Dict[str, Any]:
    """Obtiene el detalle granular de productos y método de pago de un pedido específico.

    Args:
        pedido_id: Código identificador del pedido (por ejemplo, 'P-102-3').

    Returns:
        Diccionario con la lista de ítems incluidos y la forma de pago utilizada.
    """
    detalle = _DB_DETALLE_PEDIDO.get(pedido_id)
    if not detalle:
        return {"error": f"No existe detalle para el pedido {pedido_id}"}
    return {
        "pedido_id": pedido_id,
        "detalle": detalle
    }


ALL_TOOLS = [buscar_pedidos_cliente, consultar_detalle_pedido]