"""Compatibility imports for the MemFabric host-memory registry."""

from .operators.mem_cache.memfabric.host_memory import (
    HostMemoryMapping,
    get_memfabric_pool,
    lookup_host_dva,
    register_acl_host_memory,
    register_host_memory,
    resolve_host_dva,
    set_memfabric_pool,
    uninitialize_host_memory,
)

__all__ = [
    "HostMemoryMapping",
    "get_memfabric_pool",
    "lookup_host_dva",
    "register_acl_host_memory",
    "register_host_memory",
    "resolve_host_dva",
    "set_memfabric_pool",
    "uninitialize_host_memory",
]
