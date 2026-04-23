_RING_ATTN_GROUP = None


def set_ring_attn_group(group) -> None:
    global _RING_ATTN_GROUP
    _RING_ATTN_GROUP = group


def get_ring_attn_group():
    return _RING_ATTN_GROUP
