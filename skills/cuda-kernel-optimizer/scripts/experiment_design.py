#!/usr/bin/env python3
"""Deterministic, position-balanced schedules for paired experiments."""

from __future__ import annotations

import random


def balanced_pair_orders(blocks: int, *, seed: int = 0) -> list[str]:
    """Return a seeded AB/BA schedule whose direction counts differ by at most one."""
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks <= 0:
        raise ValueError("blocks must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    rng = random.Random(seed)
    half = blocks // 2
    orders = ["AB"] * half + ["BA"] * half
    if blocks % 2:
        orders.append(rng.choice(("AB", "BA")))
    rng.shuffle(orders)
    return orders
