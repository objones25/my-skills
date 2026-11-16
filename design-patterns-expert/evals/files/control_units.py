"""Unit conversion used by one CLI command. The three units are fixed by the hardware."""


def to_celsius(value: float, unit: str) -> float:
    if unit == "C":
        return value
    elif unit == "F":
        return (value - 32) * 5 / 9
    elif unit == "K":
        return value - 273.15
    raise ValueError(f"unknown unit {unit!r}")
