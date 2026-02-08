import bchlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

BCH_POLYNOMIAL = 137
BCH_BITS = 5

TIMER_INTERVAL_MINUTES = 15
TIMERS_PER_DAY = 24 * 60 // TIMER_INTERVAL_MINUTES
DAYS_PER_YEAR = 366
TOTAL_TIMERS = DAYS_PER_YEAR * TIMERS_PER_DAY
TIMER_BITS = 16
DATA_BITS = 56
DATA_BYTES = DATA_BITS // 8
SECRET_BITS = 100
FOOTER_BITS = 4


@dataclass(frozen=True)
class ForensicMark:
    timer_index: int
    location_id: int
    adapter_bits: str
    location_bits: int


def normalize_datetime(value):
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def timer_index_for_datetime(value):
    value = normalize_datetime(value)
    day_index = (value.timetuple().tm_yday - 1) % DAYS_PER_YEAR
    quarter_hour = value.minute // TIMER_INTERVAL_MINUTES
    timer_index = day_index * TIMERS_PER_DAY + value.hour * (60 // TIMER_INTERVAL_MINUTES) + quarter_hour
    return timer_index % TOTAL_TIMERS


def datetime_for_timer_index(value, year=None):
    if year is None:
        year = datetime.now(timezone.utc).year
    day_index, remainder = divmod(value % TOTAL_TIMERS, TIMERS_PER_DAY)
    hour, quarter = divmod(remainder, 60 // TIMER_INTERVAL_MINUTES)
    minute = quarter * TIMER_INTERVAL_MINUTES
    base = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_index)
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


def adapter_bit_length(location_bits):
    return DATA_BITS - TIMER_BITS - location_bits


def _coerce_adapter_bits(adapter_bits, length):
    if adapter_bits is None:
        return "0" * length
    if isinstance(adapter_bits, int):
        if adapter_bits < 0 or adapter_bits >= (1 << length):
            raise ValueError(f"Adapter value must fit in {length} bits.")
        return format(adapter_bits, f"0{length}b")
    candidate = str(adapter_bits).strip()
    if candidate.startswith("0b"):
        candidate = candidate[2:]
    if set(candidate) <= {"0", "1"}:
        if len(candidate) != length:
            raise ValueError(f"Adapter bits must be {length} bits long.")
        return candidate
    value = int(candidate)
    if value < 0 or value >= (1 << length):
        raise ValueError(f"Adapter value must fit in {length} bits.")
    return format(value, f"0{length}b")


def build_payload_bits(timer_index, location_id, location_bits=20, adapter_bits=None):
    if location_bits not in (19, 20):
        raise ValueError("Location bits must be 19 or 20.")
    if timer_index < 0 or timer_index >= TOTAL_TIMERS:
        raise ValueError(f"Timer index must be in [0, {TOTAL_TIMERS}).")
    if location_id < 0 or location_id >= (1 << location_bits):
        raise ValueError(f"Location id must fit in {location_bits} bits.")
    adapter_len = adapter_bit_length(location_bits)
    adapter_bits = _coerce_adapter_bits(adapter_bits, adapter_len)
    payload_value = ((timer_index << (location_bits + adapter_len)) |
                     (location_id << adapter_len) |
                     int(adapter_bits, 2))
    return format(payload_value, f"0{DATA_BITS}b")


def build_payload_bytes(timer_index, location_id, location_bits=20, adapter_bits=None):
    payload_bits = build_payload_bits(timer_index, location_id, location_bits, adapter_bits)
    payload_value = int(payload_bits, 2)
    return payload_value.to_bytes(DATA_BYTES, "big")


def build_secret_bits(timer_index, location_id, location_bits=20, adapter_bits=None):
    payload = bytearray(build_payload_bytes(timer_index, location_id, location_bits, adapter_bits))
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    ecc = bch.encode(payload)
    packet = payload + ecc
    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0] * FOOTER_BITS)
    if len(secret) != SECRET_BITS:
        raise ValueError("Secret bit length mismatch.")
    return secret


def parse_payload_bits(payload_bits, location_bits=20):
    if location_bits not in (19, 20):
        raise ValueError("Location bits must be 19 or 20.")
    payload_bits = payload_bits.zfill(DATA_BITS)
    timer_bits = payload_bits[:TIMER_BITS]
    location_start = TIMER_BITS
    location_end = TIMER_BITS + location_bits
    location_bits_value = payload_bits[location_start:location_end]
    adapter_bits = payload_bits[location_end:]
    return ForensicMark(
        timer_index=int(timer_bits, 2),
        location_id=int(location_bits_value, 2),
        adapter_bits=adapter_bits,
        location_bits=location_bits,
    )


def parse_payload_bytes(payload, location_bits=20):
    payload_bits = format(int.from_bytes(payload, "big"), f"0{DATA_BITS}b")
    return parse_payload_bits(payload_bits, location_bits=location_bits)
