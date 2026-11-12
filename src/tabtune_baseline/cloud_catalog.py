"""Verified São Paulo Processing catalog and AWS bulk-price lookup."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any
from urllib.request import urlopen

from .models import InstanceCandidate, SageMakerConfig

SNAPSHOT = "data/sa_east_1_processing_prices.json"


def _snapshot() -> dict[str, Any]:
    return json.loads(files("tabtune_baseline").joinpath(SNAPSHOT).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _live_prices(url: str) -> tuple[dict[str, float], str]:
    with urlopen(url, timeout=8) as response:
        data = json.load(response)
    rates: dict[str, float] = {}
    for sku, product in data["products"].items():
        attributes = product["attributes"]
        if attributes.get("component") != "Processing" or attributes.get("regionCode") != "sa-east-1":
            continue
        name = attributes.get("instanceName")
        for term in data["terms"]["OnDemand"].get(sku, {}).values():
            for dimension in term["priceDimensions"].values():
                if dimension.get("unit") == "Hrs" and name:
                    rate = float(dimension["pricePerUnit"]["USD"]) / 60
                    if name in rates and rates[name] != rate:
                        raise ValueError(f"Conflicting live Processing rates for {name}")
                    rates[name] = rate
    return rates, data["publicationDate"]


def processing_catalog(cloud: SageMakerConfig, *, use_live: bool = True) -> tuple[list[dict[str, Any]], str, str]:
    """Only the checked Processing/API and regional price intersection is dispatchable."""
    snapshot = _snapshot()
    rows = [dict(row) for row in snapshot["products"]]
    source, at = "bundled_aws_bulk_snapshot", snapshot["publication_date"]
    if use_live:
        try:
            rates, live_at = _live_prices(snapshot["source_url"])
        except (OSError, ValueError, KeyError, TimeoutError):
            pass
        else:
            for row in rows:
                if row["type"] in rates:
                    row["usd_per_hour"] = rates[row["type"]] * 60
                else:
                    # Do not substitute a stale rate for a type absent from the current price list.
                    row["usd_per_hour"] = None
            source, at = "live_aws_bulk_price_list", live_at
    return rows, source, at


def suitable_candidates(estimate: dict[str, Any], cloud: SageMakerConfig,
                        catalog: list[dict[str, Any]], source: str,
                        priced_at: str) -> list[InstanceCandidate]:
    candidates: list[InstanceCandidate] = []
    minutes = max(15.0, 15 + estimate["reference_rows"] / 20_000 +
                  estimate["validation_rows"] / 50_000)
    usable_disk = cloud.volume_size_gb - 10
    for row in catalog:
        rate = row.get("usd_per_hour")
        if not rate or rate <= 0:
            continue
        if (row["ram_gib"] - 2 < estimate["cpu_ram_gib"] or
                row["vram_per_gpu_gib"] < estimate["vram_per_gpu_gib"] or
                usable_disk < estimate["scratch_gib"]):
            continue
        per_minute = float(rate) / 60
        candidates.append(InstanceCandidate(
            instance_type=row["type"], vcpus=row["vcpus"], ram_gib=row["ram_gib"],
            gpu_count=row["gpu_count"], vram_per_gpu_gib=row["vram_per_gpu_gib"],
            scratch_gib=usable_disk, usd_per_minute=per_minute,
            price_source=source, price_at=priced_at, estimated_minutes=round(minutes, 2),
            estimated_cost_usd=round(minutes * per_minute, 4)))
    return sorted(candidates, key=lambda item: (item.usd_per_minute, item.instance_type))
