from __future__ import annotations

import io
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import requests
from rdflib import Graph, Namespace, RDF
from rdflib.namespace import RDFS

# Healthcare dependency
from ucumvert import PintUcumRegistry
from pprint import pprint


# ============================================================
# Online sources
# ============================================================

QUDT_UNITS_TTL_URL = "http://qudt.org/3.1.11/vocab/unit"
UNECE_REC20_XLSX_URL = "https://unece.org/sites/default/files/2023-10/rec20_Rev17e-2021.xlsx"
XBRL_UTR_XML_URL = "https://www.xbrl.org/utr/utr.xml"

QUDT = Namespace("http://qudt.org/schema/qudt/")


# ============================================================
# Common helpers
# ============================================================

@dataclass
class ConversionEntry:
    unit: str
    label: str
    canonical_unit: str
    quantity_kind: str
    factor_to_canonical: Optional[float]
    offset_to_canonical: Optional[float] = None
    source: str = ""
    aliases: Optional[list[str]] = None


def _http_get(url: str, timeout: int = 60) -> requests.Response:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        s = str(v).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _local_name(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ============================================================
# QUDT loaders (environmental / products)
# ============================================================

def _load_qudt_graph() -> Graph:
    g = Graph()
    g.parse(data=_http_get(QUDT_UNITS_TTL_URL).text, format="turtle")
    return g


def _qudt_label_map(g: Graph) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for s, _, o in g.triples((None, RDFS.label, None)):
        out[str(s)] = str(o)
    return out


def _qudt_units_raw() -> Dict[str, dict]:
    g = _load_qudt_graph()
    labels = _qudt_label_map(g)
    units: Dict[str, dict] = {}

    for subj in set(g.subjects(RDF.type, QUDT.Unit)):
        subj_s = str(subj)

        label = labels.get(subj_s, _local_name(subj_s))
        symbol = None
        ucum_code = None
        conv_mult = None
        conv_off = None
        qkinds = []

        for _, _, o in g.triples((subj, QUDT.symbol, None)):
            symbol = str(o)
            break

        for _, _, o in g.triples((subj, QUDT.ucumCode, None)):
            ucum_code = str(o)
            break

        for _, _, o in g.triples((subj, QUDT.conversionMultiplier, None)):
            conv_mult = _safe_float(o)
            break

        for _, _, o in g.triples((subj, QUDT.conversionOffset, None)):
            conv_off = _safe_float(o)
            break

        for _, _, o in g.triples((subj, QUDT.hasQuantityKind, None)):
            qkinds.append(_local_name(str(o)))

        if not qkinds:
            continue

        units[subj_s] = {
            "uri": subj_s,
            "id": _local_name(subj_s),
            "label": label,
            "symbol": symbol,
            "ucum_code": ucum_code,
            "conversion_multiplier": conv_mult,
            "conversion_offset": conv_off,
            "quantity_kinds": sorted(set(qkinds)),
        }

    return units


def _build_qudt_conversion_dict(
    allowed_quantity_kinds: Iterable[str],
    source_name: str,
    prefer_ucum_code: bool = True,
) -> Dict[str, dict]:
    allowed = set(allowed_quantity_kinds)
    raw = _qudt_units_raw()

    grouped: Dict[str, list[dict]] = defaultdict(list)
    for u in raw.values():
        for qk in u["quantity_kinds"]:
            if qk in allowed:
                grouped[qk].append(u)

    out: Dict[str, dict] = {}

    for qk, items in grouped.items():
        def canonical_score(x: dict) -> tuple:
            return (
                0 if (x["conversion_multiplier"] == 1.0 and (x["conversion_offset"] in (None, 0.0))) else 1,
                0 if x.get("ucum_code") else 1,
                len(x["id"]),
            )

        items_sorted = sorted(items, key=canonical_score)
        canonical = items_sorted[0]
        canonical_token = canonical["ucum_code"] if (prefer_ucum_code and canonical.get("ucum_code")) else canonical["id"]

        units_dict: Dict[str, dict] = {}

        for u in items_sorted:
            token_candidates = [
                u.get("ucum_code"),
                u.get("symbol"),
                u.get("id"),
                u.get("label"),
            ]
            token_candidates = [t for t in token_candidates if t]

            aliases = []
            if u.get("symbol") and u["symbol"] != u.get("ucum_code"):
                aliases.append(u["symbol"])
            if u.get("id") not in aliases:
                aliases.append(u["id"])
            if u.get("label") not in aliases:
                aliases.append(u["label"])

            primary = token_candidates[0]
            units_dict[primary] = asdict(
                ConversionEntry(
                    unit=primary,
                    label=u["label"],
                    canonical_unit=canonical_token,
                    quantity_kind=qk,
                    factor_to_canonical=u["conversion_multiplier"],
                    offset_to_canonical=u["conversion_offset"],
                    source=source_name,
                    aliases=[a for a in aliases if a != primary],
                )
            )

        out[qk] = {
            "canonical_unit": canonical_token,
            "units": units_dict,
        }

    return out


# ============================================================
# Environmental
# ============================================================

def build_environmental_conversion_dict() -> Dict[str, dict]:
    environmental_qks = {
        "Mass",
        "Length",
        "Area",
        "Volume",
        "Time",
        "Pressure",
        "Temperature",
        "ThermodynamicTemperature",
        "Energy",
        "Power",
        "Velocity",
        "Acceleration",
        "Force",
        "Density",
        "MassDensity",
        "VolumetricFlowRate",
        "MassFlowRate",
        "AmountOfSubstance",
        "MolarMass",
        "AmountOfSubstanceConcentration",
        "MassConcentration",
        "Frequency",
    }
    return _build_qudt_conversion_dict(
        allowed_quantity_kinds=environmental_qks,
        source_name="QUDT",
        prefer_ucum_code=True,
    )


# ============================================================
# Healthcare (rewritten with ucumvert)
# ============================================================

def build_healthcare_conversion_dict() -> Dict[str, dict]:
    """
    Build a healthcare conversion dictionary using ucumvert.

    Output shape stays compatible:
    {
      "<quantity_kind>": {
        "canonical_unit": "<token>",
        "units": {
          "<ucum_code>": {
             "unit": ...,
             "label": ...,
             "canonical_unit": ...,
             "quantity_kind": ...,
             "factor_to_canonical": ...,
             "offset_to_canonical": ...,
             "source": ...,
             "aliases": [...]
          }
        }
      }
    }

    Fixes applied:
    - correct affine handling for temperature
    - proper grouping for stable compound healthcare units
    - numeric factors for compound units with stable metrological conversions
    """
    ureg = PintUcumRegistry()

    canonical_by_kind = {
        "Mass": "g",
        "Volume": "L",
        "Length": "m",
        "Time": "s",
        "AmountOfSubstance": "mol",
        "Temperature": "K",
        "Pressure": "Pa",
        "Ratio": "Ratio",
        "AmountConcentration": "mol/L",
        "MassConcentration": "g/L",
        "CellCountConcentration": "/uL",
        "ArbitraryClinicalUnit": None,
        "OtherHealthcareUnit": None,
    }

    common_ucum_units = [
        # mass
        "g", "kg", "mg", "ug", "ng", "pg",
        # volume
        "L", "mL", "uL", "dL",
        # length
        "m", "cm", "mm", "um", "nm",
        # time
        "s", "min", "h", "d", "wk", "mo", "a",
        # amount of substance
        "mol", "mmol", "umol", "nmol",
        # amount concentration
        "mol/L", "mmol/L", "umol/L",
        # mass concentration
        "g/L", "mg/dL", "g/dL",
        # ratio-like
        "%", "[ppth]", "[ppm]", "[ppb]",
        # pressure
        "Pa", "kPa", "mm[Hg]",
        # temperature
        "K", "Cel", "[degF]",
        # counts / cell concentrations
        "/uL", "10*3/uL", "10*6/uL",
        # arbitrary clinical units
        "[iU]", "U",
    ]

    grouped: Dict[str, dict] = defaultdict(lambda: {"canonical_unit": None, "units": {}})

    for ucum_code in common_ucum_units:
        try:
            q = ureg.from_ucum(ucum_code)
        except Exception:
            continue

        quantity_kind = _infer_healthcare_quantity_kind_from_ucum(ucum_code)
        canonical_unit = canonical_by_kind.get(quantity_kind)

        factor_to_canonical = None
        offset_to_canonical = None

        # Temperature must be handled as affine transforms, not by converting "1 unit".
        if quantity_kind == "Temperature":
            factor_to_canonical, offset_to_canonical = _healthcare_temperature_params(ucum_code)

        # Arbitrary units remain non-convertible.
        elif quantity_kind == "ArbitraryClinicalUnit":
            factor_to_canonical = None
            offset_to_canonical = None

        # All other stable units can use multiplicative conversion.
        elif canonical_unit is not None:
            try:
                factor_to_canonical = _ucum_factor_to_canonical(
                    ureg=ureg,
                    ucum_code=ucum_code,
                    canonical_unit=canonical_unit,
                )
                offset_to_canonical = None
            except Exception:
                factor_to_canonical = None
                offset_to_canonical = None

        label = ucum_code
        aliases = _healthcare_aliases(ucum_code)

        grouped[quantity_kind]["canonical_unit"] = canonical_unit or "self"
        grouped[quantity_kind]["units"][ucum_code] = asdict(
            ConversionEntry(
                unit=ucum_code,
                label=label,
                canonical_unit=canonical_unit or "self",
                quantity_kind=quantity_kind,
                factor_to_canonical=factor_to_canonical,
                offset_to_canonical=offset_to_canonical,
                source="ucumvert",
                aliases=aliases,
            )
        )

    return dict(grouped)


def _infer_healthcare_quantity_kind_from_ucum(ucum_code: str) -> str:
    code = ucum_code.strip()

    if code in {"g", "kg", "mg", "ug", "ng", "pg"}:
        return "Mass"
    if code in {"L", "mL", "uL", "dL"}:
        return "Volume"
    if code in {"m", "cm", "mm", "um", "nm"}:
        return "Length"
    if code in {"s", "min", "h", "d", "wk", "mo", "a"}:
        return "Time"
    if code in {"mol", "mmol", "umol", "nmol"}:
        return "AmountOfSubstance"
    if code in {"mol/L", "mmol/L", "umol/L"}:
        return "AmountConcentration"
    if code in {"g/L", "mg/dL", "g/dL"}:
        return "MassConcentration"
    if code in {"/uL", "10*3/uL", "10*6/uL"}:
        return "CellCountConcentration"
    if code in {"K", "Cel", "[degF]"}:
        return "Temperature"
    if code in {"Pa", "kPa", "mm[Hg]"}:
        return "Pressure"
    if code in {"%", "[ppth]", "[ppm]", "[ppb]"}:
        return "Ratio"
    if code in {"[iU]", "U"}:
        return "ArbitraryClinicalUnit"
    return "OtherHealthcareUnit"


def _ucum_factor_to_canonical(ureg: PintUcumRegistry, ucum_code: str, canonical_unit: str) -> float:
    """
    Compute multiplicative factor from ucum_code to canonical_unit.

    This is valid only for non-affine units.
    """
    q = ureg.from_ucum(ucum_code)
    converted = q.to(canonical_unit)
    return float(converted.magnitude)


def _healthcare_temperature_params(ucum_code: str) -> tuple[Optional[float], Optional[float]]:
    """
    Return affine conversion parameters to Kelvin:
        K = value * factor + offset
    """
    if ucum_code == "K":
        return 1.0, 0.0
    if ucum_code == "Cel":
        return 1.0, 273.15
    if ucum_code == "[degF]":
        return 5.0 / 9.0, 255.3722222222222
    return None, None


def _healthcare_aliases(ucum_code: str) -> list[str]:
    alias_map = {
        "g": ["gram"],
        "kg": ["kilogram"],
        "mg": ["milligram"],
        "ug": ["microgram"],
        "ng": ["nanogram"],
        "pg": ["picogram"],

        "L": ["liter", "litre"],
        "mL": ["milliliter", "millilitre"],
        "uL": ["microliter", "microlitre"],
        "dL": ["deciliter", "decilitre"],

        "m": ["meter", "metre"],
        "cm": ["centimeter", "centimetre"],
        "mm": ["millimeter", "millimetre"],
        "um": ["micrometer", "micrometre"],
        "nm": ["nanometer", "nanometre"],

        "s": ["second"],
        "min": ["minute"],
        "h": ["hour"],
        "d": ["day"],
        "wk": ["week"],
        "mo": ["month"],
        "a": ["year"],

        "mol": ["mole"],
        "mmol": ["millimole"],
        "umol": ["micromole"],
        "nmol": ["nanomole"],

        "mol/L": ["molar", "mole per liter", "mole per litre"],
        "mmol/L": ["millimolar", "millimole per liter", "millimole per litre"],
        "umol/L": ["micromolar", "micromole per liter", "micromole per litre"],

        "g/L": ["gram per liter", "gram per litre"],
        "mg/dL": ["milligram per deciliter", "milligram per decilitre"],
        "g/dL": ["gram per deciliter", "gram per decilitre"],

        "/uL": ["per microliter", "per microlitre"],
        "10*3/uL": ["thousand per microliter", "10^3 per microliter"],
        "10*6/uL": ["million per microliter", "10^6 per microliter"],

        "Pa": ["pascal"],
        "kPa": ["kilopascal"],
        "mm[Hg]": ["mmHg", "millimeter mercury"],

        "K": ["kelvin"],
        "Cel": ["celsius", "degC"],
        "[degF]": ["fahrenheit", "degF"],

        "%": ["percent", "pct"],
        "[ppth]": ["per thousand"],
        "[ppm]": ["ppm", "parts per million"],
        "[ppb]": ["ppb", "parts per billion"],

        "[iU]": ["IU", "international unit"],
        "U": ["unit"],
    }
    return alias_map.get(ucum_code, [])

# ============================================================
# Products
# ============================================================

def build_products_conversion_dict() -> Dict[str, dict]:
    """
    Product tables usually need:
    1) official trade/unit codes (UNECE Rec 20)
    2) stable conversion math for physical dimensions (QUDT)

    This version uses the UNECE HTML vocabulary page because:
    - the XLSX URL currently returns 403
    - the advertised JSON-LD URL currently returns 404
    """
    qudt = _build_qudt_conversion_dict(
        allowed_quantity_kinds={"Mass", "Length", "Area", "Volume", "Temperature", "Time"},
        source_name="QUDT+UNECE",
        prefer_ucum_code=False,
    )

    unece_pairs = _load_unece_rec20_html_pairs()

    label_to_entries = defaultdict(list)
    for qk_meta in qudt.values():
        for _, meta in qk_meta["units"].items():
            label_to_entries[_normalize_text(meta["label"])].append(meta)
            for a in meta.get("aliases") or []:
                label_to_entries[_normalize_text(str(a))].append(meta)

    for code, name in unece_pairs:
        key = _normalize_text(name)
        matches = label_to_entries.get(key, [])
        for meta in matches:
            aliases = set(meta.get("aliases") or [])
            aliases.add(code)
            aliases.add(name)
            meta["aliases"] = sorted(aliases)

    return qudt


def _load_unece_rec20_html_pairs() -> list[tuple[str, str]]:
    """
    Scrape the UNECE Rec 20 HTML vocabulary page.

    Extracts pairs like:
      ("E10", "degree day")
      ("H80", "rack unit")
    """
    url = "https://service.unece.org/trade/uncefact/vocabulary/rec20/"

    r = requests.get(
        url,
        timeout=60,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    r.raise_for_status()
    html = r.text

    # Each entry in the page looks like:
    # ## degree_day
    # * @id: rec20:degree_day
    # * @type: uncefact:UNECERec20Code
    # * Comment: ...
    # * rdfs:comment: ...
    # * rdf:value: E10
    #
    # We extract:
    # - heading after "## "
    # - code after "rdf:value:"
    pattern = re.compile(
        r"##\s+(.+?)\n(?:.*?\n)*?\s*\*\s*rdf:value:\s*([A-Z0-9]+)",
        re.MULTILINE,
    )

    pairs = []
    for raw_name, code in pattern.findall(html):
        name = raw_name.strip()
        name = name.replace("_", " ")
        pairs.append((code.strip(), name.strip()))

    # de-duplicate while preserving order
    seen = set()
    out = []
    for code, name in pairs:
        key = (code, name)
        if key not in seen:
            seen.add(key)
            out.append(key)

    return out


# ============================================================
# Finance
# ============================================================

def build_finance_conversion_dict() -> Dict[str, dict]:
    """
    Finance has very few true stable 'unit conversions' beyond:
    - decimal scales (thousand, million, billion, trillion)
    - ratio/pure/percent/basis-point normalization
    - count-like units such as shares

    This version fixes CurrencyRepresentation so it includes only genuine
    monetary units from XBRL UTR, instead of accidentally pulling in
    physical/scientific units.
    """
    xml_bytes = _http_get(XBRL_UTR_XML_URL).content
    root = ET.fromstring(xml_bytes)

    units_found = []
    for unit in root.findall(".//{*}unit"):
        unit_id = (unit.findtext("{*}unitId") or "").strip()
        unit_name = (unit.findtext("{*}unitName") or "").strip()
        symbol = (unit.findtext("{*}symbol") or "").strip()
        item_type = (unit.findtext("{*}itemType") or "").strip()
        ns_unit = (unit.findtext("{*}nsUnit") or "").strip()
        units_found.append(
            {
                "unit_id": unit_id,
                "unit_name": unit_name,
                "symbol": symbol,
                "item_type": item_type,
                "ns_unit": ns_unit,
            }
        )

    out: Dict[str, dict] = {
        "DecimalScale": {
            "canonical_unit": "unit",
            "units": {
                "unit": asdict(ConversionEntry(
                    unit="unit", label="unit", canonical_unit="unit",
                    quantity_kind="DecimalScale", factor_to_canonical=1.0,
                    source="XBRL UTR + manual stable scales", aliases=["ones"]
                )),
                "thousand": asdict(ConversionEntry(
                    unit="thousand", label="thousand", canonical_unit="unit",
                    quantity_kind="DecimalScale", factor_to_canonical=1_000.0,
                    source="XBRL UTR + manual stable scales", aliases=["k", "thousands"]
                )),
                "million": asdict(ConversionEntry(
                    unit="million", label="million", canonical_unit="unit",
                    quantity_kind="DecimalScale", factor_to_canonical=1_000_000.0,
                    source="XBRL UTR + manual stable scales", aliases=["mn", "millions", "mio"]
                )),
                "billion": asdict(ConversionEntry(
                    unit="billion", label="billion", canonical_unit="unit",
                    quantity_kind="DecimalScale", factor_to_canonical=1_000_000_000.0,
                    source="XBRL UTR + manual stable scales", aliases=["bn", "billions"]
                )),
                "trillion": asdict(ConversionEntry(
                    unit="trillion", label="trillion", canonical_unit="unit",
                    quantity_kind="DecimalScale", factor_to_canonical=1_000_000_000_000.0,
                    source="XBRL UTR + manual stable scales", aliases=["tn", "trillions"]
                )),
            }
        },
        "Ratio": {
            "canonical_unit": "pure",
            "units": {
                "pure": asdict(ConversionEntry(
                    unit="pure", label="pure", canonical_unit="pure",
                    quantity_kind="Ratio", factor_to_canonical=1.0,
                    source="XBRL UTR", aliases=["ratio", "xbrli:pure"]
                )),
                "percent": asdict(ConversionEntry(
                    unit="percent", label="percent", canonical_unit="pure",
                    quantity_kind="Ratio", factor_to_canonical=0.01,
                    source="XBRL UTR + manual stable scales", aliases=["%", "pct"]
                )),
                "basis_point": asdict(ConversionEntry(
                    unit="basis_point", label="basis point", canonical_unit="pure",
                    quantity_kind="Ratio", factor_to_canonical=0.0001,
                    source="manual stable finance rule", aliases=["bp", "bps", "bips"]
                )),
            }
        },
        "Count": {
            "canonical_unit": "count",
            "units": {
                "count": asdict(ConversionEntry(
                    unit="count", label="count", canonical_unit="count",
                    quantity_kind="Count", factor_to_canonical=1.0,
                    source="XBRL UTR + manual stable count rule", aliases=["units", "items"]
                )),
                "shares": asdict(ConversionEntry(
                    unit="shares", label="shares", canonical_unit="count",
                    quantity_kind="Count", factor_to_canonical=1.0,
                    source="XBRL UTR + manual stable count rule", aliases=["share"]
                )),
            }
        },
        "CurrencyRepresentation": {
            "canonical_unit": "currency",
            "units": {}
        }
    }

    for u in units_found:
        unit_id = u["unit_id"]
        unit_name = u["unit_name"]
        symbol = u["symbol"]
        item_type = u["item_type"]
        ns_unit = u["ns_unit"]

        if not unit_id:
            continue

        if not _is_monetary_xbrl_unit(unit_id=unit_id, item_type=item_type, ns_unit=ns_unit, unit_name=unit_name):
            continue

        if unit_id in out["CurrencyRepresentation"]["units"]:
            continue

        aliases = [a for a in [symbol] if a and a != unit_id]

        out["CurrencyRepresentation"]["units"][unit_id] = {
            "unit": unit_id,
            "label": unit_name or unit_id,
            "canonical_unit": "currency",
            "quantity_kind": "CurrencyRepresentation",
            "factor_to_canonical": None,
            "offset_to_canonical": None,
            "source": "XBRL UTR",
            "aliases": aliases,
            "convertible": False,
        }

    return out


def _is_monetary_xbrl_unit(unit_id: str, item_type: str, ns_unit: str, unit_name: str) -> bool:
    """
    True only for monetary/currency-style XBRL units.

    Keeps:
    - standard monetary item types
    - ISO 4217-like codes and XBRL monetary units
    - a small set of known special monetary units used in filings

    Rejects:
    - engineering/scientific units
    - time, mass, energy, distance units
    - shares, pure, and similar non-currency units
    """
    uid = (unit_id or "").strip()
    it = (item_type or "").strip().lower()
    ns = (ns_unit or "").strip().lower()
    unit_name = (unit_name or "").strip().lower()

    if uid in ["GWM", "MWM", "MVA", "TEU", "Volume_per_Monetary", "Energy_per_Monetary", "Emissions_per_Monetary"]:
        return False

    # Strong positive signal from item type
    if "monetary" in it:
        return True

    # Many XBRL monetary units sit in ISO4217 namespace.
    if "iso4217" in ns:
        return True

    # Exclude obvious non-currency units that sometimes appear in the UTR.
    non_currency_ids = {
        "pure", "shares", "share", "count", "rate",
    }
    if uid in non_currency_ids:
        return False

    # Typical current/historical ISO 4217 and XBRL-style monetary unit IDs
    # are 3 uppercase letters. This intentionally includes historical codes.
    if re.fullmatch(r"[A-Z]{3}", uid):
        return True

    # Special monetary / unit-of-account codes used in structured reporting.
    special_monetary_units = {
        "XDR", "CHE", "CHW", "BOV", "CLF", "COU", "MXV", "UYI", "UYW",
        "USN", "USS", "XSU", "XUA", "XBA", "XBB", "XBC", "XBD", "XFU",
        "XAF", "XOF", "XPF", "XCD",
    }
    if uid in special_monetary_units:
        return True

    return False


# ============================================================
# Build all
# ============================================================

def build_all_domain_conversion_dicts() -> Dict[str, dict]:
    return {
        "environmental": build_environmental_conversion_dict(),
        "finance": build_finance_conversion_dict(),
        "healthcare": build_healthcare_conversion_dict(),
        "products": build_products_conversion_dict(),
    }


# ============================================================
# Generic converter for the produced dictionaries
# ============================================================

def count_decimals(x):
    s = str(x)
    if "." in s:
        return len(s.split(".")[1])
    return 0

def convert_value(
    value: float,
    from_unit: str,
    to_unit: str,
    domain_dict: Dict[str, dict],
) -> (float, int):
    from_meta = None
    to_meta = None

    for _, qmeta in domain_dict.items():
        for u, meta in qmeta["units"].items():
            names = {u, *(meta.get("aliases") or [])}
            if from_unit in names:
                from_meta = meta
            if to_unit in names:
                to_meta = meta

    if not from_meta:
        raise KeyError(f"Unknown from_unit: {from_unit}")
    if not to_meta:
        raise KeyError(f"Unknown to_unit: {to_unit}")

    if from_meta["quantity_kind"] != to_meta["quantity_kind"]:
        raise ValueError(
            f"Incompatible quantity kinds: {from_meta['quantity_kind']} vs {to_meta['quantity_kind']}"
        )

    f1 = from_meta["factor_to_canonical"]
    f2 = to_meta["factor_to_canonical"]
    o1 = from_meta.get("offset_to_canonical") or 0.0
    o2 = to_meta.get("offset_to_canonical") or 0.0

    if f1 is None or f2 is None:
        raise ValueError("One of the units is non-convertible or lacks a stable factor.")

    try:
        value = float(value)
    except:
        raise ValueError()

    canonical_value = value * f1 + o1
    value_converted = (canonical_value - o2) / f2

    max_num_decimals, needed_decimals_number = 12, -1
    num_decimals = count_decimals(value)
    # roundtrip check for number of decimals
    for i in range(max_num_decimals+1):
        roundtrip_value = ((round(value_converted, i) * f2 + o2) - o1) / f1
        if round(roundtrip_value, num_decimals) == round(value, num_decimals):
            needed_decimals_number = i
            break

    if needed_decimals_number == -1:
        pass
    return value_converted, needed_decimals_number


# ============================================================
# Example usage
# ============================================================

@lru_cache
def get_unit_env(domain: str) -> dict:
    if domain == "environmental":
        env = build_environmental_conversion_dict()
    elif domain == "finance":
        env = build_finance_conversion_dict()
    elif domain == "healthcare":
        env = build_healthcare_conversion_dict()
    elif domain == "products":
        env = build_products_conversion_dict()
    else:
        raise ValueError(f"Unknown domain: {domain}")

    return env

def is_unit_in_domain(unit: str, domain: str):
    env = get_unit_env(domain)
    #is_canonical = False
    category_found = None
    for category in env:
        try: # the dictionary may possibly have typos
            if unit == env[category]["canonical_unit"]:
                #is_canonical = True
                category_found = category
                break

            for possible_unit in env[category]["units"]:
                if unit == possible_unit:
                    category_found = category
                    break

                for alias in env[category]["units"][possible_unit].get("aliases") or []:
                    if unit == alias:
                        category_found = category
                        break
        except:
            continue

    return category_found #is_canonical, category_found

def get_random_unit(unit: str, domain: str, seed: int = 0):
    env = get_unit_env(domain)
    category = is_unit_in_domain(unit, domain)
    if category is None:
        return None

    category_to_consider = category
    surrogate_initial_unit = unit
    if domain == "finance":
        if category == "CurrencyRepresentation":
            category_to_consider = "DecimalScale"
            surrogate_initial_unit = "unit"


    random_unit = set()
    for possible_unit in env[category_to_consider]["units"]:
        if env[category_to_consider]["units"][possible_unit]["factor_to_canonical"] is None or env[category_to_consider]["units"][possible_unit]["factor_to_canonical"] == 1 or env[category_to_consider]["units"][possible_unit]["factor_to_canonical"] == 0:
            continue
        if possible_unit == surrogate_initial_unit or surrogate_initial_unit in env[category_to_consider]["units"][possible_unit].get("aliases"):
            continue

        """if domain == "environmental" and "[" in possible_unit:
            possible_unit_cleaned = possible_unit.replace("[", "").replace("]", "")
        else:
            possible_unit_cleaned = possible_unit"""
        possible_unit_cleaned = possible_unit

        if not (domain == "environmental" and ("." in possible_unit_cleaned)):
            random_unit.add(possible_unit_cleaned)

        if domain == "environmental":
            aliases = env[category_to_consider]["units"][possible_unit].get("aliases")[:-1] or [] # we skip the last alias, as it may contain foreign gibberish
        else:
            aliases = env[category_to_consider]["units"][possible_unit].get("aliases") or []

        for alias in aliases:
            random_unit.add(alias)

    rnd = random.Random(seed)
    chosen_unit = rnd.choice(list(random_unit))

    return surrogate_initial_unit, chosen_unit

@lru_cache(maxsize=None)
def get_set_of_units_from_domain(domain: str, env: dict):
    set_of_canonical_units = set()
    for category in env:
        if domain == "finance" and category != "CurrencyRepresentation":
            continue
        for unit in env[category]["units"]:
            set_of_canonical_units.add(unit)

    return set_of_canonical_units

def get_n_canonical_units(domain: str, n: int, seed: int = 0):
    env = get_unit_env(domain)
    set_of_canonical_units = set()
    for category in env:
        if domain == "finance" and category != "CurrencyRepresentation":
            continue
        for unit in env[category]["units"]:
            if domain == "finance" and "aliases" not in env[category]["units"][unit]:
                continue
            if domain == "finance" and len(env[category]["units"][unit]["aliases"]) == 0:
                continue
            set_of_canonical_units.add(unit)

    rnd = random.Random(seed)
    units = rnd.sample(list(set_of_canonical_units), min(n, len(set_of_canonical_units)))
    return units

def required_decimals_for_roundtrip(factor, d_old, max_decimals=12):
    source_step = 10 ** (-d_old)
    for d_new in range(max_decimals + 1):
        target_step = round(source_step * factor, d_new)
        recovered = target_step / factor
        if round(recovered, d_old) == round(source_step, d_old):
            return d_new
    return max_decimals

def get_value(source_unit: str, final_unit: str, domain: str, value: float, nan_values: list[str] = []):
    env = get_unit_env(domain)
    if value is None or value in nan_values:
        return value, None

    value_converted, decimal_nums = convert_value(value, source_unit, final_unit, env)
    return value_converted, decimal_nums

def is_value_going_up(source_unit: str, final_unit: str, domain: str, value: float, nan_values: list[str] = []):
    val = get_value(source_unit, final_unit, domain, value, nan_values)
    if val > value:
        return True
    return False

if __name__ == "__main__":
    all_dicts = build_all_domain_conversion_dicts()

    healthcare = all_dicts["healthcare"]
    finance = all_dicts["finance"]
    environmental = all_dicts["environmental"]
    products = all_dicts["products"]

    pprint(environmental)
    print(a)

    # random unit retrieval examples
    for i in range(5):
        unit = "USD"
        value = 25
        surrogate_initial_unit, random_unit = get_random_unit(unit, "finance")
        print(random_unit)
        converted_value = convert_value(value, surrogate_initial_unit, random_unit, finance)
        if unit != surrogate_initial_unit:
            unit_new = f"{random_unit} {unit}"
            print(f"{value} {unit} to {unit_new}: {converted_value}")
        else:
            print(f"{value} {unit} to {random_unit}: {converted_value}")
    print(a)

    # Healthcare examples
    print("500 mg -> g =", convert_value(500, "mg", "g", healthcare))
    print("12 mL -> L =", convert_value(12, "mL", "L", healthcare))

    # Finance examples
    print("250 basis points -> percent =", convert_value(250, "basis_point", "percent", finance))
    print("3 million -> unit =", convert_value(3, "million", "unit", finance))

    # Environmental examples
    print("1 kg -> g =", convert_value(1, "kg", "g", environmental))
    print("2 kPa -> Pa =", convert_value(2, "kPa", "Pa", environmental))

    # Product examples (depends on QUDT + UNECE alias matches)
    if "Mass" in products:
        print("2 kilogram -> gram =", convert_value(2, "kg", "g", products))