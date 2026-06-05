from __future__ import annotations

import argparse
import bisect
import ipaddress
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


REQUIRED_FIELDS = [
    "signup_time",
    "purchase_time",
    "purchase_value",
    "device_id",
    "source",
    "browser",
    "age",
    "ip_address",
]


def project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def default_model_path() -> Path:
    return project_dir() / "outputs" / "model" / "fraud_profit_model.joblib"


def parse_ip(value: Any) -> int:
    text = str(value).strip()
    if "." in text:
        return int(ipaddress.ip_address(text))
    return int(float(text))


def parse_time(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, format="%Y-%m-%d %H:%M:%S", errors="raise")
    return pd.Timestamp(ts)


def prompt_transaction() -> dict[str, Any]:
    print("\nSaisie transaction - format date attendu : YYYY-MM-DD HH:MM:SS")
    print("Appuyer sur Entree pour utiliser l'exemple entre crochets.\n")
    defaults = {
        "signup_time": "2015-01-01 18:52:44",
        "purchase_time": "2015-01-01 18:52:45",
        "purchase_value": "120",
        "device_id": "YSSKYOSJHPPLJ",
        "source": "SEO",
        "browser": "Opera",
        "sex": "M",
        "age": "53",
        "ip_address": "2621473820",
    }
    values: dict[str, Any] = {}
    for field in [
        "signup_time",
        "purchase_time",
        "purchase_value",
        "device_id",
        "source",
        "browser",
        "sex",
        "age",
        "ip_address",
    ]:
        raw = input(f"{field} [{defaults[field]}] : ").strip()
        values[field] = raw if raw else defaults[field]
    return values


def validate(raw: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in raw or str(raw[field]).strip() == ""]
    if missing:
        raise ValueError(f"Champs manquants : {missing}")

    signup_time = parse_time(raw["signup_time"])
    purchase_time = parse_time(raw["purchase_time"])
    delay_seconds = (purchase_time - signup_time).total_seconds()
    if delay_seconds < 0:
        raise ValueError("purchase_time doit etre posterieur ou egal a signup_time.")

    return {
        "signup_time": signup_time,
        "purchase_time": purchase_time,
        "purchase_value": float(raw["purchase_value"]),
        "device_id": str(raw["device_id"]),
        "source": str(raw["source"]),
        "browser": str(raw["browser"]),
        "sex": str(raw.get("sex", "")),
        "age": int(float(raw["age"])),
        "ip_address": parse_ip(raw["ip_address"]),
        "delay_seconds": float(delay_seconds),
    }


def country_from_ip(ip_value: int, artifact: dict[str, Any]) -> tuple[str, str]:
    geo = artifact["geo"]
    lower = geo["ip_lower"]
    upper = geo["ip_upper"]
    countries = geo["ip_country"]
    idx = int(np.searchsorted(lower, ip_value, side="right") - 1)
    if idx < 0 or ip_value > int(upper[idx]):
        raw_country = "Unknown"
    else:
        raw_country = str(countries[idx])

    if raw_country in geo["top_countries"]:
        model_country = raw_country
    elif raw_country == "Unknown" and "Unknown" in geo["top_countries"]:
        model_country = "Unknown"
    else:
        model_country = "Other"
    return raw_country, model_country


def count_before(history: dict[Any, list[int]], key: Any, purchase_time: pd.Timestamp) -> int:
    values = history.get(key, [])
    return int(bisect.bisect_left(values, int(purchase_time.value)))


def build_features(raw: dict[str, Any], artifact: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    tx = validate(raw)
    raw_country, model_country = country_from_ip(tx["ip_address"], artifact)
    purchase_time = tx["purchase_time"]

    prev_tx_by_device = count_before(artifact["device_history"], tx["device_id"], purchase_time)
    prev_tx_by_ip = count_before(artifact["ip_history"], tx["ip_address"], purchase_time)

    row = {
        "purchase_value": tx["purchase_value"],
        "age": tx["age"],
        "delay_seconds": tx["delay_seconds"],
        "purchase_hour": int(purchase_time.hour),
        "purchase_weekday": int(purchase_time.dayofweek),
        "is_weekend": int(purchase_time.dayofweek in [5, 6]),
        "is_night": int(purchase_time.hour in [0, 1, 2, 3, 4, 5, 22, 23]),
        "signup_to_purchase_under_1min": int(tx["delay_seconds"] < 60),
        "prev_tx_by_device": prev_tx_by_device,
        "prev_tx_by_ip": prev_tx_by_ip,
        "device_seen_before": int(prev_tx_by_device > 0),
        "ip_seen_before": int(prev_tx_by_ip > 0),
        "source": tx["source"],
        "browser": tx["browser"],
        "country": model_country,
    }
    diagnostics = {
        "raw_country": raw_country,
        "model_country": model_country,
        "delay_seconds": tx["delay_seconds"],
        "prev_tx_by_device": prev_tx_by_device,
        "prev_tx_by_ip": prev_tx_by_ip,
        "sex_ignored": tx.get("sex", ""),
    }
    x = pd.DataFrame([row], columns=artifact["features"])
    return x, diagnostics


def heuristic_reasons(x: pd.DataFrame, diagnostics: dict[str, Any], artifact: dict[str, Any]) -> list[str]:
    row = x.iloc[0]
    reasons: list[str] = []
    if row["signup_to_purchase_under_1min"] == 1:
        reasons.append("achat moins d'une minute apres la creation du compte")
    if row["prev_tx_by_device"] > 0:
        reasons.append(f"appareil deja observe {int(row['prev_tx_by_device'])} fois avant cette transaction")
    if row["prev_tx_by_ip"] > 0:
        reasons.append(f"adresse IP deja observee {int(row['prev_tx_by_ip'])} fois avant cette transaction")
    if diagnostics["raw_country"] in artifact["geo"].get("high_risk_countries", []):
        reasons.append(f"pays a lift eleve dans l'entrainement : {diagnostics['raw_country']}")
    if row["is_night"] == 1:
        reasons.append("achat effectue sur une tranche horaire de nuit")
    if not reasons:
        reasons.append("aucun signal metier simple tres fort ; decision surtout issue de la combinaison du modele")
    return reasons[:4]


def predict(raw: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    x, diagnostics = build_features(raw, artifact)
    x_t = artifact["preprocessor"].transform(x)
    proba = float(artifact["classifier"].predict_proba(x_t)[0, 1])
    threshold = float(artifact["threshold"])
    predicted_class = int(proba >= threshold)
    decision = "BLOQUER / REVUE HUMAINE" if predicted_class == 1 else "ACCEPTER"
    reasons = heuristic_reasons(x, diagnostics, artifact)
    value = float(x.iloc[0]["purchase_value"])

    return {
        "probability": proba,
        "threshold": threshold,
        "predicted_class": predicted_class,
        "decision": decision,
        "features": x.iloc[0].to_dict(),
        "diagnostics": diagnostics,
        "reasons": reasons,
        "financial_if_legit": -15.0 if proba >= threshold else 0.10 * value,
        "financial_if_fraud": value if proba >= threshold else -1.00 * value,
    }


def print_result(result: dict[str, Any], model_type: str) -> None:
    print("\n" + "=" * 72)
    print("PREDICTION LIVE - MODELE FRAUDE")
    print("=" * 72)
    print(f"Modele charge          : {model_type}")
    print(f"Probabilite de fraude  : {100 * result['probability']:.2f} %")
    print(f"Seuil operationnel     : {result['threshold']:.2f}")
    print(f"Classe predite         : {result['predicted_class']} (1=fraude, 0=legitime)")
    print(f"Decision               : {result['decision']}")
    print("\nContexte calcule :")
    print(f"  pays IP brut/modelise : {result['diagnostics']['raw_country']} -> {result['diagnostics']['model_country']}")
    print(f"  delai signup->achat   : {result['diagnostics']['delay_seconds']:.0f} secondes")
    print(f"  device deja vu avant  : {result['diagnostics']['prev_tx_by_device']}")
    print(f"  IP deja vue avant      : {result['diagnostics']['prev_tx_by_ip']}")
    if result["diagnostics"].get("sex_ignored"):
        print("  sex                   : ignore par le modele (choix RGPD / faible signal)")
    print("\nRaisons lisibles :")
    for reason in result["reasons"]:
        print(f"  - {reason}")
    print("\nImpact financier potentiel :")
    print(f"  si client legitime : {result['financial_if_legit']:,.2f} $")
    print(f"  si fraude reelle   : {result['financial_if_fraud']:,.2f} $")
    print("=" * 72)


def load_input(args: argparse.Namespace, artifact: dict[str, Any]) -> dict[str, Any]:
    if args.sample:
        return artifact["samples"][args.sample]
    if args.json:
        return json.loads(args.json)
    if args.file:
        return json.loads(Path(args.file).read_text(encoding="utf-8"))
    return prompt_transaction()


def predict_csv(csv_path: Path, artifact: dict[str, Any], out_path: Path | None = None) -> pd.DataFrame:
    """Score one or several raw transactions stored in a CSV file.

    The CSV may contain the original Fraud_Data columns, including optional
    columns such as user_id, sex or class. Extra columns are preserved.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Fichier CSV vide : {csv_path}")

    records: list[dict[str, Any]] = []
    for idx, raw in enumerate(df.to_dict(orient="records")):
        try:
            result = predict(raw, artifact)
            record = {
                "row_index": idx,
                "probability_fraud": result["probability"],
                "threshold": result["threshold"],
                "predicted_class": result["predicted_class"],
                "decision": result["decision"],
                "raw_country": result["diagnostics"]["raw_country"],
                "model_country": result["diagnostics"]["model_country"],
                "delay_seconds": result["diagnostics"]["delay_seconds"],
                "prev_tx_by_device": result["diagnostics"]["prev_tx_by_device"],
                "prev_tx_by_ip": result["diagnostics"]["prev_tx_by_ip"],
                "reasons": " | ".join(result["reasons"]),
            }
            if "class" in raw and str(raw["class"]).strip() != "":
                record["actual_class"] = int(float(raw["class"]))
                record["is_correct"] = int(record["predicted_class"] == record["actual_class"])
            records.append(record)
        except Exception as exc:
            records.append({
                "row_index": idx,
                "error": f"{type(exc).__name__}: {exc}",
            })

    scored = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(records).drop(columns=["row_index"], errors="ignore")],
        axis=1,
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out_path, index=False, encoding="utf-8-sig")
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Live fraud prediction demo for STA218.")
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--json", type=str, default=None, help="Transaction as JSON string.")
    parser.add_argument("--file", type=Path, default=None, help="Transaction JSON file.")
    parser.add_argument("--csv", type=Path, default=None, help="CSV file with one or several raw transactions.")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path when using --csv.")
    parser.add_argument("--sample", choices=["fraud_high_score", "legit_low_score"], default=None)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Modele introuvable : {args.model}\n"
            "Lancer d'abord : python scripts/train_model_for_demo.py"
        )
    artifact = joblib.load(args.model)
    if args.csv:
        scored = predict_csv(args.csv, artifact, args.out)
        display_cols = [
            c for c in [
                "probability_fraud", "threshold", "predicted_class", "actual_class",
                "is_correct", "decision", "raw_country", "delay_seconds",
                "prev_tx_by_device", "prev_tx_by_ip", "reasons", "error",
            ]
            if c in scored.columns
        ]
        print(scored[display_cols].to_string(index=False))
        if args.out:
            print(f"\nPredictions sauvegardees : {args.out}")
        return

    raw = load_input(args, artifact)
    result = predict(raw, artifact)
    print_result(result, artifact["model_type"])


if __name__ == "__main__":
    main()
