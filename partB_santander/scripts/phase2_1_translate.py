"""
================================================================================
 PART B / STEP 2.1 : TRANSLATE SANTANDER DATA TO ENGLISH
 Smart Financial Cross-Selling Recommender System
================================================================================

 WHAT THIS DOES
     Reads train_ver2.csv (2.2 GB, Spanish column names) and writes a clean
     English CSV you can actually work with.

       - renames all 48 columns to English
       - translates the value codes too (V/H -> male/female, segment names)
       - samples N customers, keeping ALL 17 months for each one

 WHY SAMPLE BY CUSTOMER, NOT BY ROW
     This is the important design decision. If you took a random sample of
     ROWS you would get March for one customer, September for another, and
     the acquisition history would be shredded. Cross-selling is about
     sequence: "held a mortgage in March, added a pension in April". You need
     every month for whichever customers you keep.

 MEMORY
     Never loads the full 2.2 GB. Reads in chunks, writes as it goes.
     Works fine on 8 GB RAM.

 HOW TO RUN IN VSCODE
     1. Put this file next to train_ver2.csv
        (or set INPUT_PATH below)
     2. pip install pandas numpy
     3. Right-click -> "Run Python File in Terminal"   (or: python phase4_5a_translate.py)

     Takes roughly 3-8 minutes depending on your disk.

 OUTPUT
     santander_english.csv      translated, sampled, ready for analysis
     santander_column_map.json  the Spanish -> English mapping, for your report
================================================================================
"""

import os
import sys
import json
import time

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    print(f"\n[X] Missing package: {exc.name}")
    print("    pip install pandas numpy\n")
    sys.exit(1)


# ============================================================== CONFIG ====

INPUT_PATH = None
# None = auto-find train_ver2.csv near this script. Or hardcode:
# INPUT_PATH = r"C:\Users\jay4l\Desktop\RecommenderSysProject\data\raw\santander_bank\train_ver2.csv"

OUTPUT_PATH = os.path.join("..", "data", "santander_english.csv")

N_CUSTOMERS = 200_000
# How many customers to keep (ALL their months are kept).
# The full file has ~956,000 customers across 13.6M rows.
#      8 GB RAM  ->  100_000   (~1.4M rows out)
#     16 GB RAM  ->  200_000   (~2.8M rows out)   <- default
#     32 GB RAM  ->  400_000   or None for everything

CHUNK_SIZE = 500_000     # rows read at a time. Lower if you hit MemoryError.
SEED = 42

# =========================================================================


HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- mappings
# The 24 product ownership columns. Each is 0/1 = does this customer hold it
# in this month. These are the "items" in the user x item matrix that
# LendingClub could never give us.
PRODUCT_MAP = {
    "ind_ahor_fin_ult1": "prod_savings_account",
    "ind_aval_fin_ult1": "prod_guarantees",
    "ind_cco_fin_ult1":  "prod_current_account",
    "ind_cder_fin_ult1": "prod_derivatives_account",
    "ind_cno_fin_ult1":  "prod_payroll_account",
    "ind_ctju_fin_ult1": "prod_junior_account",
    "ind_ctma_fin_ult1": "prod_mas_particular_account",
    "ind_ctop_fin_ult1": "prod_particular_account",
    "ind_ctpp_fin_ult1": "prod_particular_plus_account",
    "ind_deco_fin_ult1": "prod_short_term_deposits",
    "ind_deme_fin_ult1": "prod_medium_term_deposits",
    "ind_dela_fin_ult1": "prod_long_term_deposits",
    "ind_ecue_fin_ult1": "prod_e_account",
    "ind_fond_fin_ult1": "prod_funds",
    "ind_hip_fin_ult1":  "prod_mortgage",
    "ind_plan_fin_ult1": "prod_pension_plan",
    "ind_pres_fin_ult1": "prod_loans",
    "ind_reca_fin_ult1": "prod_taxes",
    "ind_tjcr_fin_ult1": "prod_credit_card",
    "ind_valo_fin_ult1": "prod_securities",
    "ind_viv_fin_ult1":  "prod_home_account",
    "ind_nomina_ult1":   "prod_payroll",
    "ind_nom_pens_ult1": "prod_pensions",
    "ind_recibo_ult1":   "prod_direct_debit",
}

# The customer / demographic columns.
CUSTOMER_MAP = {
    "fecha_dato":            "snapshot_date",
    "ncodpers":              "customer_id",
    "ind_empleado":          "employee_index",
    "pais_residencia":       "country",
    "sexo":                  "sex",
    "age":                   "age",
    "fecha_alta":            "first_contract_date",
    "ind_nuevo":             "is_new_customer",
    "antiguedad":            "seniority_months",
    "indrel":                "is_primary_customer",
    "ult_fec_cli_1t":        "last_primary_date",
    "indrel_1mes":           "customer_type",
    "tiprel_1mes":           "relationship_type",
    "indresi":               "is_resident",
    "indext":                "is_foreigner",
    "conyuemp":              "is_employee_spouse",
    "canal_entrada":         "join_channel",
    "indfall":               "is_deceased",
    "tipodom":               "address_type",
    "cod_prov":              "province_code",
    "nomprov":               "province_name",
    "ind_actividad_cliente": "is_active",
    "renta":                 "gross_income",
    "segmento":              "segment",
}

COLUMN_MAP = {**CUSTOMER_MAP, **PRODUCT_MAP}

# Value codes are in Spanish too, not just the headers.
VALUE_MAPS = {
    "sex": {"V": "male", "H": "female"},          # Varon / Hembra
    "segment": {
        "01 - TOP":            "VIP",
        "02 - PARTICULARES":   "Individual",
        "03 - UNIVERSITARIO":  "College",
    },
    "is_resident":        {"S": 1, "N": 0},        # Si / No
    "is_foreigner":       {"S": 1, "N": 0},
    "is_deceased":        {"S": 1, "N": 0},
    "is_employee_spouse": {"S": 1, "N": 0},
    "relationship_type": {
        "A": "active", "I": "inactive", "P": "former",
        "R": "potential", "N": "none",
    },
    "employee_index": {
        "A": "active", "B": "ex_employee", "F": "filial",
        "N": "not_employee", "P": "passive",
    },
}


def banner(text, ch="="):
    print("\n" + ch * 78)
    print(text)
    print(ch * 78)


def find_input():
    if INPUT_PATH and os.path.exists(INPUT_PATH):
        return INPUT_PATH

    candidates = [
        "train_ver2.csv",
        os.path.join("data", "raw", "santander_bank", "train_ver2.csv"),
        os.path.join("santander_bank", "train_ver2.csv"),
        os.path.join("data", "train_ver2.csv"),
        os.path.join("..", "train_ver2.csv"),
        os.path.join("..", "..", "data", "raw", "stander again lol", "train_ver2.csv"),
        os.path.join("..", "..", "data", "raw", "santander_bank", "train_ver2.csv"),
    ]
    for c in candidates:
        p = c if os.path.isabs(c) else os.path.join(HERE, c)
        if os.path.exists(p):
            return p

    for root, _, files in os.walk(HERE):
        if "train_ver2.csv" in files:
            return os.path.join(root, "train_ver2.csv")
    return None


def clean_numeric(series):
    """Santander stores some numbers as padded strings like '     NA' or ' 35'.

    Plain astype(float) blows up on those, so strip and coerce instead.
    """
    return pd.to_numeric(
        series.astype(str).str.strip().replace({"NA": None, "": None}),
        errors="coerce",
    )


# ================================================================== MAIN ==
def main():
    t0 = time.time()
    banner("PART B / STEP 2.1 : TRANSLATING SANTANDER DATA TO ENGLISH")

    src = find_input()
    if src is None:
        print("\n[X] Could not find train_ver2.csv")
        print("    Put it next to this script, or set INPUT_PATH at the top.\n")
        sys.exit(1)

    size_gb = os.path.getsize(src) / 1024**3
    print(f"\nInput  : {src}")
    print(f"Size   : {size_gb:.2f} GB")
    print(f"Output : {os.path.join(HERE, OUTPUT_PATH)}")

    # ---------------------------------------------- pass 1: customer IDs
    banner("PASS 1 - collecting customer IDs", "-")
    print("\n  Reading only the customer_id column (cheap, no full load).\n")

    ids = set()
    rows_total = 0
    for chunk in pd.read_csv(src, usecols=["ncodpers"], chunksize=2_000_000):
        ids.update(chunk["ncodpers"].unique().tolist())
        rows_total += len(chunk)
        print(f"    scanned {rows_total:,} rows | {len(ids):,} customers",
              end="\r", flush=True)
    print(f"\n\n  Total rows      : {rows_total:,}")
    print(f"  Unique customers: {len(ids):,}")

    if N_CUSTOMERS and len(ids) > N_CUSTOMERS:
        rng = np.random.RandomState(SEED)
        keep = set(rng.choice(sorted(ids), N_CUSTOMERS, replace=False).tolist())
        print(f"  Sampling        : {len(keep):,} customers "
              f"(all their months kept)")
    else:
        keep = ids
        print(f"  Keeping all {len(keep):,} customers")

    # ------------------------------------------ pass 2: translate + write
    banner("PASS 2 - translating and writing", "-")
    print()

    out_path = os.path.join(HERE, OUTPUT_PATH)
    # Deletion is blocked on connected folders, so truncate via mode="w".

    product_cols = list(PRODUCT_MAP.values())
    written = 0
    first = True

    for chunk in pd.read_csv(src, chunksize=CHUNK_SIZE, low_memory=False):
        chunk = chunk[chunk["ncodpers"].isin(keep)]
        if chunk.empty:
            continue

        chunk = chunk.rename(columns=COLUMN_MAP)

        # numeric fields arrive as padded strings in places
        for col in ["age", "seniority_months", "gross_income"]:
            if col in chunk.columns:
                chunk[col] = clean_numeric(chunk[col])

        # product flags -> clean 0/1 ints
        for col in product_cols:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce") \
                               .fillna(0).astype("int8")

        # Spanish value codes -> English
        for col, mapping in VALUE_MAPS.items():
            if col in chunk.columns:
                chunk[col] = chunk[col].astype(str).str.strip().map(mapping) \
                             if col in ("sex", "segment", "relationship_type",
                                        "employee_index") \
                             else chunk[col].astype(str).str.strip().map(mapping)

        # handy derived column: how many products does this customer hold?
        have = [c for c in product_cols if c in chunk.columns]
        chunk["n_products_held"] = chunk[have].sum(axis=1)

        chunk.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
        first = False
        written += len(chunk)
        print(f"    written {written:,} rows", end="\r", flush=True)

    print(f"\n\n  Wrote {written:,} rows to {OUTPUT_PATH}")
    out_gb = os.path.getsize(out_path) / 1024**3
    print(f"  Output size: {out_gb:.2f} GB")

    # -------------------------------------------------------- save the map
    with open(os.path.join(HERE, "..", "data",
                           "santander_column_map.json"), "w") as f:
        json.dump({
            "customer_columns": CUSTOMER_MAP,
            "product_columns": PRODUCT_MAP,
            "value_translations": {
                k: {kk: str(vv) for kk, vv in v.items()}
                for k, v in VALUE_MAPS.items()
            },
        }, f, indent=2, ensure_ascii=False)
    print("  Saved santander_column_map.json  (put this in your appendix)")

    banner(f"DONE  ({time.time()-t0:.0f}s)   ->   now run  "
           f"python phase2_2_analysis.py")


if __name__ == "__main__":
    main()
