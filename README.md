# FinTrace — AP Payment Firewall & Fraud Investigation

FinTrace is a modular Python reference implementation for Accounts Payable risk scoring. It combines a supervised classifier, an unsupervised behavioral anomaly detector, deterministic policy rules, vendor text-similarity checks, a what-if simulator, and a FastAPI interface.

## What is implemented

- **Supervised pattern engine:** XGBoost with `scale_pos_weight = legitimate_training_rows / fraud_training_rows`. If XGBoost cannot be imported on a platform, a RandomForest fallback is selected and clearly reported.
- **Behavioral anomaly engine:** Isolation Forest with an empirical calibration over training-fold decision values, so the anomaly score is oriented from 0 (ordinary) to 1 (outlier).
- **Deterministic firewall rules:** Duplicate invoice, blacklisted/sanctioned vendor, and submitted-vs-registered bank account mismatch trigger an immediate `BLOCK` before model scoring. Bank identifiers are normalized and compared exactly; cosine similarity is not used for bank accounts.
- **Vendor identity similarity:** Fixed-dimension character n-gram vectors provide a near-name/address comparison signal. A low similarity leads to at least `HOLD`, not automatic proof of fraud. Optional PostgreSQL + pgvector stores profile vectors and a keyed HMAC of the registered bank-account value, not the raw account number.
- **Composite risk:** `(0.70 * P_XGBoost) + (0.30 * S_IsolationForest)`, with risk `< 0.35` = `APPROVE`, `0.35 <= risk < 0.75` = `HOLD`, and `risk >= 0.75` = `BLOCK`. Mandatory rules immediately return risk 1.0 / `BLOCK`.
- **XAI evidence:** Human-readable deterministic signals and, when the XGBoost backend is available, native per-row XGBoost contribution values. Contributions describe model influence; they are not causal proof.
- **What-if sandbox:** Merges proposed changes into an in-memory copy and returns before/after risk. It never updates the source invoice record.
- **Audit trail:** Local append-only JSONL records with a SHA-256 hash chain; invoice and supplier identifiers are hashed and raw account data is not written. This is tamper-evident, not equivalent to enterprise WORM storage.
- **Duplicate guard:** SQLite atomic uniqueness check for local/single-host development. Multi-replica production should use a PostgreSQL unique constraint/transaction.

## Project layout

```text
fintrace_ibm_project/
├── data/Procurement_Invoice_Fraud_Master.xlsx
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── inference_engine.py
│   ├── simulator.py
│   ├── vendor_similarity.py
│   ├── pgvector_store.py
│   ├── runtime_store.py
│   └── demo_data.py
├── api/app.py
├── models/
├── outputs/
├── requirements.txt
└── main.py
```

## Run in VS Code on Windows or Linux

Use Python 3.11 or 3.12 in a VS Code virtual environment.

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Linux on IBM Z / s390x, validate wheel availability for the exact OS/Python version before deployment. The current XGBoost PyPI wheel documentation lists x86_64 and aarch64 Linux wheels, not s390x; the requirements therefore skip XGBoost on s390x and the code uses its clearly reported RandomForest fallback there unless you install a compatible XGBoost build separately. Other compiled packages may also need platform-specific packages or source builds.

### 2. Train using the real workbook

Copy `Procurement_Invoice_Fraud_Master.xlsx` into the project's `data/` directory, then run from the project root:

```bash
python main.py
```

The loader validates `invoice_amount` and `is_fraud`, keeps invoice identifiers as strings, converts vendor/dept identifiers and low-cardinality fields to compact categorical storage, and logs data size and label prevalence. The pipeline then trains both models, prints metrics, writes `models/xgboost_fintrace.pkl` and `outputs/evaluation_metrics.json`, and runs one inference smoke test.

The workbook must contain a binary `is_fraud` target. Missing optional fields produce warnings and defaulted features; the loader does not silently deduplicate rows because repeated invoices may represent fraud evidence. Make sure features such as `duplicate_invoice_flag`, `supplier_risk_score`, and amount z-scores are computed from information available at decision time and are not derived from `is_fraud`, `fraud_type`, or `explanations`, otherwise results will be inflated by label leakage.

### 3. Smoke-test without the actual workbook

```bash
python main.py --demo
```

This intentionally uses synthetic records to prove the pipeline executes. It writes `models/xgboost_fintrace_demo.pkl` and `outputs/evaluation_metrics_demo.json`, leaving the production-named artifact untouched. These metrics are **not** results on your real dataset and must not be reported as model accuracy for production. The FastAPI app loads only `models/xgboost_fintrace.pkl`.

### 4. Start the REST API

After the model artifact exists:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Open the interactive API docs at `http://127.0.0.1:8000/docs`.

Endpoints:

- `GET /health` — application/model readiness.
- `POST /evaluate` — score and route one invoice. The local ledger blocks a repeated `invoice_id` on subsequent requests.
- `POST /simulate` — apply proposed field changes in memory and compare decisions.
- `GET /audit?limit=100&verify=true` — recent hash-chained events and optional integrity verification.
- `POST /vendor/profile` — optional pgvector vendor profile ingestion, available only when PostgreSQL is configured.

Example `/evaluate` JSON:

```json
{
  "invoice_id": "INV-10001",
  "supplier_id": "SUP-100",
  "invoice_amount": 97500,
  "currency": "INR",
  "invoice_date": "2026-10-17",
  "submission_hour": 2,
  "supplier_age_days": 18,
  "supplier_risk_score": 0.78,
  "supplier_invoice_count_30d": 14,
  "supplier_avg_amount_90d": 22000,
  "invoice_amount_zscore": 4.1,
  "split_invoice_flag": 1,
  "late_night_submission_flag": 1
}
```

Example `/simulate` JSON:

```json
{
  "transaction": {
    "invoice_id": "SIM-10002",
    "invoice_amount": 97500,
    "currency": "INR",
    "supplier_age_days": 22,
    "supplier_avg_amount_90d": 25000,
    "invoice_amount_zscore": 3.5
  },
  "changes": {
    "invoice_amount": 55000,
    "invoice_amount_zscore": 0.9
  }
}
```

## Optional PostgreSQL + pgvector vendor profiles

Set both environment variables before starting the API:

- `FINTRACE_DATABASE_URL` — PostgreSQL connection string for a database with permission to enable `vector` and create the profile table.
- `FINTRACE_BANK_HMAC_KEY` — high-entropy secret used to HMAC bank-account identifiers. Keep it in a secret manager; do not commit it to source control.

Then call `POST /vendor/profile` with `supplier_id`, `vendor_name`, optional address/country, and optional `registered_bank_account`. Raw account numbers and raw vendor address/name are not written by `PgVectorVendorStore`; it stores the embedding and keyed bank fingerprint. This protects against raw account retention but does not replace database encryption, access controls, rotation, backups, or compliance review.

For production, set `FINTRACE_API_KEY` to require an `X-API-Key` header on evaluation, simulation, audit, and profile write endpoints (unset means development mode), use TLS for PostgreSQL/API traffic, secrets management, a dedicated database role, PostgreSQL backups, row-level and network controls, central audit retention, rate limits, and explicit transaction/approval integration.

## IBM Z / LinuxONE deployment notes

This code is portable Python and can be containerized for an s390x Linux runtime when dependencies are available. It does **not** automatically invoke the Telum AI accelerator. The current XGBoost and Isolation Forest estimators run through their normal CPU runtime when XGBoost is installed. IBM documents optimized and accelerated serving paths including TensorFlow, PyTorch, Triton, Snap ML, and ONNX model compilation, but this project does not convert or route its hybrid model into one of those paths automatically. Telum acceleration requires a supported model format/runtime and a validated IBM AI Toolkit / Snap ML / IBM Z Deep Learning Compiler or other supported serving integration for the selected model. Benchmark it on the actual hardware; the sub-millisecond figure is a target, not a guarantee from this code.

Likewise, zCX is a container execution environment for Linux workloads hosted within z/OS, not simply a synonym for any Ubuntu VM. Follow the zCX and enterprise security team's deployment procedures. IBM Crypto Express/HSM, pervasive encryption, key management, and production-grade database encryption require host and middleware configuration; they are not enabled by importing these Python modules.

## Important limitations before production

1. **No actual workbook was bundled with this project.** Accuracy/precision/recall/ROC-AUC cannot be determined until you train and evaluate on the labelled workbook. The demo mode intentionally writes separate `*_demo` artifacts.
2. The workbook description lists 25 columns but does not list PO/GR identifiers, verified bank fields, vendor names, or historical relational tables. Those features can only be used when supplied or joined securely at inference time. The optional pgvector interface supports vendor profiles separately.
3. The 94,000–98,500 near-threshold rule is a heuristic limited to INR or unspecified currency data. Normalize currency before using amounts for thresholds; cross-currency values should never be compared without conversion.
4. The risk thresholds and model cutoffs are initial policy values and need finance/compliance approval, validation, threshold calibration, and monitoring for false positives/false negatives.
5. `BLOCK` is a decision response, not an actual payment freeze or settlement integration. Connect this API to an authenticated AP/payment workflow using an idempotent, audited downstream adapter before describing payment release/freeze as operational.
6. The local hash-chain audit log and SQLite duplicate guard are development defaults. Use a multi-node transactional store and externally protected immutable audit retention for a real enterprise deployment.
