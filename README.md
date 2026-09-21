# MediScope — Personalized Healthcare Recommendation System

A symptom-based disease prediction and care-recommendation system, built as a
real, runnable implementation of the project brief: a user reports symptoms,
the system predicts the most likely condition, and returns a personalized
recommendation (description, precautions, medications, diet, and activity
guidance) — the "Disease Prediction" + "Medicine Recommendation" components
from the original spec, served through a proper ML backend and web frontend.

This is not a mock-up. The model in this repo is trained on real data and the
API/UI have been run and tested end-to-end.

---

## 1. Why this dataset, and where it's from

The brief asked for real data, not synthetic data. This project uses a
public, widely-used symptom–disease dataset (the same one behind several
well-documented "Medical Recommendation System" student and portfolio
projects), consisting of eight CSV tables in `data/`:

| File | Rows | What it is |
|---|---|---|
| `Training.csv` | 4,920 | Patient records: 132 binary symptom columns → 1 disease label |
| `description.csv` | 41 | Plain-language description per disease |
| `precautions_df.csv` | 41 | Up to 4 precautions per disease |
| `medications.csv` | 41 | Common medications per disease |
| `diets.csv` | 41 | Suggested diet per disease |
| `workout_df.csv` | 240+ | Suggested activities per disease |
| `Symptom-severity.csv` | 133 | A severity weight per symptom |

**41 disease classes**, from common conditions (Common Cold, Migraine,
GERD, Diabetes, Hypertension) to more serious ones (Tuberculosis, Heart
attack, Hepatitis B/C/D/E, AIDS).

### Data cleaning that was actually necessary

Real datasets are never perfectly clean, and this one wasn't either. Two
issues were found and fixed while building this project (see
`src/train.py` and `src/recommend.py` for the exact fixes, with comments):

1. **`Training.csv` misspells "Peptic ulcer disease" as "Peptic ulcer
   diseae"** (missing the "s"), while the other tables spell it correctly —
   fixed by correcting the label before training, so the model's own class
   name is consistent everywhere.
2. **One disease name has inconsistent whitespace** (`"Paroymsal  Positional
   Vertigo"` with a double space in `Training.csv` vs. a single space
   elsewhere), and the same "diseae"/"disease" typo also appears in
   `precautions_df.csv` specifically — fixed with a normalization function
   used for every cross-table lookup, rather than a one-off patch.

There's a regression test (`tests/test_api.py::test_predict_every_disease_has_full_recommendation`)
that asserts all 41 diseases resolve against all five reference tables, so
this can't silently regress.

### A later audit pass also caught and fixed

- `top_k=0` (or negative) crashed the API with an unhandled 500 —
  `top_k` is now validated at the API layer (`1 ≤ top_k ≤ 41`) and clamped
  defensively inside `RecommendationEngine.predict` too, for callers that
  use it outside the API.
- Submitting the same symptom twice, or in different casing, showed up
  duplicated in the response's `resolved_symptoms` — now de-duplicated
  while preserving order.

Both have regression tests in `tests/test_api.py`.

---

## 2. Architecture

```
Browser  ──HTTP──►  FastAPI (api/main.py)  ──►  RecommendationEngine (src/recommend.py)
  (index.html)          │                              │
                         │                    ┌─────────┴─────────┐
                         │                    │  model.pkl (RF)    │
                         │                    │  label_encoder.pkl │
                         │                    │  data/*.csv        │
                         │                    └─────────────────────┘
                         └─► GET /api/symptoms, POST /api/predict, GET /api/health
```

- **`src/train.py`** — loads `Training.csv`, trains and compares 6
  classifiers with 5-fold cross-validation + a held-out test split, saves
  the best model plus metadata to `models/`.
- **`src/recommend.py`** — turns a symptom list into a full recommendation:
  resolves free-text symptoms → builds the model's input vector → predicts
  → joins against the 5 reference tables.
- **`api/main.py`** — FastAPI app: `/`, `/api/symptoms`, `/api/predict`,
  `/api/health`.
- **`api/templates/index.html`** — self-contained HTML/CSS/JS frontend
  (symptom autocomplete, results card). No build step, no framework.
- **`tests/test_api.py`** — automated tests (`pytest`).

---

## 3. Model performance (as actually trained in this repo)

| Model | 5-fold CV accuracy | Held-out test accuracy | Test F1 (weighted) | Train time |
|---|---|---|---|---|
| **Random Forest** ✅ chosen | 1.000 | 1.000 | 1.000 | 5.3s |
| SVC (linear) | 1.000 | 1.000 | 1.000 | 3.1s |
| KNN | 1.000 | 1.000 | 1.000 | 0.5s |
| Naive Bayes | 1.000 | 1.000 | 1.000 | 0.2s |
| Gradient Boosting | 0.9995 | 1.000 | 1.000 | 110.4s |
| Decision Tree | 0.9995 | 1.000 | 1.000 | 0.2s |

**Be honest about what this means.** Every model scores ~100%, including
a single decision tree — that's a signal about the *dataset*, not a
triumph of modeling. Each disease in this dataset has a fixed, distinctive
combination of symptoms repeated across its 120 records, so the classes are
linearly separable and the task is close to a lookup table. This is
realistic for teaching the full pipeline (data → model → recommendation →
API → deployment) but it is **not** how real clinical presentation works:
real patients have overlapping, noisy, and incomplete symptom sets, and a
production system would need a probabilistic, uncertainty-aware model
evaluated on a harder, messier dataset. Random Forest was still picked over
the other 100%-scorers because it degrades more gracefully than a single
Decision Tree or Naive Bayes when it later sees held-out patients whose
symptom combination it hasn't seen — the `predict_proba` output is also
usable for the "also considered" secondary predictions shown in the UI,
which cheaper models like an unweighted KNN don't give you as reliably.

---

## 4. Run it locally

```bash
# 1. create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. train the model (writes models/*.pkl + models/metrics.json)
python src/train.py

# 4. run the API + frontend
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000** — type a few symptoms (e.g. `itching`,
`high_fever`, `stomach_pain`), select from the dropdown, and click "Check
symptoms".

Interactive API docs (Swagger UI) are auto-generated at
**http://localhost:8000/docs**.

Run the test suite:

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## 5. Deployment

### Option A — Docker (works anywhere: your own server, a VPS, or any
container host)

```bash
docker build -t mediscope .
docker run -p 8000:8000 mediscope
```

The `Dockerfile` trains the model at image-build time, so the resulting
image is fully self-contained — no separate training step needed at
runtime.

### Option B — Render.com (free tier, no credit card, easiest for a
student/portfolio deployment)

1. Push this project to a GitHub repo.
2. On [render.com](https://render.com) → **New → Web Service** → connect
   your repo.
3. Render auto-detects the `Dockerfile` — leave the build settings as-is
   (or if you'd rather not use Docker, set **Build Command** to
   `pip install -r requirements.txt && python src/train.py` and **Start
   Command** to `uvicorn api.main:app --host 0.0.0.0 --port $PORT`).
4. Deploy. You'll get a public URL like `https://mediscope.onrender.com`.

### Option C — Railway.app / Fly.io

Both auto-detect the `Dockerfile` the same way as Render — connect the
repo, deploy, done. Fly.io needs `fly launch` once from the CLI to
generate a `fly.toml`; Railway is entirely web-UI driven.

### Option D — Hugging Face Spaces (free, good if you want it discoverable)

Create a new Space → SDK: **Docker** → push this repo to the Space's git
remote. HF Spaces builds the `Dockerfile` automatically and exposes it on
a `*.hf.space` URL.

> Whichever host you pick, the important part is already done here: the
> `Dockerfile` + `requirements.txt` make the app runnable with a single
> build command, so "deployment" is really just "point a host at this
> repo."

---

## 6. Project structure

```
mediscope/
├── data/                     # real dataset (see §1)
├── models/                   # generated by train.py — model + metadata
├── src/
│   ├── train.py               # trains & compares 6 classifiers
│   └── recommend.py           # symptom → disease → full recommendation
├── api/
│   ├── main.py                 # FastAPI app
│   ├── templates/index.html    # frontend
│   └── static/                 # static assets mount point
├── tests/
│   └── test_api.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 7. Honest limitations & next steps

This delivers the **disease-prediction + medicine-recommendation** core
of the original brief end-to-end and for real. It does not yet cover
everything the original concept document sketched out — worth knowing if
you're presenting this as a capstone:

- **Not a diagnosis tool.** It's trained on a clean, curated teaching
  dataset (see §3's honesty note) — it should be framed as an educational
  demo, and the UI/API already carry that disclaimer.
- **No user accounts / auth**, activity tracking, or analytics dashboard
  (the brief's "Data Collection and User Management" and "Dashboard &
  Reporting" sections) — this build focuses on getting the ML core
  genuinely correct and deployable first; that layer is a natural next
  phase (FastAPI + a users table + JWT auth is a straightforward addition
  on top of what's here).
- **No collaborative filtering / NLP sentiment analysis on reviews** — the
  original brief's e-commerce-style content-based/collaborative hybrid
  doesn't map cleanly onto structured symptom data the way it does onto
  product catalogs; those techniques are better suited to a follow-on
  "similar patients" feature (e.g. clustering on symptom vectors) if you
  want to extend this.
- **No vitals-based prediction** (age/blood pressure/glucose → diagnosis,
  from the original doc) — that needs a different dataset with continuous
  clinical measurements (e.g. a diabetes or heart-disease risk dataset)
  and is a good "Phase 2" module rather than bolted onto this one, since
  it's a genuinely different prediction task.

If you want, the next concrete step I'd suggest is deciding which of those
three you actually need for your deadline/rubric, and I can build that
module next using the same real-data standard as this one.




DEPLOYMENT LINK: https://midicaldio.onrender.com
