"""
recommend.py
------------
Loads the trained model plus the four "knowledge base" CSVs (description,
precautions, medications, diets, workouts) and turns a list of symptoms into
a full personalized recommendation.

This is the "content" side of the recommendation system described in the
project brief: given a predicted class (disease), pull structured guidance
for that class from real reference tables rather than generating anything.
"""

import ast
import json
from pathlib import Path
from typing import List

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


def _parse_list_cell(cell) -> List[str]:
    """medications.csv / diets.csv store list-like strings, e.g. "['A', 'B']"."""
    if pd.isna(cell):
        return []
    try:
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return [str(cell).strip()]


class RecommendationEngine:
    def __init__(self):
        self.model = joblib.load(MODELS_DIR / "model.pkl")
        self.encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
        with open(MODELS_DIR / "feature_columns.json") as f:
            self.feature_columns = json.load(f)

        self.description = pd.read_csv(DATA_DIR / "description.csv")
        self.precautions = pd.read_csv(DATA_DIR / "precautions_df.csv")
        self.medications = pd.read_csv(DATA_DIR / "medications.csv")
        self.diets = pd.read_csv(DATA_DIR / "diets.csv")
        self.workouts = pd.read_csv(DATA_DIR / "workout_df.csv")
        severity = pd.read_csv(DATA_DIR / "Symptom-severity.csv")

        self.severity_map = dict(
            zip(severity["Symptom"].str.strip(), severity["weight"])
        )

        # Normalize the symptom vocabulary once so lookups are forgiving of
        # spaces/case/underscores coming from user input.
        self._symptom_lookup = {
            self._normalize(col): col for col in self.feature_columns
        }

    @staticmethod
    def _normalize(s: str) -> str:
        return s.strip().lower().replace(" ", "_").replace("__", "_")

    def known_symptoms(self) -> List[str]:
        return sorted(self.feature_columns)

    def resolve_symptoms(self, symptoms: List[str]):
        """Map free-text symptom strings to valid feature names.

        Returns (resolved, unresolved).
        """
        resolved, unresolved = [], []
        for s in symptoms:
            key = self._normalize(s)
            if key in self._symptom_lookup:
                resolved.append(self._symptom_lookup[key])
            else:
                unresolved.append(s)
        return resolved, unresolved

    def _vector_for(self, resolved_symptoms: List[str]) -> pd.DataFrame:
        row = {col: 0 for col in self.feature_columns}
        for s in resolved_symptoms:
            row[s] = 1
        return pd.DataFrame([row], columns=self.feature_columns)

    @staticmethod
    def _normalize_disease_name(name: str) -> str:
        # The source tables aren't perfectly consistent with each other:
        # - one entry ships as "Paroymsal  Positional Vertigo" with a double
        #   space in Training.csv but a single space everywhere else;
        # - "Peptic ulcer disease" is misspelled "diseae" in Training.csv AND
        #   in precautions_df.csv, but spelled correctly in description.csv,
        #   medications.csv, diets.csv and workout_df.csv.
        # Collapse whitespace/case and correct the known typo before
        # comparing, so lookups don't silently fail for these rows.
        normalized = " ".join(str(name).strip().lower().split())
        return normalized.replace("diseae", "disease")

    def _lookup_disease_field(self, df: pd.DataFrame, key_col: str, disease: str):
        target = self._normalize_disease_name(disease)
        match = df[df[key_col].apply(self._normalize_disease_name) == target]
        return match.iloc[0] if not match.empty else None

    def predict(self, symptoms: List[str], top_k: int = 3):
        # Defensive bound: top_k=0 previously produced an empty prediction
        # list and then crashed with an IndexError when we read
        # top_predictions[0]; a negative top_k silently returned every class
        # instead of erroring. Clamp here too (not just in the API layer)
        # since this method is also called directly, e.g. in tests/scripts.
        n_classes = len(self.encoder.classes_)
        top_k = max(1, min(top_k, n_classes))

        # De-duplicate while preserving order: the same symptom submitted
        # twice (or in different casing) shouldn't appear twice in the
        # response.
        symptoms = list(dict.fromkeys(symptoms))

        resolved, unresolved = self.resolve_symptoms(symptoms)
        if not resolved:
            raise ValueError(
                "None of the provided symptoms matched the known symptom list."
            )
        resolved = list(dict.fromkeys(resolved))

        X = self._vector_for(resolved)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            top_idx = proba.argsort()[::-1][:top_k]
            top_predictions = [
                {
                    "disease": self.encoder.inverse_transform([i])[0],
                    "confidence": round(float(proba[i]), 4),
                }
                for i in top_idx
            ]
        else:
            pred = self.model.predict(X)[0]
            top_predictions = [
                {"disease": self.encoder.inverse_transform([pred])[0], "confidence": None}
            ]

        top_disease = top_predictions[0]["disease"]

        desc_row = self._lookup_disease_field(self.description, "Disease", top_disease)
        prec_row = self._lookup_disease_field(self.precautions, "Disease", top_disease)
        med_row = self._lookup_disease_field(self.medications, "Disease", top_disease)
        diet_row = self._lookup_disease_field(self.diets, "Disease", top_disease)

        precautions = []
        if prec_row is not None:
            for col in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]:
                if col in prec_row and pd.notna(prec_row[col]) and str(prec_row[col]).strip():
                    precautions.append(str(prec_row[col]).strip())

        target = self._normalize_disease_name(top_disease)
        workouts = (
            self.workouts[self.workouts["disease"].apply(self._normalize_disease_name)
                           == target]["workout"]
            .dropna().tolist()
        )

        severity_score = sum(self.severity_map.get(s, 0) for s in resolved)

        return {
            "input_symptoms": symptoms,
            "resolved_symptoms": resolved,
            "unresolved_symptoms": unresolved,
            "symptom_severity_score": severity_score,
            "top_predictions": top_predictions,
            "disease": top_disease,
            "description": desc_row["Description"] if desc_row is not None else None,
            "precautions": precautions,
            "medications": _parse_list_cell(med_row["Medication"]) if med_row is not None else [],
            "diet": _parse_list_cell(diet_row["Diet"]) if diet_row is not None else [],
            "workout": workouts[:6],
            "disclaimer": (
                "This is an educational ML demo, not a medical diagnosis. "
                "Always consult a licensed clinician for real symptoms."
            ),
        }


if __name__ == "__main__":
    engine = RecommendationEngine()
    result = engine.predict(["itching", "skin_rash", "nodal_skin_eruptions"])
    print(json.dumps(result, indent=2))
