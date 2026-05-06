import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

from data_generator import get_full_dataset


# ─────────────────────────────────────────────
#  LEVEL 1: POPULATION RISK STRATIFICATION
# ─────────────────────────────────────────────
class RiskStratificationModel:
    """Identifies high-risk women in the general population."""

    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)
        self.scaler = StandardScaler()
        self.le_activity = LabelEncoder()
        self.le_density = LabelEncoder()
        self.feature_names = []
        self.is_trained = False

    def _prepare_features(self, df, fit=False):
        activity_map = {"Low": 0, "Moderate": 1, "High": 2}
        density_map = {"A": 0, "B": 1, "C": 2, "D": 3}

        features = pd.DataFrame({
            "age": df["age"],
            "bmi": df["bmi"],
            "parity": df["parity"],
            "age_first_birth": df["age_first_birth"],
            "breastfeeding_months": df["breastfeeding_months"],
            "menarche_age": df["menarche_age"],
            "menopause": df["menopause"],
            "family_history": df["family_history"],
            "brca_mutation": df["brca_mutation"],
            "contraceptive_years": df["contraceptive_years"],
            "hrt_use": df["hrt_use"],
            "alcohol_use": df["alcohol_use"],
            "smoking": df["smoking"],
            "prev_biopsy": df["prev_biopsy"],
            "physical_activity_enc": df["physical_activity"].map(activity_map).fillna(0),
            "breast_density_enc": df["breast_density"].map(density_map).fillna(1),
        })
        self.feature_names = features.columns.tolist()
        return features

    def train(self, df):
        X = self._prepare_features(df, fit=True)
        y = df["has_cancer"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_prob)
        return {"auc": round(auc, 3), "n_train": len(X_train), "n_test": len(X_test)}

    def predict_risk(self, patient_dict):
        """Predict cancer risk for a single patient."""
        df = pd.DataFrame([patient_dict])
        X = self._prepare_features(df)
        prob = self.model.predict_proba(X)[0][1]
        if prob >= 0.7:
            risk_level = "High"
        elif prob >= 0.4:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        return {
            "risk_probability": round(float(prob), 3),
            "risk_level": risk_level,
            "recommendation": self._get_recommendation(risk_level),
            "top_risk_factors": self._get_top_factors(X)
        }

    def _get_recommendation(self, risk_level):
        recs = {
            "High": "Immediate referral for mammogram + breast MRI. Genetic counseling recommended.",
            "Moderate": "Annual mammogram screening. Clinical breast exam every 6 months.",
            "Low": "Routine annual breast exam. Mammogram every 2 years after age 40."
        }
        return recs[risk_level]

    def _get_top_factors(self, X):
        importances = self.model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:5]
        return [self.feature_names[i] for i in top_idx]

    def get_population_stats(self, df):
        X = self._prepare_features(df)
        probs = self.model.predict_proba(X)[:, 1]
        risk_levels = np.where(probs >= 0.7, "High", np.where(probs >= 0.4, "Moderate", "Low"))
        return {
            "total": len(df),
            "high_risk": int((risk_levels == "High").sum()),
            "moderate_risk": int((risk_levels == "Moderate").sum()),
            "low_risk": int((risk_levels == "Low").sum()),
            "probs": probs
        }


# ─────────────────────────────────────────────
#  LEVEL 2: IMAGING AI (Screening)
# ─────────────────────────────────────────────
class ImagingAIModel:
    """AI model to interpret mammogram/ultrasound features."""

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        self.is_trained = False

    def _prepare_features(self, df):
        shape_map = {"None": 0, "Round": 1, "Oval": 2, "Lobular": 3, "Irregular": 4}
        margin_map = {"None": 0, "Circumscribed": 1, "Microlobulated": 2, "Indistinct": 3, "Spiculated": 4}

        features = pd.DataFrame({
            "birads_category": df["birads_category"],
            "mass_present": df["mass_present"],
            "mass_shape_enc": df["mass_shape"].map(shape_map).fillna(0),
            "mass_margin_enc": df["mass_margin"].map(margin_map).fillna(0),
            "calcifications": df["calcifications"],
            "skin_thickening": df["skin_thickening"],
            "nipple_retraction": df["nipple_retraction"],
            "ai_malignancy_probability": df["ai_malignancy_probability"],
        })
        return features

    def train(self, df):
        X = self._prepare_features(df)
        y = df["has_cancer"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_prob)
        return {"auc": round(auc, 3)}

    def analyze_imaging(self, imaging_dict):
        df = pd.DataFrame([imaging_dict])
        X = self._prepare_features(df)
        prob = self.model.predict_proba(X)[0][1]
        birads = imaging_dict.get("birads_category", 1)
        
        if prob >= 0.75 or birads >= 4:
            flag = "SUSPICIOUS - Biopsy Recommended"
            color = "red"
        elif prob >= 0.45 or birads == 3:
            flag = "INDETERMINATE - Short-interval Follow-up"
            color = "orange"
        else:
            flag = "BENIGN - Routine Screening"
            color = "green"

        return {
            "malignancy_probability": round(float(prob), 3),
            "flag": flag,
            "color": color,
            "birads_interpretation": self._interpret_birads(birads),
            "findings_summary": self._summarize_findings(imaging_dict)
        }

    def _interpret_birads(self, birads):
        interp = {
            1: "Negative – No abnormality found",
            2: "Benign – Definitely benign finding",
            3: "Probably Benign – Short interval follow-up suggested",
            4: "Suspicious – Biopsy should be considered",
            5: "Highly Suggestive of Malignancy – Biopsy required"
        }
        return interp.get(birads, "Unknown")

    def _summarize_findings(self, d):
        findings = []
        if d.get("mass_present"): findings.append(f"{d.get('mass_shape', '')} mass with {d.get('mass_margin', '')} margins")
        if d.get("calcifications"): findings.append("Microcalcifications present")
        if d.get("skin_thickening"): findings.append("Skin thickening noted")
        if d.get("nipple_retraction"): findings.append("Nipple retraction observed")
        return findings if findings else ["No significant findings"]


# ─────────────────────────────────────────────
#  LEVEL 3: DIAGNOSIS & STAGING
# ─────────────────────────────────────────────
class DiagnosisModel:
    """Confirms cancer presence, grades tumor, predicts molecular subtype."""

    def __init__(self):
        self.cancer_model = GradientBoostingClassifier(n_estimators=150, random_state=42)
        self.stage_model = GradientBoostingClassifier(n_estimators=150, random_state=42)
        self.is_trained = False

    def _prepare_features(self, df):
        shape_map = {"None": 0, "Round": 1, "Oval": 2, "Lobular": 3, "Irregular": 4}
        margin_map = {"None": 0, "Circumscribed": 1, "Microlobulated": 2, "Indistinct": 3, "Spiculated": 4}

        features = pd.DataFrame({
            "age": df["age"],
            "bmi": df["bmi"],
            "family_history": df["family_history"],
            "brca_mutation": df["brca_mutation"],
            "birads_category": df["birads_category"],
            "mass_present": df["mass_present"],
            "mass_shape_enc": df["mass_shape"].map(shape_map).fillna(0),
            "mass_margin_enc": df["mass_margin"].map(margin_map).fillna(0),
            "calcifications": df["calcifications"],
            "ai_malignancy_probability": df["ai_malignancy_probability"],
            "er_positive": df["er_positive"],
            "pr_positive": df["pr_positive"],
            "her2_positive": df["her2_positive"],
            "ki67_percent": df["ki67_percent"],
            "ca153_u_ml": df["ca153_u_ml"],
            "lymph_node_involvement": df["lymph_node_involvement"],
        })
        return features

    def train(self, df):
        X = self._prepare_features(df)
        y_cancer = df["has_cancer"]
        cancer_mask = df["has_cancer"] == 1
        y_stage = df.loc[cancer_mask, "cancer_stage"]

        X_train, X_test, y_train, y_test = train_test_split(X, y_cancer, test_size=0.2, random_state=42)
        self.cancer_model.fit(X_train, y_train)

        X_stage = X[cancer_mask]
        X_str, X_ste, y_str, y_ste = train_test_split(X_stage, y_stage, test_size=0.2, random_state=42)
        self.stage_model.fit(X_str, y_str)

        self.is_trained = True
        auc = roc_auc_score(y_test, self.cancer_model.predict_proba(X_test)[:, 1])
        return {"auc": round(auc, 3)}

    def diagnose(self, patient_data):
        df = pd.DataFrame([patient_data])
        X = self._prepare_features(df)

        cancer_prob = self.cancer_model.predict_proba(X)[0][1]
        stage_probs = self.stage_model.predict_proba(X)[0]
        predicted_stage = self.stage_model.predict(X)[0]

        subtype = self._determine_subtype(patient_data)
        grade = self._determine_grade(patient_data)

        return {
            "cancer_probability": round(float(cancer_prob), 3),
            "predicted_stage": int(predicted_stage),
            "stage_probabilities": {
                f"Stage {i+1}": round(float(p), 3) for i, p in enumerate(stage_probs)
            },
            "molecular_subtype": subtype,
            "tumor_grade": grade,
            "lymph_node_status": "Positive" if patient_data.get("lymph_node_involvement") else "Negative",
            "prognosis": self._get_prognosis(predicted_stage, subtype),
        }

    def _determine_subtype(self, d):
        er = d.get("er_positive", 0)
        pr = d.get("pr_positive", 0)
        her2 = d.get("her2_positive", 0)
        ki67 = d.get("ki67_percent", 0)

        if er and not her2:
            return "Luminal A (ER+/PR+, HER2-, Low Ki-67)" if ki67 < 20 else "Luminal B (ER+, HER2-, High Ki-67)"
        elif er and her2:
            return "Luminal B HER2+ (ER+, HER2+)"
        elif her2 and not er:
            return "HER2-Enriched (ER-, HER2+)"
        else:
            return "Triple Negative (ER-, PR-, HER2-)"

    def _determine_grade(self, d):
        ki67 = d.get("ki67_percent", 0)
        margin = d.get("mass_margin", "")
        if ki67 > 30 or margin == "Spiculated":
            return "Grade 3 (High Grade – Poorly Differentiated)"
        elif ki67 > 15:
            return "Grade 2 (Intermediate)"
        else:
            return "Grade 1 (Low Grade – Well Differentiated)"

    def _get_prognosis(self, stage, subtype):
        survival = {1: "95%", 2: "80%", 3: "55%", 4: "25%"}
        s5 = survival.get(stage, "Unknown")
        tnbc_note = " Note: TNBC has more aggressive behavior; closer monitoring required." if "Triple Negative" in subtype else ""
        return f"5-year survival rate: {s5}.{tnbc_note}"


# ─────────────────────────────────────────────
#  LEVEL 4: TREATMENT RECOMMENDATION
# ─────────────────────────────────────────────
class TreatmentRecommendationModel:
    """ML-guided personalized treatment plan."""

    TREATMENT_PROTOCOLS = {
        "Luminal A": {
            1: ["Lumpectomy", "Hormone therapy (Tamoxifen/Letrozole)", "Radiation therapy"],
            2: ["Lumpectomy or Mastectomy", "Hormone therapy", "Radiation therapy"],
            3: ["Neoadjuvant chemotherapy", "Surgery", "Hormone therapy", "Radiation"],
            4: ["Palliative hormone therapy", "Targeted therapy", "Supportive care"]
        },
        "Triple Negative": {
            1: ["Surgery", "Adjuvant chemotherapy (AC-T)", "Radiation"],
            2: ["Neoadjuvant chemotherapy", "Surgery", "Radiation", "Capecitabine if residual disease"],
            3: ["Aggressive neoadjuvant chemo", "Surgery", "Radiation", "Immunotherapy (Pembrolizumab)"],
            4: ["Chemotherapy", "Immunotherapy", "PARP inhibitors if BRCA+", "Palliative care"]
        },
        "HER2-Enriched": {
            1: ["Surgery", "Trastuzumab (Herceptin)", "Chemotherapy"],
            2: ["Neoadjuvant chemo + Trastuzumab", "Surgery", "Pertuzumab"],
            3: ["Neoadjuvant chemo + dual HER2 blockade", "Surgery", "T-DM1 if residual"],
            4: ["Trastuzumab + Pertuzumab + chemo", "TKIs", "Palliative care"]
        },
        "Luminal B": {
            1: ["Surgery", "Chemotherapy", "Hormone therapy"],
            2: ["Surgery", "Chemo + hormone therapy", "Radiation"],
            3: ["Neoadjuvant chemo", "Surgery", "Hormone therapy", "Radiation"],
            4: ["Palliative chemo", "Hormone therapy", "CDK4/6 inhibitors"]
        }
    }

    def recommend(self, stage, subtype, patient_data):
        subtype_key = "Triple Negative" if "Triple Negative" in subtype else \
                      "HER2-Enriched" if "HER2-Enriched" in subtype else \
                      "Luminal B" if "Luminal B" in subtype else "Luminal A"

        protocol = self.TREATMENT_PROTOCOLS.get(subtype_key, self.TREATMENT_PROTOCOLS["Luminal A"])
        stage_protocol = protocol.get(stage, protocol[4])

        age = patient_data.get("age", 50)
        brca = patient_data.get("brca_mutation", 0)
        
        additional = []
        if brca and stage <= 2:
            additional.append("Consider bilateral mastectomy given BRCA mutation")
        if brca and "Triple Negative" in subtype:
            additional.append("PARP inhibitor (Olaparib) – BRCA mutation detected")
        if age < 40:
            additional.append("Fertility preservation counseling before chemotherapy")
        if patient_data.get("lymph_node_involvement"):
            additional.append("Extended lymph node dissection may be required")

        kenya_context = self._kenya_availability(stage_protocol)

        return {
            "primary_protocol": stage_protocol,
            "additional_considerations": additional,
            "kenya_availability": kenya_context,
            "monitoring": self._monitoring_plan(stage),
            "clinical_trial_eligible": stage >= 3
        }

    def _kenya_availability(self, protocol):
        available_in_kenya = {
            "Lumpectomy": "KNH, Aga Khan, MP Shah",
            "Mastectomy": "Most county referral hospitals",
            "Radiation therapy": "KNH, Aga Khan (Nairobi only)",
            "Tamoxifen": "Widely available – KEMSA",
            "Letrozole": "Available – major pharmacies",
            "Trastuzumab (Herceptin)": "Limited – Aga Khan, MP Shah (expensive)",
            "Chemotherapy (AC-T)": "KNH, Aga Khan, Moi Teaching Hospital",
            "Immunotherapy (Pembrolizumab)": "Very limited – private hospitals only",
            "Palliative care": "Nairobi Hospice, county hospitals"
        }
        result = {}
        for item in protocol:
            for key, val in available_in_kenya.items():
                if any(k in item for k in key.split()):
                    result[item] = val
                    break
            if item not in result:
                result[item] = "Consult oncologist"
        return result

    def _monitoring_plan(self, stage):
        if stage <= 2:
            return "Clinical exam every 3 months × 2 years, then every 6 months × 3 years, then annually"
        elif stage == 3:
            return "Clinical exam every 3 months + imaging every 6 months for 3 years"
        else:
            return "Monthly symptom monitoring, palliative team review every 4–6 weeks"


# ─────────────────────────────────────────────
#  LEVEL 5: RECURRENCE PREDICTION
# ─────────────────────────────────────────────
class RecurrencePredictionModel:
    """Predicts post-treatment recurrence risk."""

    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def train(self, df):
        cancer_df = df[df["has_cancer"] == 1].copy()
        if len(cancer_df) < 50:
            cancer_df = df.copy()

        # Simulate recurrence labels
        np.random.seed(789)
        recurrence_prob = (
            (cancer_df["cancer_stage"] / 4) * 0.4 +
            cancer_df["her2_positive"] * 0.1 +
            (cancer_df["tumor_type"] == "TNBC").astype(int) * 0.2 +
            cancer_df["lymph_node_involvement"] * 0.15 +
            np.random.uniform(0, 0.15, len(cancer_df))
        ).clip(0, 1)

        cancer_df["recurrence"] = (recurrence_prob > 0.5).astype(int)

        features = cancer_df[["age", "bmi", "cancer_stage", "ki67_percent",
                               "her2_positive", "er_positive", "lymph_node_involvement",
                               "ca153_u_ml", "brca_mutation"]].fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            features, cancer_df["recurrence"], test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        self.is_trained = True
        auc = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])
        return {"auc": round(auc, 3)}

    def predict_recurrence(self, patient_data):
        features = pd.DataFrame([{
            "age": patient_data.get("age", 50),
            "bmi": patient_data.get("bmi", 26),
            "cancer_stage": patient_data.get("cancer_stage", 2),
            "ki67_percent": patient_data.get("ki67_percent", 20),
            "her2_positive": patient_data.get("her2_positive", 0),
            "er_positive": patient_data.get("er_positive", 1),
            "lymph_node_involvement": patient_data.get("lymph_node_involvement", 0),
            "ca153_u_ml": patient_data.get("ca153_u_ml", 25),
            "brca_mutation": patient_data.get("brca_mutation", 0),
        }])

        prob = self.model.predict_proba(features)[0][1]
        if prob >= 0.65:
            risk = "High Recurrence Risk"
            action = "Intensive surveillance + extended adjuvant therapy discussion"
        elif prob >= 0.35:
            risk = "Moderate Recurrence Risk"
            action = "Standard follow-up + adherence to adjuvant therapy"
        else:
            risk = "Low Recurrence Risk"
            action = "Standard annual monitoring"

        return {
            "recurrence_probability": round(float(prob), 3),
            "recurrence_risk": risk,
            "recommended_action": action,
            "follow_up_interval": "3 months" if prob >= 0.5 else "6 months"
        }


# ─────────────────────────────────────────────
#  MASTER PIPELINE
# ─────────────────────────────────────────────
class BreastCancerPipeline:
    """End-to-end breast cancer detection and management pipeline."""

    def __init__(self):
        self.risk_model = RiskStratificationModel()
        self.imaging_model = ImagingAIModel()
        self.diagnosis_model = DiagnosisModel()
        self.treatment_model = TreatmentRecommendationModel()
        self.recurrence_model = RecurrencePredictionModel()
        self.df = None
        self.metrics = {}

    def train_all(self):
        """Train all models on synthetic Kenyan data."""
        print("Generating Kenyan synthetic dataset...")
        self.df = get_full_dataset()

        print("Training Risk Stratification Model...")
        self.metrics["risk"] = self.risk_model.train(self.df)

        print("Training Imaging AI Model...")
        self.metrics["imaging"] = self.imaging_model.train(self.df)

        print("Training Diagnosis Model...")
        self.metrics["diagnosis"] = self.diagnosis_model.train(self.df)

        print("Training Recurrence Prediction Model...")
        self.metrics["recurrence"] = self.recurrence_model.train(self.df)

        print("All models trained!")
        return self.metrics

    def run_full_pipeline(self, patient_data):
        """Run a patient through all 5 pipeline levels."""
        results = {}

        # Level 1
        results["risk"] = self.risk_model.predict_risk(patient_data)

        # Level 2
        results["imaging"] = self.imaging_model.analyze_imaging(patient_data)

        # Level 3
        results["diagnosis"] = self.diagnosis_model.diagnose(patient_data)

        # Level 4 (only if suspicious)
        if results["diagnosis"]["cancer_probability"] > 0.4:
            stage = results["diagnosis"]["predicted_stage"]
            subtype = results["diagnosis"]["molecular_subtype"]
            results["treatment"] = self.treatment_model.recommend(stage, subtype, patient_data)
        else:
            results["treatment"] = {"primary_protocol": ["Routine follow-up – no cancer detected"],
                                    "additional_considerations": [], "kenya_availability": {},
                                    "monitoring": "Annual mammogram screening", "clinical_trial_eligible": False}

        # Level 5
        results["recurrence"] = self.recurrence_model.predict_recurrence({
            **patient_data,
            "cancer_stage": results["diagnosis"]["predicted_stage"]
        })

        return results


if __name__ == "__main__":
    pipeline = BreastCancerPipeline()
    metrics = pipeline.train_all()
    print("\nModel Performance (AUC-ROC):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
