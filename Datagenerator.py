import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

KENYAN_COUNTIES = [
    "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret",
    "Thika", "Malindi", "Kitale", "Garissa", "Kakamega",
    "Nyeri", "Meru", "Embu", "Machakos", "Kilifi"
]

KENYAN_NAMES = [
    "Wanjiru", "Akinyi", "Chebet", "Zawadi", "Nafula",
    "Mwende", "Njeri", "Atieno", "Koech", "Mutua",
    "Kamau", "Ochieng", "Kimani", "Waweru", "Mugo",
    "Adhiambo", "Chepkoech", "Mumbi", "Nyambura", "Wangari"
]

ETHNIC_GROUPS = ["Kikuyu", "Luo", "Luhya", "Kalenjin", "Kamba", "Kisii", "Meru", "Mijikenda", "Somali", "Turkana"]

EDUCATION_LEVELS = ["None", "Primary", "Secondary", "Tertiary", "University"]

OCCUPATION = ["Farmer", "Business", "Teacher", "Nurse", "Housewife", "Student", "Civil Servant", "Casual Worker"]


def generate_patient_population(n=500):
    """Generate synthetic Kenyan breast cancer patient population."""
    
    ages = np.random.normal(45, 12, n).clip(20, 80).astype(int)
    
    # Parity (number of children) - higher in Kenya
    parity = np.random.poisson(3.5, n).clip(0, 12)
    
    # Age at first birth
    age_first_birth = np.where(parity > 0, np.random.normal(20, 3, n).clip(14, 35), 0).astype(int)
    
    # Breastfeeding duration (months) - generally longer in Kenya
    breastfeeding_months = np.where(parity > 0, np.random.exponential(18, n).clip(0, 60), 0).astype(int)
    
    # BMI - increasing obesity in urban Kenya
    bmi = np.random.normal(26.5, 5, n).clip(15, 50)
    
    # Family history
    family_history = np.random.binomial(1, 0.15, n)
    
    # BRCA mutation (lower testing rates in Kenya)
    brca_mutation = np.random.binomial(1, 0.05, n)
    
    # Hormonal contraceptive use
    contraceptive_use = np.random.binomial(1, 0.35, n)
    contraceptive_years = np.where(contraceptive_use == 1, np.random.exponential(4, n).clip(0, 20), 0)
    
    # HRT use (low in Kenya)
    hrt_use = np.where(ages > 45, np.random.binomial(1, 0.05, n), 0)
    
    # Alcohol use (lower but present)
    alcohol_use = np.random.binomial(1, 0.12, n)
    
    # Smoking (low in Kenyan women)
    smoking = np.random.binomial(1, 0.04, n)
    
    # Physical activity
    physical_activity = np.random.choice(["Low", "Moderate", "High"], n, p=[0.4, 0.45, 0.15])
    
    # Breast density (dense tissue more common in younger/African women)
    breast_density = np.random.choice(["A", "B", "C", "D"], n, p=[0.1, 0.25, 0.4, 0.25])
    
    # Menarche age
    menarche_age = np.random.normal(13.5, 1.5, n).clip(9, 18).astype(int)
    
    # Menopause
    menopause = np.where(ages > 50, np.random.binomial(1, 0.7, n),
                         np.where(ages > 45, np.random.binomial(1, 0.2, n), 0))
    
    # Geography
    county = np.random.choice(KENYAN_COUNTIES, n)
    urban_rural = np.where(np.isin(county, ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"]),
                           np.random.choice(["Urban", "Peri-urban"], n, p=[0.7, 0.3]),
                           np.random.choice(["Rural", "Peri-urban"], n, p=[0.7, 0.3]))
    
    education = np.random.choice(EDUCATION_LEVELS, n, p=[0.05, 0.25, 0.35, 0.25, 0.10])
    occupation = np.random.choice(OCCUPATION, n)
    ethnicity = np.random.choice(ETHNIC_GROUPS, n)
    
    # Previous biopsy
    prev_biopsy = np.random.binomial(1, 0.08, n)
    
    # Distance to hospital (km)
    distance_to_hospital = np.where(urban_rural == "Urban",
                                    np.random.exponential(5, n).clip(1, 30),
                                    np.random.exponential(35, n).clip(5, 150))
    
    # Income proxy (monthly KES)
    income_kes = np.where(education == "University",
                          np.random.normal(80000, 30000, n).clip(20000, 300000),
                          np.where(education == "Tertiary",
                                   np.random.normal(40000, 15000, n).clip(10000, 100000),
                                   np.random.normal(15000, 8000, n).clip(0, 50000)))
    
    # Compute risk score (for label generation)
    risk_score = (
        (ages - 20) / 60 * 0.25 +
        family_history * 0.20 +
        brca_mutation * 0.25 +
        (bmi - 18.5) / 31.5 * 0.08 +
        (breast_density == "D").astype(int) * 0.10 +
        (breast_density == "C").astype(int) * 0.05 +
        hrt_use * 0.05 +
        alcohol_use * 0.03 +
        np.random.uniform(0, 0.1, n)  # noise
    ).clip(0, 1)
    
    # Cancer status with stage
    has_cancer = (risk_score > np.random.uniform(0.3, 0.7, n)).astype(int)
    
    stage_probs = np.where(has_cancer == 1,
                           np.random.choice([1, 2, 3, 4], n, p=[0.15, 0.25, 0.35, 0.25]),
                           0)
    
    # Tumor characteristics (only meaningful if cancer)
    tumor_size_mm = np.where(has_cancer == 1,
                              np.where(stage_probs >= 3,
                                       np.random.normal(45, 15, n).clip(20, 100),
                                       np.random.normal(18, 8, n).clip(5, 40)),
                              0)
    
    tumor_type = np.where(has_cancer == 1,
                           np.random.choice(["IDC", "ILC", "TNBC", "HER2+", "Luminal A", "Luminal B"], n,
                                            p=[0.40, 0.10, 0.25, 0.10, 0.10, 0.05]),
                           "None")
    
    lymph_node_involvement = np.where(stage_probs >= 2,
                                       np.random.binomial(1, 0.6, n),
                                       0)
    
    # Names
    first_name = np.random.choice(KENYAN_NAMES, n)
    patient_id = [f"KBC{str(i+1001).zfill(4)}" for i in range(n)]
    
    df = pd.DataFrame({
        "patient_id": patient_id,
        "name": first_name,
        "age": ages,
        "county": county,
        "urban_rural": urban_rural,
        "ethnicity": ethnicity,
        "education": education,
        "occupation": occupation,
        "income_kes": income_kes.astype(int),
        "distance_to_hospital_km": distance_to_hospital.round(1),
        "parity": parity,
        "age_first_birth": age_first_birth,
        "breastfeeding_months": breastfeeding_months,
        "menarche_age": menarche_age,
        "menopause": menopause,
        "bmi": bmi.round(1),
        "family_history": family_history,
        "brca_mutation": brca_mutation,
        "contraceptive_use": contraceptive_use,
        "contraceptive_years": contraceptive_years.round(1),
        "hrt_use": hrt_use,
        "alcohol_use": alcohol_use,
        "smoking": smoking,
        "physical_activity": physical_activity,
        "breast_density": breast_density,
        "prev_biopsy": prev_biopsy,
        "risk_score": risk_score.round(3),
        "has_cancer": has_cancer,
        "cancer_stage": stage_probs,
        "tumor_size_mm": tumor_size_mm.round(1),
        "tumor_type": tumor_type,
        "lymph_node_involvement": lymph_node_involvement,
    })
    
    return df


def generate_imaging_data(n=500):
    """Simulate imaging features from mammogram/ultrasound analysis."""
    np.random.seed(123)
    
    # BI-RADS categories
    birads = np.random.choice([1, 2, 3, 4, 5], n, p=[0.25, 0.30, 0.20, 0.15, 0.10])
    
    # Imaging features
    mass_present = np.where(birads >= 4, np.random.binomial(1, 0.8, n),
                            np.random.binomial(1, 0.15, n))
    
    mass_shape = np.where(mass_present == 1,
                           np.random.choice(["Round", "Oval", "Irregular", "Lobular"], n,
                                            p=[0.15, 0.20, 0.45, 0.20]),
                           "None")
    
    mass_margin = np.where(mass_present == 1,
                            np.random.choice(["Circumscribed", "Microlobulated", "Indistinct", "Spiculated"], n,
                                             p=[0.20, 0.20, 0.30, 0.30]),
                            "None")
    
    calcifications = np.where(birads >= 3,
                               np.random.binomial(1, 0.45, n),
                               np.random.binomial(1, 0.05, n))
    
    skin_thickening = np.where(birads >= 4, np.random.binomial(1, 0.3, n), 0)
    nipple_retraction = np.where(birads >= 4, np.random.binomial(1, 0.2, n), 0)
    
    # AI confidence score
    ai_malignancy_prob = np.where(birads >= 4,
                                   np.random.beta(7, 3, n),
                                   np.where(birads == 3,
                                            np.random.beta(3, 7, n),
                                            np.random.beta(1, 9, n)))
    
    imaging_df = pd.DataFrame({
        "birads_category": birads,
        "mass_present": mass_present,
        "mass_shape": mass_shape,
        "mass_margin": mass_margin,
        "calcifications": calcifications,
        "skin_thickening": skin_thickening,
        "nipple_retraction": nipple_retraction,
        "ai_malignancy_probability": ai_malignancy_prob.round(3),
    })
    
    return imaging_df


def generate_biomarker_data(n=500):
    """Generate lab / biomarker data."""
    np.random.seed(456)
    
    # Hormone receptor status
    er_positive = np.random.binomial(1, 0.65, n)
    pr_positive = np.where(er_positive, np.random.binomial(1, 0.75, n), np.random.binomial(1, 0.15, n))
    her2_positive = np.random.binomial(1, 0.20, n)
    
    # Ki-67 proliferation index
    ki67 = np.random.beta(2, 5, n) * 100
    
    # CA 15-3 tumor marker (U/mL) - elevated in cancer
    ca153 = np.random.exponential(25, n).clip(5, 300)
    
    # CEA
    cea = np.random.exponential(3, n).clip(0.5, 50)
    
    # CBC
    hemoglobin = np.random.normal(12.5, 1.5, n).clip(7, 17)
    wbc = np.random.normal(7.5, 2, n).clip(3, 15)
    platelets = np.random.normal(280, 60, n).clip(100, 500)
    
    biomarker_df = pd.DataFrame({
        "er_positive": er_positive,
        "pr_positive": pr_positive,
        "her2_positive": her2_positive,
        "ki67_percent": ki67.round(1),
        "ca153_u_ml": ca153.round(1),
        "cea_ng_ml": cea.round(2),
        "hemoglobin_g_dl": hemoglobin.round(1),
        "wbc_10e3": wbc.round(1),
        "platelets_10e3": platelets.round(0).astype(int),
    })
    
    return biomarker_df


def get_full_dataset():
    """Combine all data into one master dataset."""
    pop = generate_patient_population(500)
    img = generate_imaging_data(500)
    bio = generate_biomarker_data(500)
    
    df = pd.concat([pop, img, bio], axis=1)
    return df


if __name__ == "__main__":
    df = get_full_dataset()
    print(df.head())
    print(df.shape)
    print(df["has_cancer"].value_counts())
