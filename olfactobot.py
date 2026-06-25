#!/usr/bin/env python3
"""
OlfactoBot AI

A full runnable simulation of a mobile AI-powered artificial nose that combines:
- broad chemical sensing
- sensor drift correction
- analytical chemistry-like volatile features
- biological meaning labels
- multimodal machine learning
- uncertainty-aware decision logic
- autonomous chemical plume mapping

Run:
    python olfactobot.py --mode full
"""

import argparse
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
MODELS = BASE / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


CHEMICAL_CLASSES = {
    "safe_background": {
        "ketones": 0.05,
        "aldehydes": 0.05,
        "alcohols": 0.08,
        "amines": 0.02,
        "sulfur": 0.01,
        "terpenes": 0.04,
        "acids": 0.02,
        "hazard": 0.01,
        "bio_meaning": "safe_background",
    },
    "food_spoilage": {
        "ketones": 0.35,
        "aldehydes": 0.30,
        "alcohols": 0.25,
        "amines": 0.55,
        "sulfur": 0.45,
        "terpenes": 0.08,
        "acids": 0.60,
        "hazard": 0.15,
        "bio_meaning": "food_spoilage",
    },
    "infection_like_voc": {
        "ketones": 0.55,
        "aldehydes": 0.20,
        "alcohols": 0.18,
        "amines": 0.35,
        "sulfur": 0.28,
        "terpenes": 0.05,
        "acids": 0.45,
        "hazard": 0.25,
        "bio_meaning": "infection_like_voc",
    },
    "environmental_hazard": {
        "ketones": 0.40,
        "aldehydes": 0.45,
        "alcohols": 0.35,
        "amines": 0.15,
        "sulfur": 0.65,
        "terpenes": 0.10,
        "acids": 0.20,
        "hazard": 0.85,
        "bio_meaning": "environmental_hazard",
    },
}


SENSOR_WEIGHTS = {
    "MOS_1": {"ketones": 0.90, "aldehydes": 0.20, "alcohols": 0.25, "amines": 0.15, "sulfur": 0.10, "terpenes": 0.20, "acids": 0.10, "hazard": 0.20},
    "MOS_2": {"ketones": 0.20, "aldehydes": 0.80, "alcohols": 0.15, "amines": 0.20, "sulfur": 0.30, "terpenes": 0.10, "acids": 0.20, "hazard": 0.30},
    "MOS_3": {"ketones": 0.25, "aldehydes": 0.20, "alcohols": 0.85, "amines": 0.15, "sulfur": 0.20, "terpenes": 0.25, "acids": 0.15, "hazard": 0.25},
    "EC_1": {"ketones": 0.10, "aldehydes": 0.20, "alcohols": 0.10, "amines": 0.70, "sulfur": 0.50, "terpenes": 0.05, "acids": 0.20, "hazard": 0.55},
    "EC_2": {"ketones": 0.10, "aldehydes": 0.25, "alcohols": 0.10, "amines": 0.20, "sulfur": 0.85, "terpenes": 0.05, "acids": 0.20, "hazard": 0.75},
    "QCM_1": {"ketones": 0.15, "aldehydes": 0.20, "alcohols": 0.25, "amines": 0.35, "sulfur": 0.20, "terpenes": 0.10, "acids": 0.70, "hazard": 0.20},
    "PID_1": {"ketones": 0.45, "aldehydes": 0.45, "alcohols": 0.40, "amines": 0.25, "sulfur": 0.30, "terpenes": 0.25, "acids": 0.25, "hazard": 0.65},
    "IR_1": {"ketones": 0.20, "aldehydes": 0.35, "alcohols": 0.60, "amines": 0.15, "sulfur": 0.20, "terpenes": 0.15, "acids": 0.35, "hazard": 0.25},
}


FEATURES = [
    "humidity",
    "temperature",
    "airflow",
    "distance_to_source",
    "ketones",
    "aldehydes",
    "alcohols",
    "amines",
    "sulfur",
    "terpenes",
    "acids",
    "hazard",
    "bio_attraction",
    "bio_avoidance",
    "bio_stress",
    "bio_metabolic_shift",
] + [f"sensor_{name}" for name in SENSOR_WEIGHTS.keys()]


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    battery: float = 100.0
    mission_step: int = 0


class OlfactoBotSimulator:
    """Simulates volatile mixtures, sensor-array outputs, and biological meaning."""

    def __init__(self, seed: int = RANDOM_SEED):
        self.rng = np.random.default_rng(seed)

    def simulate_environment(self, label: str, n: int, source_strength: float = 1.0) -> pd.DataFrame:
        rows = []
        base = CHEMICAL_CLASSES[label]

        for i in range(n):
            humidity = float(np.clip(self.rng.normal(55, 12), 20, 90))
            temperature = float(np.clip(self.rng.normal(22, 3), 10, 35))
            airflow = float(np.clip(self.rng.normal(0.45, 0.18), 0.02, 1.3))
            distance = float(np.clip(self.rng.gamma(2.0, 1.5), 0.1, 12.0))

            # Plume dilution, distance and airflow both reduce apparent concentration.
            plume_factor = source_strength * math.exp(-distance / 6.0) * (1.0 / (1.0 + airflow))
            plume_factor = float(np.clip(plume_factor, 0.02, 1.5))

            chemistry = {}
            for k in ["ketones", "aldehydes", "alcohols", "amines", "sulfur", "terpenes", "acids", "hazard"]:
                chemistry[k] = float(np.clip(base[k] * plume_factor + self.rng.normal(0, 0.035), 0, 1.2))

            # Biological interpretation layer, inspired by attraction, avoidance, stress, and metabolism.
            bio_attraction = float(np.clip(0.75 - 0.75 * base["hazard"] - 0.30 * chemistry["sulfur"] + self.rng.normal(0, 0.08), 0, 1))
            bio_avoidance = float(np.clip(0.20 + 0.90 * base["hazard"] + 0.35 * chemistry["amines"] + 0.30 * chemistry["sulfur"] + self.rng.normal(0, 0.08), 0, 1))
            bio_stress = float(np.clip(0.15 + 0.70 * base["hazard"] + 0.30 * chemistry["acids"] + 0.25 * chemistry["sulfur"] + self.rng.normal(0, 0.08), 0, 1))
            bio_metabolic_shift = float(np.clip(0.20 + 0.45 * chemistry["ketones"] + 0.20 * chemistry["alcohols"] + self.rng.normal(0, 0.08), 0, 1))

            sensors = {}
            for sensor_name, weights in SENSOR_WEIGHTS.items():
                signal = 0.0
                for chem_name, w in weights.items():
                    signal += w * chemistry[chem_name]

                # Environmental artefacts and sensor noise.
                signal += 0.004 * (humidity - 55)
                signal += 0.010 * (temperature - 22)
                signal += self.rng.normal(0, 0.04)

                sensors[f"sensor_{sensor_name}"] = float(np.clip(signal, 0, 2.0))

            row = {
                "sample_id": f"{label}_{i:05d}",
                "true_label": label,
                "humidity": humidity,
                "temperature": temperature,
                "airflow": airflow,
                "distance_to_source": distance,
                **chemistry,
                "bio_attraction": bio_attraction,
                "bio_avoidance": bio_avoidance,
                "bio_stress": bio_stress,
                "bio_metabolic_shift": bio_metabolic_shift,
                **sensors,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def make_training_data(self, n_per_class: int = 550) -> pd.DataFrame:
        frames = []
        for label in CHEMICAL_CLASSES.keys():
            frames.append(self.simulate_environment(label, n_per_class))
        df = pd.concat(frames, ignore_index=True)
        return df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


class DriftCorrector:
    """Simple reproducible drift and artefact correction."""

    def __init__(self):
        self.reference_means: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame):
        background = df[df["true_label"] == "safe_background"]
        for c in [x for x in df.columns if x.startswith("sensor_")]:
            self.reference_means[c] = float(background[c].mean())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c, ref in self.reference_means.items():
            if c in out.columns:
                # Correct humidity and temperature artefacts using simple known-control adjustment.
                out[c] = out[c] - 0.004 * (out["humidity"] - 55) - 0.010 * (out["temperature"] - 22)
                out[c] = out[c] - 0.15 * ref
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class BioMeaningOlfactionAI:
    """Trains and serves the artificial olfaction model."""

    def __init__(self):
        self.drift = DriftCorrector()
        self.classifier = Pipeline([
            ("scale", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=450,
                random_state=RANDOM_SEED,
                class_weight="balanced",
                max_depth=None,
            )),
        ])
        self.uncertainty_detector = IsolationForest(
            n_estimators=200,
            contamination=0.04,
            random_state=RANDOM_SEED,
        )

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        corrected = self.drift.fit_transform(df)
        X = corrected[FEATURES]
        y = corrected["true_label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
        )

        self.classifier.fit(X_train, y_train)
        self.uncertainty_detector.fit(X_train)

        pred = self.classifier.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
        }

        report = classification_report(y_test, pred, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(OUT / "classification_report.csv")

        cm = confusion_matrix(y_test, pred, labels=sorted(y.unique()))
        self.plot_confusion_matrix(cm, sorted(y.unique()))

        perm = permutation_importance(
            self.classifier, X_test, y_test,
            n_repeats=8, random_state=RANDOM_SEED,
            scoring="balanced_accuracy"
        )
        importance = pd.DataFrame({
            "feature": FEATURES,
            "importance": perm.importances_mean,
            "sd": perm.importances_std,
        }).sort_values("importance", ascending=False)
        importance.to_csv(OUT / "feature_importance.csv", index=False)
        self.plot_feature_importance(importance.head(18))

        joblib.dump(self, MODELS / "olfactobot_ai_model.joblib")
        return metrics

    def predict(self, sample: pd.DataFrame) -> Dict:
        corrected = self.drift.transform(sample)
        X = corrected[FEATURES]
        proba = self.classifier.predict_proba(X)[0]
        classes = self.classifier.classes_
        best_idx = int(np.argmax(proba))
        label = str(classes[best_idx])
        confidence = float(proba[best_idx])

        anomaly_score = float(self.uncertainty_detector.decision_function(X)[0])
        outlier = bool(self.uncertainty_detector.predict(X)[0] == -1)
        uncertainty = float(np.clip(1.0 - confidence + (0.25 if outlier else 0.0), 0, 1))

        action = self.choose_action(label, confidence, uncertainty, sample.iloc[0].to_dict())

        return {
            "predicted_label": label,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "anomaly_score": anomaly_score,
            "recommended_action": action,
            "class_probabilities": {str(c): float(p) for c, p in zip(classes, proba)},
        }

    @staticmethod
    def choose_action(label: str, confidence: float, uncertainty: float, row: Dict) -> str:
        if uncertainty > 0.45:
            return "collect_confirmatory_sample"
        if label == "environmental_hazard" and confidence > 0.70:
            return "retreat_and_alert_operator"
        if label == "infection_like_voc" and confidence > 0.70:
            return "collect_sample_and_recommend_confirmatory_test"
        if label == "food_spoilage" and confidence > 0.70:
            return "mark_location_and_request_food_quality_check"
        if label == "safe_background":
            return "continue_patrol"
        return "repeat_sampling"

    @staticmethod
    def plot_confusion_matrix(cm, labels: List[str]):
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm)
        ax.set_title("OlfactoBot AI confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(OUT / "confusion_matrix.png", dpi=220)
        plt.close(fig)

    @staticmethod
    def plot_feature_importance(importance: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(9, 7))
        plot_df = importance.iloc[::-1]
        ax.barh(plot_df["feature"], plot_df["importance"], xerr=plot_df["sd"])
        ax.set_xlabel("Permutation importance")
        ax.set_title("Most important variables for biological chemical meaning")
        fig.tight_layout()
        fig.savefig(OUT / "feature_importance.png", dpi=220)
        plt.close(fig)


class ChemicalPlumeWorld:
    """2D chemical plume world for robot mission simulation."""

    def __init__(self, width: int = 40, height: int = 30, source: Tuple[int, int] = (30, 18), label: str = "environmental_hazard"):
        self.width = width
        self.height = height
        self.source = source
        self.label = label
        self.sim = OlfactoBotSimulator(seed=123)

    def plume_strength_at(self, x: float, y: float) -> float:
        sx, sy = self.source
        d = math.sqrt((x - sx) ** 2 + (y - sy) ** 2)
        wind_bias = max(0.25, 1.0 + 0.025 * (sx - x))
        return float(np.clip(1.4 * math.exp(-d / 8.0) * wind_bias, 0.02, 1.5))

    def sample_at(self, x: float, y: float) -> pd.DataFrame:
        strength = self.plume_strength_at(x, y)
        df = self.sim.simulate_environment(self.label, 1, source_strength=strength)
        df["robot_x"] = x
        df["robot_y"] = y
        df["plume_strength"] = strength
        return df


class OlfactoBotRobot:
    """Mobile robot controller for plume mapping."""

    def __init__(self, ai: BioMeaningOlfactionAI, world: ChemicalPlumeWorld):
        self.ai = ai
        self.world = world
        self.state = RobotState(x=2.0, y=2.0)
        self.history = []

    def move(self, dx: float, dy: float):
        self.state.x = float(np.clip(self.state.x + dx, 0, self.world.width - 1))
        self.state.y = float(np.clip(self.state.y + dy, 0, self.world.height - 1))
        self.state.battery = max(0, self.state.battery - 0.4)
        self.state.mission_step += 1

    def step(self):
        sample = self.world.sample_at(self.state.x, self.state.y)
        prediction = self.ai.predict(sample)

        row = sample.iloc[0].to_dict()
        row.update(prediction)
        row["x"] = self.state.x
        row["y"] = self.state.y
        row["battery"] = self.state.battery
        row["step"] = self.state.mission_step
        self.history.append(row)

        # Move toward stronger signal unless hazard is high-confidence.
        if prediction["recommended_action"] == "retreat_and_alert_operator":
            self.move(-1.5, -1.0)
        elif prediction["recommended_action"] == "collect_confirmatory_sample":
            self.move(0.5, 0.2)
        else:
            # Gradient-like movement toward source, with exploratory motion.
            sx, sy = self.world.source
            dx = np.sign(sx - self.state.x) * 1.1 + np.random.normal(0, 0.25)
            dy = np.sign(sy - self.state.y) * 0.8 + np.random.normal(0, 0.25)
            self.move(dx, dy)

    def run_mission(self, steps: int = 45) -> pd.DataFrame:
        for _ in range(steps):
            self.step()
        mission = pd.DataFrame(self.history)
        mission.to_csv(OUT / "robot_mission_log.csv", index=False)
        self.plot_mission(mission)
        return mission

    def plot_mission(self, mission: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(9, 7))
        sc = ax.scatter(mission["x"], mission["y"], c=mission["plume_strength"], s=80)
        ax.plot(mission["x"], mission["y"], linewidth=1)
        ax.scatter([self.world.source[0]], [self.world.source[1]], marker="*", s=300, label="chemical source")
        ax.set_xlim(0, self.world.width)
        ax.set_ylim(0, self.world.height)
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        ax.set_title("OlfactoBot autonomous chemical plume mapping")
        ax.legend()
        fig.colorbar(sc, ax=ax, label="simulated plume strength")
        fig.tight_layout()
        fig.savefig(OUT / "robot_plume_map.png", dpi=220)
        plt.close(fig)


def plot_dataset_overview(df: pd.DataFrame):
    # Sensor PCA
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    X = StandardScaler().fit_transform(df[sensor_cols])
    pcs = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X)
    pca_df = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], "label": df["true_label"]})

    fig, ax = plt.subplots(figsize=(8, 7))
    for label, sub in pca_df.groupby("label"):
        ax.scatter(sub["PC1"], sub["PC2"], label=label, alpha=0.6, s=20)
    ax.set_title("Sensor-array PCA, simulated e-nose response space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sensor_pca.png", dpi=220)
    plt.close(fig)

    # Biological meaning by class
    bio = df.groupby("true_label")[["bio_attraction", "bio_avoidance", "bio_stress", "bio_metabolic_shift"]].mean()
    bio.to_csv(OUT / "biological_meaning_summary.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    bio.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean score")
    ax.set_title("Biological meaning layer by chemical class")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT / "biological_meaning_by_class.png", dpi=220)
    plt.close(fig)


def run_train() -> BioMeaningOlfactionAI:
    sim = OlfactoBotSimulator()
    df = sim.make_training_data(n_per_class=650)
    df.to_csv(OUT / "olfactobot_training_dataset.csv", index=False)
    plot_dataset_overview(df)

    ai = BioMeaningOlfactionAI()
    metrics = ai.train(df)
    (OUT / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\nTraining complete.")
    print(json.dumps(metrics, indent=2))
    print(f"\nOutputs saved to: {OUT}")
    print(f"Model saved to: {MODELS / 'olfactobot_ai_model.joblib'}")
    return ai


def run_mission(ai: BioMeaningOlfactionAI = None):
    if ai is None:
        model_path = MODELS / "olfactobot_ai_model.joblib"
        if not model_path.exists():
            print("No trained model found. Training first.")
            ai = run_train()
        else:
            ai = joblib.load(model_path)

    world = ChemicalPlumeWorld(label="environmental_hazard", source=(31, 21))
    robot = OlfactoBotRobot(ai, world)
    mission = robot.run_mission(steps=50)

    print("\nMission complete.")
    print("Last 8 robot decisions:")
    cols = ["step", "x", "y", "predicted_label", "confidence", "uncertainty", "recommended_action", "plume_strength"]
    print(mission[cols].tail(8).to_string(index=False))
    print(f"\nMission log: {OUT / 'robot_mission_log.csv'}")
    print(f"Plume map: {OUT / 'robot_plume_map.png'}")


def write_terminal_runner():
    runner = BASE / "run_olfactobot.sh"
    runner.write_text("""#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python olfactobot.py --mode full
""")
    runner.chmod(0o755)


def main():
    parser = argparse.ArgumentParser(description="OlfactoBot AI, artificial olfaction robot simulation.")
    parser.add_argument("--mode", choices=["train", "mission", "full"], default="full")
    args = parser.parse_args()

    write_terminal_runner()

    if args.mode == "train":
        run_train()
    elif args.mode == "mission":
        run_mission()
    elif args.mode == "full":
        ai = run_train()
        run_mission(ai)

        summary = {
            "system": "OlfactoBot AI",
            "concept": "mobile AI-powered artificial nose",
            "capabilities": [
                "broad chemical sensing",
                "sensor drift correction",
                "biological meaning modelling",
                "multimodal machine learning",
                "uncertainty-aware decisions",
                "autonomous plume mapping",
            ],
            "outputs_folder": str(OUT),
        }
        (OUT / "olfactobot_summary.json").write_text(json.dumps(summary, indent=2))
        print("\nFull run complete.")


if __name__ == "__main__":
    main()
