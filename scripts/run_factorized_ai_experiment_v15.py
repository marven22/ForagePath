from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim" / "factorized_model_v15"
OUT = ROOT / "results" / "factorized_ai_v15"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_SEED = 20260815
ENSEMBLE_SIZE = 2
BATCH_SIZE = 8192
MAX_EPOCHS = 30
PATIENCE = 5

BEE_CATS = [
    "bee_family", "bee_tribe", "bee_genus", "nesting_behavior", "voltinism",
    "sociality", "foraging_behavior", "tongue_length",
]
PLANT_CATS = ["plant_family", "plant_genus"]
PLANT_TRAITS = [
    "flores_flowering_start", "flores_flowering_end", "flores_flowers_m2",
    "flores_nectar_ml", "flores_sugar_concentration_mol",
    "flores_sugar_concentration_perc", "flores_nectar_sugar_content_mg",
    "flores_pollen_mg", "flores_corolla_mm", "flores_protein",
]
BEE_FLAGS = ["beegap_exact_match", "bee_core_traits_complete", "habitat_profile_available"]
PLANT_FLAGS = [
    "flores_exact_match", "filipiak_exact_match", "has_any_open_floral_resource_trait",
    "has_corolla_depth", "has_pollen_quantity", "has_nectar_or_sugar",
]


def bucket(value: str, modulus: int) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % modulus


def hash_index(value: str, size: int) -> int:
    return bucket(value, size)


def embedding_size(classes: int) -> int:
    return min(16, max(2, int(round(math.sqrt(classes)))))


class Encoder:
    def __init__(self, train: pd.DataFrame, use_traits: bool):
        self.use_traits = use_traits
        self.vocab = {}
        for column in BEE_CATS + PLANT_CATS + ["sampling_grain"]:
            values = sorted(train[column].fillna("unknown").astype(str).unique())
            self.vocab[column] = {value: index + 1 for index, value in enumerate(values)}
        self.plant_means, self.plant_scales = {}, {}
        for column in PLANT_TRAITS:
            transformed = self._trait_transform(column, train[column])
            mean = float(transformed.mean(skipna=True)) if transformed.notna().any() else 0.0
            scale = float(transformed.std(skipna=True)) if transformed.notna().sum() > 1 else 1.0
            self.plant_means[column] = mean
            self.plant_scales[column] = scale if scale > 1e-8 else 1.0
        self._fit_presence(train)
        self._fit_graph(train)
        self._fit_detection(train)

    @staticmethod
    def _trait_transform(column: str, values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        if column in {"flores_flowering_start", "flores_flowering_end"}:
            return numeric
        return np.log1p(numeric.clip(lower=0))

    def _fit_presence(self, train: pd.DataFrame) -> None:
        events = train[["event_id", "week"]].drop_duplicates()
        week_events = events.groupby("week")["event_id"].nunique()
        total_events = max(events["event_id"].nunique(), 1)
        bee_events = train[["event_id", "week", "bee", "bee_family"]].drop_duplicates()
        plant_events = train[["event_id", "week", "plant", "plant_family"]].drop_duplicates()
        self.bee_presence = bee_events.groupby("bee")["event_id"].nunique() / total_events
        self.plant_presence = plant_events.groupby("plant")["event_id"].nunique() / total_events
        self.bee_week = (
            bee_events.groupby(["bee", "week"])["event_id"].nunique()
            / bee_events.groupby(["bee", "week"])["week"].first().map(week_events)
        )
        self.plant_week = (
            plant_events.groupby(["plant", "week"])["event_id"].nunique()
            / plant_events.groupby(["plant", "week"])["week"].first().map(week_events)
        )
        self.bee_family_week = (
            bee_events.groupby(["bee_family", "week"])["event_id"].nunique()
            / bee_events.groupby(["bee_family", "week"])["week"].first().map(week_events)
        )
        self.plant_family_week = (
            plant_events.groupby(["plant_family", "week"])["event_id"].nunique()
            / plant_events.groupby(["plant_family", "week"])["week"].first().map(week_events)
        )

    def _fit_graph(self, train: pd.DataFrame) -> None:
        positive = train[train["interaction_observed"].eq(1)][
            ["bee", "plant", "bee_family", "plant_family"]
        ].drop_duplicates(["bee", "plant"])
        self.bee_degree = np.log1p(positive.groupby("bee")["plant"].nunique())
        self.plant_degree = np.log1p(positive.groupby("plant")["bee"].nunique())
        bee_neighbours: dict[str, np.ndarray] = {}
        plant_neighbours: dict[str, np.ndarray] = {}
        for bee, group in positive.groupby("bee"):
            vector = np.zeros(32, dtype=np.float32)
            for family in group["plant_family"]:
                vector[hash_index(str(family), len(vector))] += 1
            bee_neighbours[bee] = vector / max(vector.sum(), 1)
        for plant, group in positive.groupby("plant"):
            vector = np.zeros(8, dtype=np.float32)
            for family in group["bee_family"]:
                vector[hash_index(str(family), len(vector))] += 1
            plant_neighbours[plant] = vector / max(vector.sum(), 1)
        self.bee_neighbours = bee_neighbours
        self.plant_neighbours = plant_neighbours

    def _fit_detection(self, train: pd.DataFrame) -> None:
        study_events = train[["study", "event_id"]].drop_duplicates().groupby("study")["event_id"].nunique()
        study_sites = train[["study", "site"]].drop_duplicates().groupby("study")["site"].nunique()
        self.study_events = np.log1p(study_events)
        self.study_sites = np.log1p(study_sites)
        event_size = train[["event_id", "log_event_candidate_pairs"]].drop_duplicates()["log_event_candidate_pairs"]
        self.det_mean = np.asarray([
            event_size.mean(), self.study_events.mean(), self.study_sites.mean()
        ], dtype=np.float32)
        self.det_scale = np.asarray([
            event_size.std(), self.study_events.std(), self.study_sites.std()
        ], dtype=np.float32)
        self.det_scale[self.det_scale < 1e-6] = 1.0

    def categorical(self, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
        output = np.zeros((len(frame), len(columns)), dtype=np.int64)
        biological_bee_traits = {
            "nesting_behavior", "voltinism", "sociality",
            "foraging_behavior", "tongue_length",
        }
        for index, column in enumerate(columns):
            if not self.use_traits and column in biological_bee_traits:
                # Keep this ablation free of biological trait information while
                # retaining taxonomy and the same training-graph evidence.
                output[:, index] = 0
                continue
            output[:, index] = frame[column].fillna("unknown").astype(str).map(self.vocab[column]).fillna(0).astype(int)
        return output

    @staticmethod
    def lookup_multi(frame: pd.DataFrame, columns: list[str], values: pd.Series, default: float = 0.0) -> np.ndarray:
        index = pd.MultiIndex.from_frame(frame[columns])
        return values.reindex(index).fillna(default).to_numpy(float)

    def transform(self, frame: pd.DataFrame) -> dict[str, torch.Tensor]:
        bee_cat = self.categorical(frame, BEE_CATS)
        plant_cat = self.categorical(frame, PLANT_CATS)
        grain = self.categorical(frame, ["sampling_grain"]).reshape(-1)

        bee_numeric = []
        if self.use_traits:
            bee_numeric.extend([frame[column].fillna(0).to_numpy(float) for column in BEE_FLAGS])
        else:
            bee_numeric.extend([np.zeros(len(frame)) for _ in BEE_FLAGS])
        bee_numeric.append(frame["bee"].map(self.bee_degree).fillna(0).to_numpy(float))
        bee_numeric.extend([
            np.asarray([self.bee_neighbours.get(bee, np.zeros(32))[position] for bee in frame["bee"]])
            for position in range(32)
        ])

        plant_numeric = []
        for column in PLANT_TRAITS:
            values = self._trait_transform(column, frame[column])
            standardized = ((values - self.plant_means[column]) / self.plant_scales[column]).fillna(0).to_numpy(float)
            plant_numeric.append(standardized if self.use_traits else np.zeros(len(frame)))
            plant_numeric.append(frame[f"{column}_missing"].to_numpy(float) if self.use_traits else np.zeros(len(frame)))
        plant_numeric.extend([
            frame[column].fillna(0).to_numpy(float) if self.use_traits else np.zeros(len(frame))
            for column in PLANT_FLAGS
        ])
        plant_numeric.append(frame["plant"].map(self.plant_degree).fillna(0).to_numpy(float))
        plant_numeric.extend([
            np.asarray([self.plant_neighbours.get(plant, np.zeros(8))[position] for plant in frame["plant"]])
            for position in range(8)
        ])

        bee_week = self.lookup_multi(frame, ["bee", "week"], self.bee_week)
        plant_week = self.lookup_multi(frame, ["plant", "week"], self.plant_week)
        bee_family_week = self.lookup_multi(frame, ["bee_family", "week"], self.bee_family_week)
        plant_family_week = self.lookup_multi(frame, ["plant_family", "week"], self.plant_family_week)
        bee_week = np.where(bee_week > 0, bee_week, bee_family_week)
        plant_week = np.where(plant_week > 0, plant_week, plant_family_week)
        encounter = np.column_stack([
            frame["week_sin"], frame["week_cos"],
            frame["bee"].map(self.bee_presence).fillna(0),
            frame["plant"].map(self.plant_presence).fillna(0),
            bee_week, plant_week,
        ]).astype(np.float32)
        detection = np.column_stack([
            frame["log_event_candidate_pairs"].to_numpy(float),
            frame["study"].map(self.study_events).fillna(0).to_numpy(float),
            frame["study"].map(self.study_sites).fillna(0).to_numpy(float),
        ]).astype(np.float32)
        detection = (detection - self.det_mean) / self.det_scale

        return {
            "bee_cat": torch.as_tensor(bee_cat, dtype=torch.long),
            "plant_cat": torch.as_tensor(plant_cat, dtype=torch.long),
            "grain": torch.as_tensor(grain, dtype=torch.long),
            "bee_num": torch.as_tensor(np.column_stack(bee_numeric), dtype=torch.float32),
            "plant_num": torch.as_tensor(np.column_stack(plant_numeric), dtype=torch.float32),
            "encounter": torch.as_tensor(encounter, dtype=torch.float32),
            "detection": torch.as_tensor(detection, dtype=torch.float32),
            "target": torch.as_tensor(frame["interaction_observed"].to_numpy(float), dtype=torch.float32),
        }


class FactorizedNetwork(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.bee_embeddings = nn.ModuleList()
        self.plant_embeddings = nn.ModuleList()
        bee_embedding_total = plant_embedding_total = 0
        for column in BEE_CATS:
            classes = len(encoder.vocab[column]) + 1
            dimension = embedding_size(classes)
            self.bee_embeddings.append(nn.Embedding(classes, dimension))
            bee_embedding_total += dimension
        for column in PLANT_CATS:
            classes = len(encoder.vocab[column]) + 1
            dimension = embedding_size(classes)
            self.plant_embeddings.append(nn.Embedding(classes, dimension))
            plant_embedding_total += dimension
        grain_classes = len(encoder.vocab["sampling_grain"]) + 1
        self.grain_embedding = nn.Embedding(grain_classes, 3)
        bee_num = len(BEE_FLAGS) + 1 + 32
        plant_num = 2 * len(PLANT_TRAITS) + len(PLANT_FLAGS) + 1 + 8
        self.bee_tower = nn.Sequential(
            nn.Linear(bee_embedding_total + bee_num, 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, 32)
        )
        self.plant_tower = nn.Sequential(
            nn.Linear(plant_embedding_total + plant_num, 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, 32)
        )
        self.compatibility_residual = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.encounter_network = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 1))
        self.detection_network = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bee_cat, plant_cat, grain, bee_num, plant_num, encounter, detection = batch
        bee_embed = torch.cat([embedding(bee_cat[:, i]) for i, embedding in enumerate(self.bee_embeddings)], dim=1)
        plant_embed = torch.cat([embedding(plant_cat[:, i]) for i, embedding in enumerate(self.plant_embeddings)], dim=1)
        bee_latent = self.bee_tower(torch.cat([bee_embed, bee_num], dim=1))
        plant_latent = self.plant_tower(torch.cat([plant_embed, plant_num], dim=1))
        compatibility = (
            (bee_latent * plant_latent).sum(dim=1) / math.sqrt(bee_latent.shape[1])
            + self.compatibility_residual(torch.cat([bee_latent, plant_latent], dim=1)).squeeze(1)
        )
        encounter_value = self.encounter_network(encounter).squeeze(1)
        detection_value = self.detection_network(
            torch.cat([self.grain_embedding(grain), detection], dim=1)
        ).squeeze(1)
        logits = self.bias + compatibility + encounter_value + detection_value
        return logits, compatibility, encounter_value, detection_value


def tensors_to_dataset(tensors: dict[str, torch.Tensor]) -> TensorDataset:
    order = ["bee_cat", "plant_cat", "grain", "bee_num", "plant_num", "encounter", "detection", "target"]
    return TensorDataset(*(tensors[column] for column in order))


def train_one(encoder: Encoder, train: pd.DataFrame, validation: pd.DataFrame, seed: int) -> FactorizedNetwork:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_data = tensors_to_dataset(encoder.transform(train))
    validation_tensors = encoder.transform(validation)
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = FactorizedNetwork(encoder).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_state, best_loss, stale = None, float("inf"), 0
    validation_inputs = tuple(validation_tensors[column].to(DEVICE) for column in [
        "bee_cat", "plant_cat", "grain", "bee_num", "plant_num", "encounter", "detection"
    ])
    validation_y = validation_tensors["target"].to(DEVICE)
    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in loader:
            inputs = tuple(value.to(DEVICE) for value in batch[:-1])
            target = batch[-1].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits, compatibility, encounter, detection = model(inputs)
            regularizer = 1e-4 * (compatibility.square().mean() + encounter.square().mean() + detection.square().mean())
            loss = criterion(logits, target) + regularizer
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits, _, _, _ = model(validation_inputs)
            value = float(criterion(logits, validation_y).cpu())
        if value < best_loss - 1e-5:
            best_loss, stale = value, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if epoch >= 15 and stale >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_model(model: FactorizedNetwork, encoder: Encoder, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tensors = encoder.transform(frame)
    inputs = tuple(tensors[column].to(DEVICE) for column in [
        "bee_cat", "plant_cat", "grain", "bee_num", "plant_num", "encounter", "detection"
    ])
    model.eval()
    with torch.no_grad():
        logits, compatibility, encounter, detection = model(inputs)
    return tuple(value.cpu().numpy() for value in (logits, compatibility, encounter, detection))


def expected_calibration_error(y: np.ndarray, p: np.ndarray) -> float:
    bins = np.minimum((p * 10).astype(int), 9)
    return float(sum(
        (bins == item).mean() * abs(y[bins == item].mean() - p[bins == item].mean())
        for item in range(10) if (bins == item).any()
    ))


def score(name: str, variant: str, frame: pd.DataFrame, probability: np.ndarray) -> dict:
    y = frame["interaction_observed"].to_numpy(int)
    return {
        "split": name, "model": variant, "rows": len(frame), "events": int(frame["event_id"].nunique()),
        "positive_rate": float(y.mean()), "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)), "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "calibration_error": expected_calibration_error(y, probability),
    }


def evaluate(name: str, outer_train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    internal = outer_train["event_id"].map(lambda value: bucket(str(value), 7))
    validation = outer_train[internal.eq(0)].copy()
    calibration = outer_train[internal.eq(1)].copy()
    fit = outer_train[~internal.isin([0, 1])].copy()
    metric_rows = []
    output = test[[
        "event_id", "study", "site", "date", "year", "week", "bee", "plant",
        "interaction_observed", "observed_frequency", "beegap_exact_match",
        "has_any_open_floral_resource_trait",
    ]].copy()
    output.insert(0, "split", name)

    for variant, use_traits in (("factorized_no_biological_traits", False), ("factorized_traits", True)):
        encoder = Encoder(fit, use_traits=use_traits)
        ensemble_logits, components = [], [[], [], []]
        calibration_logits = []
        for member in range(ENSEMBLE_SIZE):
            model = train_one(encoder, fit, validation, BASE_SEED + member)
            cal_values = predict_model(model, encoder, calibration)
            test_values = predict_model(model, encoder, test)
            calibration_logits.append(cal_values[0])
            ensemble_logits.append(test_values[0])
            for index in range(3):
                components[index].append(test_values[index + 1])
        cal_logit = np.mean(calibration_logits, axis=0)
        test_logit = np.mean(ensemble_logits, axis=0)
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=BASE_SEED).fit(
            cal_logit.reshape(-1, 1), calibration["interaction_observed"].to_numpy(int)
        )
        probability = calibrator.predict_proba(test_logit.reshape(-1, 1))[:, 1]
        metric_rows.append(score(name, variant, test, probability))
        output[f"p_{variant}"] = probability
        if variant == "factorized_traits":
            output["compatibility_contribution"] = np.mean(components[0], axis=0)
            output["encounter_contribution"] = np.mean(components[1], axis=0)
            output["relative_detection_contribution"] = np.mean(components[2], axis=0)
            output["ensemble_probability_sd"] = np.std(1 / (1 + np.exp(-np.asarray(ensemble_logits))), axis=0)
    return metric_rows, output


def splits(data: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    definitions = []
    held = data["event_id"].map(lambda value: bucket(str(value), 5)).eq(0)
    definitions.append(("held_event", data[~held], data[held]))
    pair = (data["bee"] + "|" + data["plant"]).map(lambda value: bucket(value, 5)).eq(0)
    definitions.append(("unseen_pair", data[~pair], data[pair]))
    plant = data["plant"].map(lambda value: bucket(value, 5)).eq(0)
    definitions.append(("unseen_plant", data[~plant], data[plant]))
    bee = data["bee"].map(lambda value: bucket(value, 5)).eq(0)
    definitions.append(("unseen_bee", data[~bee], data[bee]))
    site_years = data.groupby(["study", "site"])["year"].nunique()
    eligible_sites = set(site_years[site_years.ge(2)].index)
    eligible = pd.MultiIndex.from_frame(data[["study", "site"]]).isin(eligible_sites)
    latest = data.groupby(["study", "site"])["year"].transform("max")
    future = eligible & data["year"].eq(latest)
    definitions.append(("future_year", data[~future], data[future]))
    for fold in range(5):
        test_studies = {study for study in data["study"].unique() if bucket(str(study), 5) == fold}
        mask = data["study"].isin(test_studies)
        definitions.append((f"unseen_study_fold_{fold}", data[~mask], data[mask]))
    return definitions


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA / "factorized_event_records.csv.gz", parse_dates=["date"])
    metric_rows, prediction_parts = [], []
    for name, train, test in splits(data):
        if test.empty:
            continue
        print(f"{name}: train={len(train):,}, test={len(test):,}", flush=True)
        metrics, predictions = evaluate(name, train.copy(), test.copy())
        metric_rows.extend(metrics)
        prediction_parts.append(predictions)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    source = predictions[predictions["split"].str.startswith("unseen_study_fold_")]
    study_rows = []
    for (study, variant), group in source.melt(
        id_vars=["study", "event_id", "interaction_observed"],
        value_vars=["p_factorized_no_biological_traits", "p_factorized_traits"],
        var_name="variant", value_name="probability",
    ).groupby(["study", "variant"]):
        model_name = variant.removeprefix("p_")
        if group["interaction_observed"].nunique() == 2:
            study_rows.append(score(f"unseen_study::{study}", model_name, group, group["probability"].to_numpy()))
    study_metrics = pd.DataFrame(study_rows)
    metrics = pd.concat([metrics, study_metrics], ignore_index=True)
    metrics.to_csv(OUT / "factorized_ai_metrics.csv", index=False)
    predictions.to_csv(OUT / "factorized_ai_predictions.csv.gz", index=False, compression="gzip")

    primary = metrics[metrics["split"].isin([
        "held_event", "unseen_pair", "unseen_plant", "unseen_bee", "future_year"
    ])]
    summary = {
        "device": str(DEVICE), "ensemble_members": ENSEMBLE_SIZE,
        "rows": len(data), "studies": int(data["study"].nunique()),
        "models": ["factorized_no_biological_traits", "factorized_traits"],
        "primary_metrics": primary.to_dict(orient="records"),
        "component_interpretation": (
            "Regularized relative log-odds contributions, not separately identified literal probabilities."
        ),
        "inductive_rule": "Exact species identity is excluded; unseen taxa use traits and taxonomic backoff.",
        "uncertainty": "Two-member deep ensemble plus held-event probability calibration.",
    }
    (OUT / "factorized_ai_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
