# Experiment Log Schema

Shared schema for tracking experiment runs across all three studies (circuit-tracing, sae-comparison, distributed-interp).

---

## JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "experiment-log-entry",
  "title": "Experiment Log Entry",
  "type": "object",
  "required": [
    "experiment_id",
    "paper",
    "hypothesis",
    "model_name",
    "dataset",
    "hyperparameters",
    "results",
    "status"
  ],
  "properties": {
    "experiment_id": {
      "type": "string",
      "description": "Unique slug. Convention: {paper-abbrev}-{model-abbrev}-{method}-v{n}. E.g. ioi-llama-ap-v1"
    },
    "paper": {
      "type": "string",
      "enum": ["circuit-tracing", "sae-comparison", "distributed-interp"],
      "description": "Which of the three papers this run belongs to"
    },
    "hypothesis": {
      "type": "string",
      "description": "The falsifiable claim being tested. Reference labels from the experiment brief (H1, H1a, H2 ...) and state the direction of the expected effect."
    },
    "model_name": {
      "type": "string",
      "description": "HuggingFace model ID or absolute path to a local checkpoint"
    },
    "dataset": {
      "type": "object",
      "required": ["name", "n_prompts"],
      "properties": {
        "name": {
          "type": "string",
          "description": "Short dataset name, e.g. ioi-wang2022, pile-random"
        },
        "n_prompts": {
          "type": "integer",
          "minimum": 1,
          "description": "Number of prompts / examples used in this run"
        },
        "source": {
          "type": "string",
          "description": "Citation, URL, or generation script that produced the dataset"
        },
        "split": {
          "type": "string",
          "description": "train / val / test / all, if applicable"
        },
        "conditions": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Experimental conditions present in the dataset, e.g. [\"ABB\", \"ABC\"]"
        }
      }
    },
    "hyperparameters": {
      "type": "object",
      "description": "Free-form dict of all intervention or training parameters that vary across runs. Keys and types depend on the experiment type.",
      "examples": [
        {
          "intervention": "activation_patching",
          "patch_granularity": "head",
          "corruption_method": "name_swap",
          "n_ref_sentences": 200,
          "dtype": "float32"
        }
      ]
    },
    "results": {
      "type": ["object", "null"],
      "description": "Free-form dict of metric-name → value. Null when the run has not yet completed.",
      "examples": [
        {
          "mean_logit_diff_clean": 4.21,
          "mean_logit_diff_corrupted": -1.03,
          "mean_recovery_score": 0.87,
          "top1_accuracy_clean": 0.94,
          "top1_accuracy_circuit_only": 0.89,
          "n_circuit_heads_identified": 14
        }
      ]
    },
    "status": {
      "type": "string",
      "enum": ["planned", "running", "done"],
      "description": "Lifecycle state of the run"
    },
    "created_at": {
      "type": "string",
      "format": "date",
      "description": "ISO-8601 date the entry was first created"
    },
    "updated_at": {
      "type": "string",
      "format": "date",
      "description": "ISO-8601 date the entry was last modified"
    },
    "notes": {
      "type": "string",
      "description": "Free-form observations, blockers, deviations from protocol, or next steps"
    },
    "tags": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Optional labels for filtering, e.g. [\"replication\", \"pilot\", \"neurips-workshop\"]"
    }
  }
}
```

---

## Field Notes

| Field | Required | Notes |
|---|---|---|
| `experiment_id` | yes | Must be unique across all logs. Bump `v{n}` on re-runs with changed hyperparameters. |
| `paper` | yes | Ties the run to one of the three papers. |
| `hypothesis` | yes | Copy the label from the brief (e.g. "H1a") and state the predicted direction; don't just reference the label. |
| `model_name` | yes | Use the canonical HuggingFace ID so runs are reproducible without prose lookup. |
| `dataset.n_prompts` | yes | Record the actual count used, not the target. |
| `hyperparameters` | yes | Any parameter that, if changed, would warrant a new `experiment_id`. Leave out constants that are fixed by the protocol. |
| `results` | yes | Set to `null` for planned/running entries. Populate with final values when `status` moves to `done`. |
| `status` | yes | `planned` → intent recorded, no compute started. `running` → job active. `done` → results finalized. |
| `notes` | no | Use for anything that would otherwise live in a lab notebook: deviations, surprises, next steps. |

---

## File Naming Convention

Store entries as individual JSON files under `docs/experiment-logs/`:

```
docs/experiment-logs/{experiment_id}.json
```

One file per run. Group related runs by ID prefix (e.g. all `ioi-*` entries belong to circuit-tracing).
