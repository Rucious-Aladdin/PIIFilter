# PII Filter

CSED490E term project (POSTECH, Fall 2026). An on-device filter that finds personal
information (PII) in a chatbot prompt, replaces each value with a placeholder such as
`<NAME_1>` before the prompt leaves the device, and puts the real values back into the
LLM's answer.

```
user prompt → detector (NER) → mask → cloud LLM → unmask → answer shown to the user
                                 └── placeholder → value mapping stays on the device
```

The full plan is in [docs/PIIGuard_Proposal_Merged.docx](docs/PIIGuard_Proposal_Merged.docx)
and the slides in [docs/presentation_ppt.pdf](docs/presentation_ppt.pdf).

## Repository layout

```
PIIFilter/                 the filter itself
  masking.py               Span, mask() and unmask() with numbered placeholders
  llm.py                   LLM backends: local (transformers), openai, anthropic
  pipeline.py              run_filter(): detect → mask → LLM → unmask, returns every step
baselines/                 PII detectors compared in the benchmark
  gliner_detector.py       GLiNER (urchade/gliner_multi_pii-v1), given our label names
experiments/
  toy_experiment.py        toy experiment (proposal Section 5.4)
  scenario_samples/        toy experiment inputs, one JSON file per prompt
results/toy/<backend>/     toy experiment outputs (toy_results.json, toy_results.md)
slides/toy_experiment/     Beamer frames for the Toy Experiment slides + compiled preview
docs/                      proposal and presentation
pyproject.toml, uv.lock    environment (uv)
```

Labels used everywhere: `NAME`, `EMAIL`, `PHONE`, `ADDRESS`, `DATE_OF_BIRTH`, `CARD_NUMBER`.

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned
in `.python-version`; uv downloads it if needed.

```bash
# install uv once (skip if `uv --version` works)
curl -LsSf https://astral.sh/uv/install.sh | sh

# create .venv and install the exact versions in uv.lock
uv sync
```

- API keys go in a `.env` file at the repository root (it is git-ignored and loaded by the
  experiment script). The default `openai` backend needs `OPENAI_API_KEY`
  (https://platform.openai.com/api-keys; the API is billed separately from ChatGPT).

  ```
  OPENAI_API_KEY=...
  ANTHROPIC_API_KEY=...
  ```

- The first run downloads GLiNER from Hugging Face (about 1 GB). GLiNER runs on the GPU by
  default; on a machine without one, add `--device cpu`.
- Only the optional `local` backend needs a large GPU: Qwen2.5-7B-Instruct is about 15 GB
  to download and needs about 16 GB of GPU memory in bf16.

## Toy experiment

The toy experiment checks that the whole filter works end to end on two prompts from the
proposal: a message to a landlord typed carefully (clean) and carelessly (noisy). It gives
no scores. For each prompt it records the output of every step and answers three questions:

- **Private?** No PII value (or part of one) appears in the text sent to the LLM.
  Checked automatically against the gold spans in the scenario file.
- **Enough context?** The LLM answer does the task and uses every placeholder.
- **Usable answer?** After unmasking, no placeholder is left in the answer.

The last two are automatic proxies; confirm them by reading the answers in `toy_results.md`.

### Run

```bash
# OpenAI (default; model gpt-5.5) - the setting used for the slides
uv run python experiments/toy_experiment.py

# Anthropic (model claude-opus-5-5)
uv run python experiments/toy_experiment.py --backend anthropic

# local open-weight LLM (Qwen/Qwen2.5-7B-Instruct, greedy decoding)
uv run python experiments/toy_experiment.py --backend local
```

Hosted models do not decode deterministically, so the wording of the answers can change
from run to run.

| Option          | Default                          | Meaning                                                  |
| --------------- | -------------------------------- | -------------------------------------------------------- |
| `--backend`   | `openai`                       | `openai`, `anthropic` or `local`                   |
| `--model`     | per backend                      | LLM model id, e.g. `--model gpt-5-mini`                |
| `--threshold` | `0.5`                          | GLiNER confidence cut-off                                |
| `--device`    | `cuda`                         | device for GLiNER and the local LLM                      |
| `--samples`   | `experiments/scenario_samples` | folder of scenario JSON files                            |
| `--out`       | `results/toy`                  | outputs go to `<out>/<backend>/`                       |

To pick a GPU, prefix the command with e.g. `CUDA_VISIBLE_DEVICES=0`.

The script writes `results/toy/<backend>/toy_results.json` (every step, spans with
confidences, check details, model ids and package versions) and `toy_results.md` (the
same as tables), and prints the Markdown.

The `anthropic` backend uses the Messages API with `effort: medium` and opts into the
server-side refusal fallback (`fallbacks: "default"`) for models that support it.

### Scenario samples

Each file in `experiments/scenario_samples/` is one prompt. Files are run in name order.

```json
{
  "id": "landlord_clean",
  "title": "Example 1 (clean)",
  "prompt": "Write a short message to my landlord. I'm Minji Park from 41 Maple Street, ...",
  "gold": [
    {"label": "NAME", "value": "Minji Park"},
    {"label": "ADDRESS", "value": "41 Maple Street"},
    {"label": "PHONE", "value": "010-2345-6789"}
  ]
}
```

To add a scenario, add a file such as `03_email_card.json`. Each gold `value` must appear
in `prompt` exactly as written.

### Results (gpt-5.5)

Full outputs: [results/toy/openai/toy_results.md](results/toy/openai/toy_results.md).

|                                            | Example 1 (clean)     | Example 2 (noisy)     |
| ------------------------------------------ | --------------------- | --------------------- |
| GLiNER confidence (NAME / ADDRESS / PHONE) | 0.999 / 0.969 / 1.000 | 0.686 / 0.921 / 0.985 |
| Private?                                   | Yes                   | Yes                   |
| Enough context?                            | Yes                   | Yes                   |
| Usable answer?                             | Yes                   | Yes                   |

Answer shown to the user (clean): *Hi, this is Minji Park from 41 Maple Street. My heater is
broken and needs repair. Please call me at 010-2345-6789 when you can. Thank you.*

- GLiNER masked all three values in both prompts, so nothing leaked. On the noisy prompt
  the lowercase name's confidence fell from 0.999 to 0.686, close to the 0.5 cut-off.
- GPT-5.5 used every placeholder and all of them were restored. Two more runs (not saved)
  gave the same checks with slightly different wording.
- Unmasking restores what the user typed, so the noisy answer reads "this is minji park
  from 41 maple stret".
- An earlier run with the local Qwen2.5-7B-Instruct backend is kept in
  [results/toy/local/](results/toy/local/toy_results.md). There the LLM ended both answers
  with `<NAME_1` (no closing `>`), which simple replacement does not restore.

## Slides

`slides/toy_experiment/toy_frames.tex` holds three Beamer frames (setup, the clean example
end to end, results) for the Toy Experiment section of the deck. `preview.pdf` shows them in
a look close to the deck.

```bash
cd slides/toy_experiment
latexmk -pdf preview.tex
```

To use them in the deck, add `\input{toy_frames}` where the empty Toy Experiment slide is.
The deck's preamble needs `\usepackage[T1]{fontenc}`, `tikz`, `tcolorbox`, `booktabs` and
`colortbl`.
With TinyTeX, the packages for the preview are:
`tlmgr install beamer translator fira fontaxes mweights tcolorbox pgf booktabs colortbl cm-super`.
