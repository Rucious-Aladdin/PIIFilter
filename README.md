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

- A CUDA GPU is needed for the local LLM (Qwen2.5-7B-Instruct in bf16 needs about 16 GB of
  GPU memory). GLiNER also runs on CPU with `--device cpu`.
- The first run downloads the models from Hugging Face: GLiNER (about 1 GB) and, for the
  local backend, Qwen2.5-7B-Instruct (about 15 GB).
- API keys for the `openai` and `anthropic` backends go in a `.env` file at the repository
  root (it is git-ignored and loaded by the experiment script):

  ```
  OPENAI_API_KEY=...
  ANTHROPIC_API_KEY=...
  ```

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
# local open-weight LLM (default: Qwen/Qwen2.5-7B-Instruct, greedy decoding)
uv run python experiments/toy_experiment.py --backend local

# hosted LLMs (need the API key in .env)
uv run python experiments/toy_experiment.py --backend openai     # default model: gpt-5-mini
uv run python experiments/toy_experiment.py --backend anthropic  # default model: claude-opus-5-5
```

| Option | Default | Meaning |
|---|---|---|
| `--backend` | `local` | `local`, `openai` or `anthropic` |
| `--model` | per backend | LLM model id, e.g. `--model Qwen/Qwen2.5-14B-Instruct` |
| `--threshold` | `0.5` | GLiNER confidence cut-off |
| `--device` | `cuda` | device for GLiNER and the local LLM |
| `--samples` | `experiments/scenario_samples` | folder of scenario JSON files |
| `--out` | `results/toy` | outputs go to `<out>/<backend>/` |

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

### Results (local backend)

Full outputs: [results/toy/local/toy_results.md](results/toy/local/toy_results.md).

| | Example 1 (clean) | Example 2 (noisy) |
|---|---|---|
| GLiNER confidence (NAME / ADDRESS / PHONE) | 0.999 / 0.969 / 1.000 | 0.686 / 0.921 / 0.985 |
| Private? | Yes | Yes |
| Enough context? | Yes | Yes |
| Usable answer? | No: sign-off left as `<NAME_1` | No: same |

- GLiNER masked all three values in both prompts, so nothing leaked. On the noisy prompt
  the lowercase name's confidence fell from 0.999 to 0.686, close to the 0.5 cut-off.
- The LLM used every placeholder correctly but ended both answers with `<NAME_1` (no
  closing `>`), which simple replacement does not restore.
- In the noisy example the restored values keep the user's typos (`41 maple stret`).

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
