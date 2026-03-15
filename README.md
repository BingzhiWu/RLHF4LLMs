# RLHF4LLMs

Project paper: [Overleaf manuscript](https://www.overleaf.com/7793569333nsvdtsgrsdvh#92703f)

This repository is a reproducible RLHF coursework/experiment workspace covering the full pipeline from supervised fine-tuning (SFT), to reward modeling (RM), to PPO alignment, and finally cross-stage result comparison. The project is notebook-based and is intended to be easy to run step by step in either Google Colab or a local Python environment.

The internal notebook logic has been kept unchanged. This cleanup focuses only on repository-level documentation and project organization so the work is easier to understand, reproduce, deploy, and share.

## 1. Project Scope

The repository contains four main experiment stages:

1. `notebooks/sft/*.ipynb`
   Fine-tunes a base language model with SFT and produces a LoRA adapter that can be used as the starting policy for PPO.
2. `notebooks/reward_model/*.ipynb`
   Trains a reward model on preference data to distinguish better and worse responses.
3. `notebooks/PPO/*.ipynb`
   Runs PPO alignment using the saved SFT model and reward model.
4. `notebooks/analysis/04_Compare_Results_HH_RLHF.ipynb`
   Compares the base model, SFT model, and PPO model and exports consolidated results.

## 2. Notebook Overview

The repository currently includes both 0.5B and 1.5B experiment tracks organized by stage:

| Notebook | Purpose | Default model/data |
|---|---|---|
| `notebooks/sft/01_SFT_Baseline.ipynb` | SFT baseline training | `Qwen/Qwen2.5-0.5B` + `tatsu-lab/alpaca` |
| `notebooks/sft/01_SFT_Baseline_1p5B.ipynb` | 1.5B SFT training | `Qwen/Qwen2.5-1.5B` + `tatsu-lab/alpaca` |
| `notebooks/reward_model/02_Reward_Model.ipynb` | HH-RLHF reward model | `Qwen/Qwen2.5-0.5B` + `Anthropic/hh-rlhf` |
| `notebooks/reward_model/02_Reward_Model_1p5B.ipynb` | 1.5B HH-RLHF reward model | `Qwen/Qwen2.5-1.5B` + `Anthropic/hh-rlhf` |
| `notebooks/PPO/03_PPO.ipynb` | PPO alignment training | Depends on 0.5B SFT + RM outputs |
| `notebooks/PPO/03_PPO_1p5B.ipynb` | 1.5B PPO alignment training | Depends on 1.5B SFT + RM outputs |
| `notebooks/PPO/03_PPO_HH_RLHF_1p5B.ipynb` | 1.5B PPO with HH-RLHF prompts | Depends on 1.5B SFT + RM outputs |
| `notebooks/analysis/04_Compare_Results_HH_RLHF.ipynb` | Unified comparison and export | Compares base / SFT / PPO |

## 3. Recommended Execution Order

### 0.5B main track

1. Run `01_SFT_Baseline.ipynb`
2. Run `02_Reward_Model.ipynb`
3. Run `03_PPO.ipynb`
4. Run `04_Compare_Results_HH_RLHF.ipynb`

### 1.5B main track

1. Run `01_SFT_Baseline_1p5B.ipynb`
2. Run `02_Reward_Model_1p5B.ipynb`
3. Run `03_PPO_1p5B.ipynb` or `03_PPO_HH_RLHF_1p5B.ipynb`
4. Run `04_Compare_Results_HH_RLHF.ipynb`

## 4. Models and Datasets

### Base models

- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen2.5-1.5B`

### Datasets

- SFT: `tatsu-lab/alpaca`
- Reward modeling / comparison: `Anthropic/hh-rlhf`
Notes:

- `Anthropic/hh-rlhf` may contain harmful, unsafe, or offensive content, which is normal for preference-learning benchmarks.
- The repository does not store local dataset copies. Datasets are downloaded through Hugging Face `datasets`.

## 5. Environment Setup

### Open Google Colab in the browser

The recommended workflow for this project is to run the notebooks in Google Colab.

1. Open [Google Colab](https://colab.research.google.com/) in your browser.
2. Sign in with the Google account that has access to your Google Drive.
3. Choose one of the following ways to open the notebooks:
   - Upload the `.ipynb` files from this repository directly into Colab
   - Upload the repository to Google Drive first, then open the notebooks from Drive in Colab
4. When prompted, allow Colab to connect to Google Drive so saved models and results can persist across sessions.

### Recommended notebook access pattern

Because the notebooks save intermediate models and results across multiple stages, the cleanest setup is:

- keep the repository files together in one Google Drive folder
- open notebooks from that same folder in Colab
- run the notebooks in stage order so later notebooks can find earlier outputs

The notebooks are designed around the default Colab storage root:

```text
/content/drive/MyDrive/RLHF4LLMs
```

### Runtime and performance guidance

When opening a notebook in Colab, enable a GPU runtime:

1. In Colab, open `Runtime` -> `Change runtime type`
2. Set `Hardware accelerator` to `GPU`
3. Save and reconnect the session

Performance expectations:

- 0.5B track: the most practical option for coursework replication, faster iteration, and lower VRAM usage
- 1.5B track: significantly heavier, slower, and more sensitive to available GPU memory
- PPO notebooks: usually the most expensive stage because they depend on both prior model outputs and reinforcement-learning training

Practical recommendation:

- start with the 0.5B notebooks if your goal is to verify the pipeline quickly
- use the 1.5B notebooks only when you have enough runtime budget and GPU memory
- if Colab free-tier resources are unstable, rerun from the last completed saved stage rather than restarting the full pipeline

### Dependencies in Colab

The notebooks already contain package installation cells, so no manual environment setup is required before opening them in the browser.

If Colab asks for a runtime restart after installation, restart the runtime and continue running the notebook from the next required step.

## 6. Running the Project

### Option A: Google Colab

This project is most naturally run in Colab. The notebooks already include:

- Colab environment detection
- Google Drive mount logic
- Automatic output directory creation

The default Colab root path is:

```text
/content/drive/MyDrive/RLHF4LLMs
```

This means:

- Each stage writes its training artifacts to Google Drive
- PPO and comparison notebooks load model outputs saved by earlier stages
- When the notebooks are run in Colab, the saved models, checkpoints, logs, and result files are stored in Google Drive rather than only inside the temporary Colab session

### Option B: Local execution

The notebooks also support local execution, but the path assumptions differ slightly from Colab:

- The notebooks are now grouped under `notebooks/` by stage
- The training artifacts are still intended to be created as run directories such as `sft_baseline_lora/`, `reward_model_hh_rlhf_lora/`, and `ppo_minimal_lora/`
- Some notebook path logic was originally written for a flatter layout, so local runs should be checked carefully before launching long jobs

If a notebook assumes the repository root as `BASE_ROOT`, make sure your local execution setup matches that expectation.

## 7. Repository Structure

The repository currently follows a stage-based notebook layout:

```text
RLHF4LLMs/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    ├── sft/
    │   ├── 01_SFT_Baseline.ipynb
    │   └── 01_SFT_Baseline_1p5B.ipynb
    ├── reward_model/
    │   ├── 02_Reward_Model.ipynb
    │   └── 02_Reward_Model_1p5B.ipynb
    ├── PPO/
    │   ├── 03_PPO.ipynb
    │   ├── 03_PPO_1p5B.ipynb
    │   └── 03_PPO_HH_RLHF_1p5B.ipynb
    └── analysis/
        └── 04_Compare_Results_HH_RLHF.ipynb
```

After running notebooks, experiment directories such as the following will be created:

```text
sft_baseline_lora/
reward_model_hh_rlhf_lora/
ppo_minimal_lora/
sft_baseline_qwen25_1p5b_lora/
reward_model_hh_rlhf_qwen25_1p5b_lora/
ppo_minimal_qwen25_1p5b_lora/
ppo_hh_rlhf_qwen25_1p5b_lora/
compare_results_.../
```

Where these directories are stored depends on the runtime:

- In Google Colab, they are typically saved under Google Drive, usually inside `/content/drive/MyDrive/RLHF4LLMs`
- In local execution, they are typically created under the local repository root

These run directories typically contain:

- `checkpoints/`
- `final_model/`
- `results/`
- `tb_logs/` or other logs

## 8. Outputs

Different notebooks export different result files, for example:

- `dataset_info.json`
- `train_metrics.json`
- `eval_metrics.json`
- `save_info.json`
- `base_samples.json`
- `sft_samples.json`
- `paired_samples.json`
- `rm_pair_scores.json`
- `before_ppo_samples.json`
- `ppo_log_history.json`
- `ppo_sample_comparison.json`
- comparison `.csv` / `.json` summary files

These outputs support:

- training traceability
- sample-level qualitative analysis
- cross-stage model comparison
- report writing and result visualization

## 9. Reproducibility Notes

The notebooks already include several features that support reproducibility:

- fixed stage order
- explicit `RUN_NAME` values
- automatic directory creation
- deterministic output file naming
- TensorBoard logging
- saved samples before and after training
- JSON/CSV exports suitable for reports

To keep the resulting directory structure consistent with this project, it is recommended to:

1. Keep the notebook `RUN_NAME` values unchanged
2. Run notebooks in stage order
3. Use the same Drive root or the same local repository root throughout the workflow
4. Preserve each stage's `final_model/` and `results/` directories

## 10. Cross-Platform Notes

This project is primarily intended to run in Google Colab.

In practice, that means the operating system matters much less than the ability to use a modern web browser and access Colab.

If a system can:

- open a browser
- access Google Colab
- connect to Google Drive

then it can be used to run this project workflow.

This makes the project broadly usable from:

- macOS
- Windows
- Linux
- ChromeOS
- other browser-capable systems

Because the main execution environment is Colab, local differences in Python setup, CUDA installation, and package compatibility are less important than they would be in a fully local workflow.

## 11. Sharing and Submission Guidance

To keep the repository as a reproducible working repository instead of a model-artifact dump:

- track notebooks, `README.md`, `requirements.txt`, and small report-ready outputs in Git
- exclude large checkpoints, caches, and intermediate model artifacts from version control
- if results need to be shared, prefer compact `results/*.json` or `results/*.csv` exports

## 12. Current Boundaries

- The project is notebook-first, not a packaged Python module
- No dataset copies are bundled in the repository
- The notebooks are organized more cleanly now, but some path logic may still assume a repository-root execution context
