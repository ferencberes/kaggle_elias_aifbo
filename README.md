Kaggle Competition Solution Report - AI-Based Modeling for Energy-Efficient Buildings
================================

Intro
---

This repository heavily builds upon the [example code](https://github.com/boschresearch/elias_aifbo/tree/main) provided by the competition hosts.
I kept the main pipeline (data loading, preprocessing, model training, submission creation) and upgraded some parts to be able to use multiple sensor groups
and to have a more efficient model to predict the state of the HVAC system three hours ahead.

More details about the task, data, and evaluation can be found on the [competition page](https://www.kaggle.com/competitions/ai-based-modeling-for-energy-efficient-buildings/overview).

Software setup
---

The package manager used to set up this repo and that we recommend to get a virtual environment to run the scripts 
is [uv](https://docs.astral.sh/uv/guides/install-python/). We recommend to install it.

Once uv is installed, we recommend the following steps:
1. go to root dir of this repo, 
2. `uv venv`,
3. then run `uv sync`.

Set up and run simple load/preprocess/train/submission pipeline
---

**Run the modeling pipeline to produce my final submission for the competition (June and July 2025 data):**

1. Download the competition data from Kaggle and put it into a (newly created) folder `data/kaggle_dl/` (there should then be folders like
`data/kaggle_dl/RBHU-2025-01/` etc., 
so that the file structure complies with the format required by the `main*.py` script).
2. Run the fixed original main script as a baseline model: `uv run main_baseline.py`. A fix was needed to exclude erronous day of year feature calculation.
3. Run the multichannel model: `uv run main_multi_channels.py`. This script also creates the ensemble of the baseline (50%) and multichannel model (50%).
4. Once run, this produces several outputs, among others the `outputs_2025/final_submission.csv` which reflects my final submission to the challenge.

**Extra experiment for 2024 data:**

1. Download additional data for 2024 provided by competition hosts from Zenodo, [part 1](https://zenodo.org/records/12590466), 
[part 2](https://zenodo.org/records/14591934), and put it into the same folder `data/kaggle_dl/` so that the script can also use it.
2. Change the `YEAR` variable at the top of `main_multi_channels.py` to `2024`.
3. Run `uv run main_multi_channels.py` to train and evaluate models for 2024 data. This setup covers extra channel choice validations for June and July 2024.

Running time
---

For training the neural networks, I used one node from a NVIDIA A100 GPU cluster.

| **Runtime Component**        | **Baseline setup** | **Multichannel setup** |
|------------------------------|--------------------|------------------------|
| **Data Reload Time (mins)**  | 5.27 (only once)   | 0.02 (reloading saved) |
| **Feature Prep Time (mins)** | 0.27               | 21.39                  |
| **Training Time (mins)**     | 4.65               | 8.71                   |
| **Eval & Submission (mins)** | 0.04               | 0.06                   |
| **Total Time (mins)**        | 10.23              | 30.18                  |

My final submission was an ensemble that was the combination of the baseline model and the best multichannel model.
Thus, the **total runtime** is the sum of both runtimes, 10.23 + 30.18 = **40.41 minutes**.

Purpose of the repository and disclaimer
---

This software is a research prototype, solely developed for and published as a solution to the aforementioned ELIAS competition.
It will neither be maintained nor monitored in any way.

License
---

This project is open-sourced under the MIT license. See the
[LICENSE](LICENSE) file for details.
