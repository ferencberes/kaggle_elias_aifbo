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

**TODO: update environment to contain every package, for parquet loading as well!!!**

The package manager used to set up this repo and that we recommend to get a virtual environment to run the scripts 
is [uv](https://docs.astral.sh/uv/guides/install-python/). We recommend to install it.

Once uv is installed, we recommend the following steps:
1. go to root dir of this repo, 
2. `uv venv`,
3. then run `uv sync`.


Set up and run simple load/preprocess/train/submission pipeline
---

A simple sample pipeline for load/preprocess/train/submission is contained in `main.py`. Run it in the following way:

1. Download the competition data from Kaggle and put it into a (newly created) folder `data/kaggle_dl/` (there should then be folders like
`data/kaggle_dl/RBHU-2025-01/` etc., 
so that the file structure complies with the format required by the `main.py` script).
2. Download additional data for 2024 provided by competition hosts from Zenodo, [part 1](https://zenodo.org/records/12590466), 
[part 2](https://zenodo.org/records/14591934), and put it into the same folder `data/kaggle_dl/` so that the script can also use it. 
3. Do `uv run main_multi_channels.py`.
4. Once run, this produces several outputs, among others the `submission_file.csv` which can then be uploaded to the competition

Running time
---

For training the neural networks, I used one node from a NVIDIA A100 GPU cluster.

| **Runtime Component**        | **Baseline setup** | **Multichannel setup** |
|------------------------------|--------------------|------------------------|
| **Data Reload Time (mins)**  | 5.27               | 0.02                   |
| **Feature Prep Time (mins)** | 0.27               | 21.39                  |
| **Training Time (mins)**     | 4.65               | 8.71                   |
| **Eval & Submission (mins)** | 0.04               | 0.06                   |
| **Total Time (mins)**        | 10.23              | 30.18                  |

My final submission was an ensemble that was the combination of the baseline model and the multichannel model.
Thus, the **total runtime** is the sum of both runtimes, 10.23 + 30.18 = **40.41 minutes**.

Purpose of the repository and disclaimer
---

This software is a research prototype, solely developed for and published as a solution to the aforementioned ELIAS competition.
It will neither be maintained nor monitored in any way.

License
---

This project is open-sourced under the MIT license. See the
[LICENSE](LICENSE) file for details.
