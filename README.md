# AIS kinematic-consistency screening

Analysis software for *Evaluating Kinematic-Consistency Screening of Synthetic AIS Position Manipulation*. The code compares position-derived motion with reported speed and course, evaluates classifiers across the Guam and Los Angeles test sets, and estimates uncertainty by resampling vessels.

## Check the published confidence intervals

From the repository root, run `python scripts/summarize_bootstrap.py` with pandas installed. The included CSV contains 1,000 paired vessel-cluster draws per experiment. The command reports HGB-minus-RF means and percentile intervals for F1 and AUPRC in all four experiments. For example, the Guam Generator-A F1 difference is −0.0037 [−0.0082, 0.0007].

## Analysis with external inputs

Use `conda env create -f environment.yml` and `conda activate AIS_SCI_PAPER` for the study environment. The protocol and ordered feature list are in `config/experiment_protocol.json` and `data/windows_v1/model_feature_columns.txt`.

- For the descriptive interval and class statistics, place the Guam CSV at `data/raw/ais_sample.csv` and the LA CSVs at `external_validation/la_2025/raw/la_2025_*.csv`, then run `python scripts/reviewer_t3_report_intervals.py`. Numbered preprocessing scripts in `src/` and `src/src/` implement the region-specific screening and feature definitions.
- The benchmark starts from the study's prepared scenario or window files. To build windows from scenario CSVs, use `src/07_build_sliding_window_dataset.py` and `src/src/17_la_06_build_external_windows_v2.py`. The expected directories are `data/synthetic_v2/` and `external_validation/la_2025/synthetic_v3/`. The scenario files and generators are outside this analysis release.
- With the prepared Guam train, validation and test window CSVs in `data/windows_v1/` and the LA test windows in `external_validation/la_2025/windows_v2/`, run `python src/08_train_baseline_models.py` for the five tabular classifiers. The sequential GRU benchmark also requires the corresponding point-level scenario files. Run it with `python src/src/24b_gru_main_benchmark.py`.
- After tabular training, run `python src/src/23_mmsi_cluster_bootstrap_1000.py` for window-level uncertainty or `python src/src/29_event_level_cluster_bootstrap.py` for event-level uncertainty. The adapted Kalman/NIS comparator is in `scripts/reviewer_t4_cv_kalman_baseline.py`.
- Generate the manuscript workflow diagram with `python scripts/plot_workflow.py workflow.pdf`.

The included bootstrap CSV contains summary statistics for resamples. Vessel-level records and fitted model files are external inputs. The paper's Data Availability section identifies the original AIS data sources.

