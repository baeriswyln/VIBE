# DataAnalysis

## Requirements

The project uses uv as a package manager. Use `uv sync` to install all required packages.

The datasets must be stored in `data/raw`. All notebooks will look there to get the data, and also store any temporary
results in there.

The results can be logged to wandb (flags in the training methods). For this reason, wand will require a login on first
startup of the classification notebook. 

## Notebook overview

All notebooks should be run in their numbered order, as they produce outputs required by the next notebook.

| Notebook                              | Description                                                                                                               |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| 11_preprocessing                      | The main preprocessing stage. Creates filtered and annotated versions of the dataset, as well as the fixed epoch dataset. |
| 12_data_quality_analysis              | Basic anaylsis of metrics that indicate the data quality, and creates a new dataset with rejected epochs (AutoReject).    |
| 13_free_variable_comparison           | Analysis of the signals given free variables.                                                                             |
| 14_robust_epoching                    | Creating the robust epoch dataset with overlapping epochs, and non-fixed pinch locations.                                 |
| 15_classification                     | Runs all models with all available datasets and datasplits. Takes a long time (multiple hours on an RTX 2070)             |
| 16_results_analysis                   | General analysis of the classification results.                                                                           |
| 17_feedback_and_significance_analysis | Analysis of the feedbacks impact on results.                                                                              |
| 18_publication_figures                | Additional figure creation for the thesis.                                                                                |
| 19_induced_error_errp                 | Analysing the ERRP for the induced errors.                                                                                |
| 20_teaser_creator                     | A small notebook creating the figure used as teaser in the thesis.                                                        |
