# Quality Control

Code to reproduce our quality control results in *Appendix C. Quality Control*.

To reproduce our results, repeat these steps below for all datasets (domains):

- Open the `run_experiment_one_dataset.ipynb` notebook
- Specify `DOMAIN` with the dataset (domain) you would like to run quality control on
- Run the notebook and save its results with `USE_ORIGINAL_CONTENT_IMAGES=True`
- Run the notebook and save its results with `USE_ORIGINAL_CONTENT_IMAGES=False`

After collecting all results, run the notebook `visualize_fidelity.ipynb` to visualize the final results
