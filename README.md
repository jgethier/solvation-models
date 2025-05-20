# Theory-Informed Machine Learning to Predict Solvation Free Energy

File Descriptions:

plot_results.ipynb - Jupyter notebook to create manuscript figures from saved models and result files. 
solvation_model_results.ipynb - Jupyter notebook to analyze data, calculate results, and plots in manuscript.

main.py - Python code to run Hyperopt training, exploration campain, and metrics for saved models tuned via Hyperopt.
utils.py - Python file containing functions required for main.py including descriptor generation for solute and solvent.
model.py - Python file containing Gaussian Process Regression model adapted from EDBO+.
data.py - Python file for data loading and curation.

*.csv files are the experimental data files curated from Minnesota and dGSolvDB1 databases. 
