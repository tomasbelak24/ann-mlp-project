How to run:
create a conda env with python 3.12.9 and activate it

after that run this comman:
pip install requirements.txt

When everything is installed the code should 100% work.

main.py  - contains the script where data is loaded, preprocessed, models are trained, evaluated and eventually the best one is tested.
mlp.py - contains MLP class which handles layer dimensions, the weight initialization and backpropagation.
classifier.py - contains MLPclassifier class that wraps around the MLP class and handles trainig 
best_model.py - used for adjusting plots of the results of the best model
best_model_params.json - contains the best hyperparameters from both grid searches that were done during this project
