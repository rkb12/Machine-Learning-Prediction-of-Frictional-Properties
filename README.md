# Machine-Learning-Prediction-of-Frictional-Properties
This code is used for ML fitting and predicting the adhesion, corrugation, and van der Waals energy of bilayer materials. Bilayer materials are taken from our recently developed BMDB database (High Throughput Calculations of Bilayer Materials: [BMDB Database](//doi.org/10.6084/m9.figshare.21799475), [Sci. Data 10, 232, 2023](https://www.nature.com/articles/s41597-023-02146-7).

N.B.: We used [scikit-learn](https://scikit-learn.org/stable/), [SHAP](https://pypi.org/project/shap/), [XGBoost](https://pypi.org/project/xgboost/), [python](https://anaconda.org/anaconda/python), and python libraries like numpy, matplotlib, seaborn, pandas etc.

The following files are available for editing the code:

1. ml_models.py: contains all the calculators, like ML model fitting, different plots related to ML models, and shaply analysis.

2. result.py: contains all calling functions to calculate the required properties

3. Datasets are available in the "data" directory

4. Codes for three different ML models are available in the "individual_models" directory

5. To predict desired frictional response properties, you have to keep your new input data in the "test" directory in the name and format of "X_new.csv" which is given in the "test" directory. Then choose a required option by running the result.py code to calculate predicted responses.


To see outputs:
python result.py
 1 Adhesion energy 
 2 Corrugation energy 
 3 Van der Waals energy 
 4 Exit 

Choose an option lets 1, then press enter; you will find the following options

Choose options from 1-10: 
 1 Features 
 2 ML metrics 	 3 scatterd plot  	 4 residual error 
 5 feature importance by fitted model 	 6 shaply interpretation of features 
 7 shaply dependence plot for individual features 
 8 shaply force plot for invidual bilayers (you can enter row_number of bilayers) 
 9 Input x to predict y (keep your x in test directory with name X_new.csv) 
 10 exit 

Choose your required option from above and see the results
