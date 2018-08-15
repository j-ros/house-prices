# House Prices: Advanced Regression Techniques

This repository contains the submission notebook for the Kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The objective was to get some familiarity with the 
[Light GBM](https://lightgbm.readthedocs.io/en/latest/index.html) model using [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization)
for parameter tuning.

## Submission results

- 1201/4529 with a RMSE score of 0.12448 on the public leaderboard (15/08/2018)

## What I have learned

- Light GBM is a decision tree boosting algorithm. The major advantage over xgboost is that it uses a histogram-based algorithm instead of a pre-sort-based
algorithm for learning decision trees. This has the main advantage of reducing training time and memory usage. Another difference is that it uses leaf-wise tree growth
instead of level-wise tree growth. This means it always chooses the leaf that will reduce the loss the most to grow first. This behaviour is prone to overfitting on
small datasets which is the reason why we have performed tuning using cross-validation on parameters such as max_depth, num_leaves, min_data_in_leaf and feature_fraction.

- Light GBM accepts categorical variables without the need for one-hot-encoding them. It uses an algorithm to find the best split on the histogram for a categorical
feature by sorting based on the training objective at each split. The only caveat is that the categorical variables have to be encoded as integers (not strings).

- We have used Bayesian Optimization to tune the hyperparameters of the model. It works by iteratively constructing a posterior distribution of functions (gaussian process)
that approximates the target function. By evaluating the function in several points the posterior is updated to better reflect the true distribution and gain knowledge
about which areas are worth exploring further.

- A quick exploration of the target variable, SalePrice, reveals that there are some outliers in the dataset. As indicated in the dataset [documentation](http://ww2.amstat.org/publications/jse/v19n3/Decock/DataDocumentation.txt)
there are two points worth discarding since they have a very low price for the living area of the houses.

- The dataset has a great number of NULL values to handle. Since some variables have a high percentage of NULL values and our dataset is not very large, it is
advisable to impute the missing values so as not to discard useful information for the model. The imputation process that we have followed depends largely on the variable.
We have used a mix of domain knowledge about how the data was collected and the values of non-NULL rows to clean the dataset.

- Finally we have used an out-of-folds prediction ensemble for the final prediction, based on a 5-fold split of the training data. 


## Future improvements

- Try model stacking or ensembling with different models for better predictions.
