# Kaggle_Predicting-Parkinson-Disease-Progression

A repo to store my attempt at the AMP Parkinson's Disease Progression Kaggle AI/ML competition. See the [competition overview](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/overview).

## Methodology (so far)
### `NB1_PreProcessing_Data.ipynb`
1. Concat the supplemental clinical data (no associated Prot/Pep data) to the main clinical data. This is done to help train future predictions.
2. Pivot protien and peptide data such that each row is a unique visit_id.
3. Create all labels we will need to predict for by passing future visit data forward to a given visit, and renaming the visit specific columns to match our submission format (i.e., updrs_2 -> updrs_2_plus_0_months).
4. Use both Iterative and KNN imputation to fill in nans for both the Protien and Peptide data. Save as .parquet files.
5. Create a .parquet file that stores missing value locations for Protein and Peptide data. This will be used in future notebooks to adjust training weights.

## Directory Organization
Our main workflow notebooks will be stored in the parent directory and labeled by their order (i.e. 'NB1_' => notebook 1). Other sub-directories are as follows.

* **`raw_inputs`-** Self explanatory. Stores training data downloaded from [Kaggle](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/data). 

* **`prepped_inputs`-** Stores model input files (i.e. a training table) after pre-processing or feature engineering.

* **`submissions`-** Stores submission `.csv` files.

* **`competition_provided_code`-** Stores code provided by the competition.


## To-Do

* **Enable custom loss function (i.e. SMAPE), and use in training (where possible).**
  * This should be passed in as a model_param, such that we can evaluate if training for SMAPE is even beneficial across K-Folds.
    * Use SMAPE + 1.
* Finish our concrete MLModel implementations.
* Do some EDA, without reinventing the wheel, and make public.
* Automate new logging files with a datetime stamp.
* Consider stratified K-Fold!
* Work on feature engineering.
* Set things up for GPU when possible.
* Use Kaggle `Gist` syntax to import our repo after making it public.

All together, with clever FE, optuna optimization overnight, and testing different random states we can get a great score.

## Workflow ideas

* Figure out if KNN imputation will help fill nans.
* Identify a subset of protien/peptide data to use. Maybe with high variability, correlation to outcomes, reference in medical literature, but with a strong amount of observations.
* We may need to train 16 models. One for each UDRS score X month gap. Alternatively we can just train 4 core models, and then one simple linear regression model to adjust for months.
* **Seems like we should start by making a classification model to predict whether a patient is on/off medication based on protien data.**
  * If it works well, we can include the predicted values as a feature.
    * We should train a quick and dirty model predicting UDRS (where the medication has an impact) on rows w/o medication status. If it shows up as a prominent feature across K-Folds then we keep it.

## NOTES
* Imputation (especially across a clinical visit data row) will be key. Note that protein/peptides not found in a clinical sample does NOT indicate that there is zero abundance (see discussion [here](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/discussion/396052))
* We will only be expected to make predictions using visits with Protien/Peptide data.
* We will be using the data from a given month (plus any previous months) to make future predictions at different month steps.
* This means we may want to train a future looking model that is different from the same visit model since we can incorporate time series based features (change over time of a peptide for example).
* Supplemental clinical data is really just to train month step models.
