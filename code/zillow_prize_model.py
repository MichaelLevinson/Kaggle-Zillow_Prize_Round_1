import os, sys, gc
import numpy as np, pandas as pd, scipy.stats as stats
import lightgbm as lgb
import zillow_prize_utilities as zpu
import time

from pathlib import Path
from sklearn import cross_validation as cv


# __author__  : Michael Levinson
# __project__ : Zillow Prize - Round 1


# note: Scaling/normalizaiton is ignored in the case of lightgbm
# tree ensembles are scale-invariant


class ZillowModel(object):
  """Zillow Prize Model

  :path: string, path to data directory
  :submission_filename: string, submission filename
  :seed: int, initialize randomstate for reproducibility
  """
  def __init__(self, path, submission_filename, seed=0):
    self.path = path
    self.submission_filename = submission_filename
    self.seed = seed
    self.model_name = 'lightgbm'

  def _initialize_sub_file(self):
    self.sub = pd.read_csv(self.path + '/' + 'sample_submission.csv')
    self.sub_cols = self.sub.columns

  def _create_submission_file(self, model, prop, features):
    """ Create Submission File

    :model: lightgbm trained model
    :prop: pandas DataFrame, properties dataframe to get predictions
    :features: list of strings, feature names for predictive model
    """

    # (year, month)
    pred_times = [(0,10),(0,11),(0,12),(1,10),(1,11),(1,12)]
    preds = self.sub.copy()
    prop['month'] = -1
    prop['year'] = -1
    i = 0

    for (yr, mo) in pred_times:
      # to ensure memory is released more quickly. Issues with lightgbm 0.2
      gc.collect()

      i += 1
      prop['month'] = mo
      prop['year'] = yr
      tic = time.time()
      preds.iloc[:,i] = model.predict(prop[features])
      print('month {}, year {}, done in {} seconds'.format(mo, yr,time.time() - tic))

    preds.to_csv(self.submission_filename, index=False,compression='gzip',  float_format='%.7f')


  def _run(self, dsl_params, model_params=None, INIT_NUM_ROUNDS=20000):
    """ run model

    :dsl_params: dict, parameters for DatasetLoader
    :model_params: dict, parameters for model
    :NUM_ROUNDS: int, when using gradient boosted decision trees, set high for early stopping

    """

    np.random.seed(self.seed)

    self._initialize_sub_file()

    DatasetLoader = zpu.DatasetLoader(**dsl_params)
    prop = DatasetLoader._load()
    prop.set_index('parcelid', inplace = True)
    prop = prop.loc[self.sub.ParcelId]

    features=prop.columns.difference(['parcelid','assessmentyear'])
    features = list(features) + ['month','year']

    train2016 = pd.read_csv(self.path + '/' +'train_2016_v2.csv')
    train2017 = pd.read_csv(self.path + '/' +'train_2017.csv.zip')
    train = pd.concat([train2016,train2017])

    df_train = train.merge(prop.reset_index(), how='left', on='parcelid')

    month  =  df_train.transactiondate.apply(lambda x: pd.to_datetime(x)).dt.month
    year   = df_train.transactiondate.apply(lambda x: pd.to_datetime(x)).dt.year.astype('category',ordered=True).cat.codes

    df_train['month'] = month
    df_train['year']  = year

    county =  df_train.regionidcounty.fillna(0).astype('category').cat.codes

    month_county = (month.astype('str') +'_' +county.fillna(999).astype('int').astype('str'))

    x_train = df_train[features]
    y_train = df_train.logerror

    # treat month_county as what we seek to stratify, provided a more stable cv
    # similiar to a cluster based cv I developed a few years ago
    # k = 3, k = 5 found to overfit
    k = 3
    folds = cv.StratifiedKFold(month_county, k, random_state=2319)

    print('\n training model: {} \n'.format(self.model_name))
    # run cross validation for early stopping
    # best cv : holdout timestep t, model time 1,..., t-1
    cvlgb = lgb.cv(model_params, lgb.Dataset(x_train,label=y_train), INIT_NUM_ROUNDS, folds=folds,early_stopping_rounds=10, verbose_eval=1)
    NUM_ROUNDS = len(pd.DataFrame(cvlgb))
    # using the early stopping feature in lightgbm, we select optimal number of rounds to train on
    model = lgb.train(model_params, lgb.Dataset(x_train, label=y_train), NUM_ROUNDS)

    print('\n creating submission file ... \n')
    self._create_submission_file(model, prop, features)
    print('\n ... done \n')


if __name__ == '__main__':
  _dsl_params = dict(prop_filename = 'properties_2017.csv.zip',
        output_filename = 'processed_prop_test2.csv.gz', path = '../input',
        seed = 5123, _save_to_file = True)


  _model_params = {}
  _model_params['learning_rate'] = 0.001 # shrinkage_rate
  _model_params['boosting_type'] = 'gbdt'
  _model_params['objective'] = 'regression_l1'
  _model_params['metric'] = 'mae'          # or 'mae'
  _model_params['sub_feature'] = 0.5    # feature_fraction
  _model_params['bagging_fraction'] = 0.67 # sub_row
  _model_params['bagging_freq'] = 1
  _model_params['num_leaves'] = 40       # num_leaf
  _model_params['max_depth'] = 18
  _model_params['min_data'] = 500         # min_data_in_leaf
  _model_params['min_hessian'] = 0.05    # min_sum_hessian_in_leaf



  check_procfile = Path(_dsl_params['path'] + '/' + _dsl_params['output_filename'])

  Model = ZillowModel(path='../input', submission_filename = 'sub0001.csv.gz', seed=2112)

  if not check_procfile.is_file():
    print('\n creating processed properties file ... \n')
    DatasetLoader = zpu.DatasetLoader(**_dsl_params)
    DatasetLoader._run()

  Model._run(_dsl_params, _model_params)

