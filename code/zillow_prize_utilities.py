import os, sys, gc
import numpy as np, pandas as pd

from sklearn.neighbors import KDTree



# __author__  : Michael Levinson
# __project__ : Zillow Prize - Round 1


class DatasetLoader(object):
    """Dataset Lader for Zillow Prize

    :prop_filename: STRING, properties filename
    :output_filename: STRING, processed filename
    :path: STRING, location of data files
    :seed: Int, initialize random seed
    """
    def __init__(self, prop_filename = 'properties_2017.csv.zip',
        output_filename = 'processed_prop.csv.gz', path = '../input',
        seed = 5123, _save_to_file = True):

      self.prop_filename = prop_filename
      self.output_filename = output_filename
      self.path = path
      self.coords = ['longitude','latitude']
      self.seed = seed
      self._save_to_file = _save_to_file


    def _construct_datatypes(self):
      """Define Datatypes for 32-bit loading to conserve memory
      """
      # objects
      OBJ = [(z,object) for z in ['hashottuborspa', 'propertycountylandusecode',
        'propertyzoningdesc','fireplaceflag', 'taxdelinquencyflag']]

      # floats
      FLT = [(z,np.float32) for z in ['bathroomcnt','calculatedbathnbr',
        'lotsizesquarefeet','rawcensustractandblock','taxamount']]

      # Integers
      INT = [('parcelid',np.int32)]

      # Treat like floats
      NINT = [(z,np.float32) for z in ['airconditioningtypeid',
        'architecturalstyletypeid', 'assessmentyear',
        'basementsqft', 'bedroomcnt', 'buildingclasstypeid',
        'buildingqualitytypeid', 'calculatedfinishedsquarefeet',
        'censustractandblock', 'decktypeid', 'finishedfloor1squarefeet',
        'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',
        'finishedsquarefeet50', 'finishedsquarefeet6', 'fips', 'fireplacecnt',
        'fullbathcnt', 'garagecarcnt', 'garagetotalsqft',
        'heatingorsystemtypeid', 'landtaxvaluedollarcnt', 'latitude',
        'longitude', 'numberofstories', 'poolcnt', 'poolsizesum',
        'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'propertylandusetypeid',
        'regionidcity', 'regionidcounty', 'regionidneighborhood', 'regionidzip',
        'roomcnt', 'storytypeid', 'structuretaxvaluedollarcnt',
        'taxdelinquencyyear', 'taxvaluedollarcnt', 'threequarterbathnbr',
        'typeconstructiontypeid', 'unitcnt', 'yardbuildingsqft17',
        'yardbuildingsqft26', 'yearbuilt']]

      # used to identify categorical data types
      NUM = [z for z in ['airconditioningtypeid', 'architecturalstyletypeid', 'assessmentyear',
       'basementsqft', 'bedroomcnt', 'buildingclasstypeid',
       'buildingqualitytypeid', 'calculatedfinishedsquarefeet',
       'censustractandblock', 'decktypeid', 'finishedfloor1squarefeet',
       'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',
       'finishedsquarefeet50', 'finishedsquarefeet6', 'fips', 'fireplacecnt',
       'fullbathcnt', 'garagecarcnt', 'garagetotalsqft',
       'heatingorsystemtypeid', 'landtaxvaluedollarcnt', 'latitude',
       'longitude', 'numberofstories', 'poolcnt', 'poolsizesum',
       'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'propertylandusetypeid',
       'regionidcity', 'regionidcounty', 'regionidneighborhood', 'regionidzip',
       'roomcnt', 'storytypeid', 'structuretaxvaluedollarcnt',
       'taxdelinquencyyear', 'taxvaluedollarcnt', 'threequarterbathnbr',
       'typeconstructiontypeid', 'unitcnt', 'yardbuildingsqft17',
       'yardbuildingsqft26', 'yearbuilt'] if z.endswith('cnt')] + ['taxdelinquencyyear','yearbuilt','longitude','latitude']

      # Categorical data
      self.CAT = pd.Index(['airconditioningtypeid', 'architecturalstyletypeid', 'assessmentyear',
       'basementsqft', 'bedroomcnt', 'buildingclasstypeid',
       'buildingqualitytypeid', 'calculatedfinishedsquarefeet',
       'censustractandblock', 'decktypeid', 'finishedfloor1squarefeet',
       'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',
       'finishedsquarefeet50', 'finishedsquarefeet6', 'fips', 'fireplacecnt',
       'fullbathcnt', 'garagecarcnt', 'garagetotalsqft',
       'heatingorsystemtypeid', 'landtaxvaluedollarcnt', 'latitude',
       'longitude', 'numberofstories', 'poolcnt', 'poolsizesum',
       'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'propertylandusetypeid',
       'regionidcity', 'regionidcounty', 'regionidneighborhood', 'regionidzip',
       'roomcnt', 'storytypeid', 'structuretaxvaluedollarcnt',
       'taxdelinquencyyear', 'taxvaluedollarcnt', 'threequarterbathnbr',
       'typeconstructiontypeid', 'unitcnt', 'yardbuildingsqft17',
       'yardbuildingsqft26', 'yearbuilt']).difference(NUM)

      self.DTYPES = dict(INT + FLT + NINT + OBJ)

    def _to_cartesian(self, lon, lat):
      """ Feature Engineering for coordinates : Transform longitude and latitude to 3D cartesian

      :lon: float, longitude
      :lat: float, latitude
      """
      R = 6371
      x = R * np.cos(lat) * np.cos(lon)
      y = R * np.cos(lat) * np.sin(lon)
      z = R * np.sin(lat)
      XYZ = pd.DataFrame(np.hstack ([x.reshape((-1,1)),y.reshape((-1,1)),z.reshape((-1,1)) ]))
      XYZ.columns = ['x','y','z']
      return  XYZ

    def _read_data(self):
      print('\n reading data ... \n')
      data = pd.read_csv(self.path + '/' + self.prop_filename, dtype=self.DTYPES)
      return data

    def _preprocess(self, prop):
      """ Preprocessing

      :prop: pandas DataFrame
      """
      print('\n preprocessing ...\n')
      np.random.seed(self.seed)

      prop['rawcensustractandblock_L'] = prop['rawcensustractandblock'].apply(lambda x: str(x).split('.')[0] if(pd.notnull(x)) else x).astype('category').cat.codes
      prop[self.coords] /= 1.e6
      prop[self.coords] = prop[self.coords].fillna(-1) # store missing as -1
      _tmp = self._to_cartesian(prop['longitude'], prop['latitude'])
      prop = pd.concat([prop, _tmp], axis = 1)
      prop[self.CAT] = prop[self.CAT].apply(lambda x: x.astype('category',ordered=True).cat.codes,axis=0)
      prop_codes=['propertycountylandusecode', 'propertylandusetypeid','propertyzoningdesc']
      prop[prop_codes] = prop[prop_codes].apply( lambda x: x.astype('category',ordered=True).cat.codes,axis=0)

      for c in prop.dtypes[prop.dtypes == object].index.values:
        prop[c] = 1*((prop[c] == True) | (prop[c]=='Y'))

      regions = [z for z in prop.columns if z.startswith('region')]

      prop[regions] = prop[regions].apply(lambda x: x.astype('category',ordered=True).cat.codes)
      # redundant : meant to ensure everything remains 32 bit for memory conservation
      prop[prop.columns.difference(['parcelid'])]=prop[prop.columns.difference(['parcelid'])].apply(lambda x: x.astype(np.float32) if(x.dtype==np.float64) else x)
      prop[prop.columns.difference(['parcelid'])]=prop[prop.columns.difference(['parcelid'])].apply(lambda x: x.astype(np.int32) if(x.dtype==np.int64) else x)

      print('\n constructing new features : missing, ratios, value ...\n')
      pf = prop.columns[1:-3] # base property feature names
      # treat missing patterns as a fingerprint, construct categorical for patterns
      prop['missing_pattern'] = pd.Series((1*prop[pf].isnull()).values.tolist(),name='missing_pattern').apply(lambda x: tuple(x)).astype('category').cat.codes.values

      # ratio_livelot attempts to create a feature to adjust for homebuyers that do (or do not) consider the amount of land vs homesize
      prop['ratio_livelot'] = prop['calculatedfinishedsquarefeet'] / prop['lotsizesquarefeet']

      # ratio_strlnd seeks tax feature tp capture how valuable structure is vs land
      prop['ratio_strlnd']  = prop['structuretaxvaluedollarcnt'] / prop['landtaxvaluedollarcnt']

      # given the ratio of home to lot, we try to capture the value of the open land.
      prop['value_openlnd'] = (1. - prop['ratio_livelot']) * prop['landtaxvaluedollarcnt']

      xyz = prop[['x','y','z']].copy()
      print('\n constructing neighbor features ...\n')
      # define neighbor features to construct, consider building a model to assess risk for tax delinquency
      nbr_features = ['bathroomcnt', 'bedroomcnt', 'fireplacecnt', 'fullbathcnt',
       'lotsizesquarefeet', 'garagecarcnt', 'poolcnt', 'roomcnt', 'unitcnt',
       'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
       'landtaxvaluedollarcnt', 'taxamount', 'taxdelinquencyflag']

      Z = pd.DataFrame(np.zeros(prop[nbr_features].shape))

      nbr_k = 24 # found via tuning kfold
      tree = KDTree(xyz.values)
      dist, ind = tree.query(xyz.values,k=nbr_k+1) # k+1 since their is self connection

      # construct Weights from distance for average weighting of neighbor features
      # we want the closest neighbors to have more weight in this pattern
      P = dist[:,1:].copy()
      P[P==0] = 1
      invP = 1/P
      invP /= invP.sum(1)[:,np.newaxis]
      for i in range(1,nbr_k+1):
        Z += prop[nbr_features].fillna(-1).iloc[ind[:,i]].values * invP[:,[i-1]]

      Z.columns = nbr_features
      Z.columns = [z +'_nbrmean_lyr1' for z in Z.columns]

      prop = pd.concat([prop,Z],axis=1)

      # future intention for mean distances,
      #   We consider that some homebuyers might place emphasis on where they live vs where they work
      #   Additionally, some may want to be close to the nightlife vs far from (families, etc.)
      #   For round2 might consider distances to schools, and school ratings amongst other potential features

      print('\n constructing distance based features ... \n')
      # consider mean distance to center of county
      selected_regionid = 'regionidcounty'
      ridc_getdist = prop[['x','y','z']].groupby(prop[selected_regionid]).mean()

      # consider mean distance to center of neighborhood
      selected_regionid = 'regionidneighborhood'
      ridnbr_getdist = prop[['x','y','z']].groupby(prop[selected_regionid]).mean()

      # consider mean distance to center of city
      selected_regionid = 'regionidcity'
      ridcty_getdist = prop[['x','y','z']].groupby(prop[selected_regionid]).mean()

      tree = KDTree(ridc_getdist.values)
      dist, ind = tree.query(prop[['x','y','z']].values,k=4)

      a3 = pd.DataFrame(dist)
      a3.columns = ['dist_com_cnty_{}'.format(i) for i in range(1,5)]

      tree = KDTree(ridnbr_getdist.values)
      dist, ind = tree.query(prop[['x','y','z']].values,k=5)

      a2 = pd.DataFrame(dist)
      a2.columns = ['dist_com_nbr_{}'.format(i) for i in range(1,6)]

      tree = KDTree(ridcty_getdist.values)
      dist, ind = tree.query(prop[['x','y','z']].values,k=5)

      a1 = pd.DataFrame(dist)
      a1.columns = ['dist_com_cty_{}'.format(i) for i in range(1,6)]

      prop = pd.concat([prop,a1,a2, a3],axis=1)
      # return bookmarked missing to NaN
      # most models in consideration manage missing internally
      prop = prop.replace(-999,np.nan).replace(-1,np.nan)

      # again, redundancy, ensure all floats in 64bit are converted to 32bit
      # this is more important if we return as opposed to save a property file
      _floats = prop.columns[prop.dtypes==np.float64]
      prop[_floats] = prop[_floats].apply(lambda x: x.astype(np.float32))
      del tree, Z, a1, a2, a3
      gc.collect()
      return prop

    def _run(self):
      self._construct_datatypes()
      prop = self._read_data()
      prop = self._preprocess(prop)
      if self._save_to_file:
        print('\n saving datafile {} \n'.format(self.output_filename))
        prop.to_csv(self.path + '/' + self.output_filename,compression='gzip')
        del prop
        gc.collect()
      else:
        return prop

    def _load(self):
      _tmp = pd.read_csv(self.path + '/' + self.output_filename, nrows=2)
      data_types = _tmp.dtypes.to_dict()
      data_types = {k:(np.int32 if v == np.int64 else np.float32) for k, v in data_types.items()}
      data = pd.read_csv(self.path + '/' + self.output_filename, dtype=data_types)
      return data

if __name__ == '__main__':
  DSL = DatasetLoader(prop_filename = 'properties_2017.csv.zip',
        output_filename = 'processed_prop_test.csv.gz', path = '../input',
        seed = 5123, _save_to_file = True)

  DSL._run()


