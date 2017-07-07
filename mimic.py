import utils
import pandas as pd
from constants import ALL,column_names,NO_UNITS,START_DT,END_DT
import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from fuzzywuzzy import fuzz
import re
import random
import units
import transformers
import dask.dataframe as dd
from extract_transform_load import ETLManager

ITEMID = 'itemid'
SUBINDEX = 'subindex'
HADM_ID = 'hadm_id'

UOM_MAP = {
    '#': 'number',
     '%': 'percent',
     '(CU MM|mm3)': 'mm**3',
     '24(hr|hour)s?': 'day',
     'Deg\\.? F': 'degF',
     'Deg\\.? C': 'degC',
     'I.U.': 'IU',
     'MEQ': 'mEq',
     'MM HG': 'mmHg',
     '\\+': 'pos',
     '\\-': 'neg',
     'cmH20': 'cmH2O',
     'gms': 'grams',
     'kg x m ': 'kg*m',
     'lpm': 'liter/min',
     'm2': 'm**2',
     'mcghr': 'mcg/hr',
     'mcgkghr': 'mcg/kg/hr',
     'mcgkgmin': 'mcg/kg/min',
     'mcgmin': 'mcg/min',
     'mghr': 'mg/hr',
     'mgkghr': 'mg/kg/hr',
     'mgmin': 'mg/min',
     '\?F':'degF',
     '\?C':'degC',
     'Uhr':'U/hr',
     'Umin':'U/min'
    }
"""
EXPLORING MIMIC-III database
"""

class explorer(object):

    def __init__(self, mimic_conn):

        columns_to_keep = ['component','abbreviation','itemid','linksto','category','unitname']
        self.df_all_defs = item_defs(mimic_conn)[columns_to_keep]
        self.df_all_defs.set_index('itemid',inplace=True, drop=True)

    def search(self,terms,loinc_code=None):

        results = None
        for term in terms:
            score = self.df_all_defs[['category','component','abbreviation','unitname']].applymap(lambda x: fuzzy_score(str(x),term)).max(axis=1)
            if results is None: results = score
            else: results = pd.concat([results, score], axis=1).max(axis=1)

        results.name = 'score'
        return self.df_all_defs.join(results.to_frame()).sort_values('score',ascending=False)


def fuzzy_score(x,y):
    if len(x)==0 or len(y) == 0: return 0
    x = x.lower()
    y = y.lower()
    score = pd.np.mean([
                fuzz.partial_ratio(x,y),
                fuzz.token_sort_ratio(x,y),
                fuzz.ratio(x,y)
                ])
    bonus = 10 if (x in y) or (y in x) else 0
    return score + bonus


"""
EXTRACTING Data from MIMIC-III
- Timeseries data
- Context/demographic data
"""

class MimicETLManager(ETLManager):

    def __init__(self,data_dict,cleaners,hdf5_fname,mimic_item_map_fname):
        self.conn = connect()
        self.item_map = pd.read_csv(mimic_item_map_fname)
        cleaners = standard_cleaners(data_dict)
        super(MimicETLManager,self).__init__(data_dict,cleaners,hdf5_fname)

    def extract(self,component):
        return extract_component(self.conn,component,self.item_map,self.data_dict)

    def transform(self,df,component):
        transformers = transform_pipeline(first_component=component)
        return transformers.fit_transform(df)

    def extracted_ids(self,df_extracted):
        return df_extracted[column_names.ID].unique().tolist()

    def extracted_data_count(self,df_extracted):
        return df_extracted[column_names.VALUE].count()


def extract_component(mimic_conn,component,item_map,data_dict,hadm_ids=ALL):
    itemids = items_for_components(item_map,[component])
    if len(itemids) == 0: return None
    #Get item defs and filter to what we want
    df_item_defs = item_defs(mimic_conn)
    df_item_defs = df_item_defs[df_item_defs.itemid.isin(itemids)]
    df_item_defs = df_item_defs[~(df_item_defs.linksto == '')]
    grouped = df_item_defs.groupby('linksto')

    df_list = []
    df_columns = column_map()

    too_many_ids = len(hadm_ids) > 2000

    for table,group in grouped:
        itemids = group.itemid.astype(int).tolist()
        logger.log('Extracting {} items from {}'.format(len(itemids),table))
        is_iemv = table == 'inputevents_mv'
        df_col = df_columns.columns.tolist() + (['statusdescription'] if is_iemv else [])
        for ix,column_set in df_columns.loc[[table]].iterrows():
            psql_col = column_set.tolist() + (['statusdescription'] if is_iemv else [])
            query = 'SELECT {} FROM mimiciii.{} WHERE itemid = ANY (ARRAY{})'.format(','.join(psql_col),table,itemids)
            if not (hadm_ids == ALL) and not too_many_ids:
                query += ' AND hadm_id = ANY (ARRAY{})'.format(hadm_ids)
            df = pd.read_sql_query(query,mimic_conn)
            df.columns = df_col
            if too_many_ids:
                df = df[df[column_names.ID].isin(hadm_ids)]
            if is_iemv:
                df = df.loc[df['statusdescription'].astype(str) != 'Rewritten']
                df.drop('statusdescription', axis=1,inplace=True)
            df_list.append(df)



    logger.log('Combine DF')
    df_all = pd.concat(df_list)
    logger.log('Clean UOM')
    df_all = clean_uom(df_all,component,data_dict)

    return df_all

def clean_uom(df,component,data_dict):
    grouped = df.groupby(column_names.UNITS)
    for old_uom,group in grouped:
        new_uom = process_uom(old_uom,component,data_dict)
        df.loc[group.index,column_names.UNITS] = new_uom
        if not (old_uom == new_uom):
            df.loc[group.index,ITEMID] = utils.append_to_description(df.loc[group.index,ITEMID].astype(str),old_uom)
    return df

def process_uom(units,component,data_dict):

    if units in ['BPM','bpm']:
        if component == data_dict.components.HEART_RATE: units = 'beats/min'
        if component == data_dict.components.RESPIRATORY_RATE: units = 'breaths/min'
    for to_replace,replacement in UOM_MAP.iteritems():
        units = re.sub(to_replace, replacement,units,flags=re.IGNORECASE)
    return units

def get_context_data(hadm_ids=ALL,mimic_conn=None):

        if mimic_conn is None:
            mimic_conn = connect()
        #get HADM info (includes patient demographics)
        df_hadm = hadm_data(mimic_conn,hadm_ids)

        #get icu data
        df_icu = icu_data(mimic_conn,hadm_ids)

        #merge into single dataframe
        df_hadm_info = df_hadm.merge(df_icu,on=HADM_ID,how='left')

        df_hadm_info.rename(columns={HADM_ID : column_names.ID},inplace=True)

        return df_hadm_info



def hadm_data(mimic_conn,hadm_ids):
    """
    expects a TUPLE of hadm_ids
    """


    """
    @@@@@@@@@@@@
    1. Get all demographic data from the ADMISSIONS table = df_hadm
       https://mimic.physionet.org/mimictables/admissions/

       SELECT subject_id, hadm_id, admittime, dischtime, language, religion,
           marital_status, ethnicity, diagnosis, admission_location
       FROM admissions
       WHERE hadm_id IN hadm_ids
    @@@@@@@@@@@@
    """
    table = 'mimiciii.admissions'
    hadm_where_case = None if hadm_ids == ALL else 'hadm_id IN {}'.format(tuple(hadm_ids))
    col_psql = ['subject_id', HADM_ID, 'admittime', 'dischtime', 'language',
                        'religion','marital_status', 'ethnicity', 'diagnosis','admission_location']
    col_df = ['pt_id',HADM_ID,START_DT,END_DT,'lang',
                        'religion','marital_status','ethnicity','dx_info','admission_location']
    df_hadm = context_extraction_helper(mimic_conn,table,col_psql,col_df,hadm_where_case)

    """
    @@@@@@@@@@@@
    2. Get all demographic data from PATIENTS table = df_pt
       https://mimic.physionet.org/mimictables/patients/

       SELECT gender, dob, dod
       FROM patients
       WHERE subject_id IN pt_ids
    @@@@@@@@@@@@
    """

    table = 'mimiciii.patients'
    pt_ids = df_hadm['pt_id'].unique().tolist()
    col_psql = ['subject_id','gender','dob','dod']
    col_df = ['pt_id','gender','dob','dod']
    df_pt = context_extraction_helper(mimic_conn,table,col_psql,col_df)

    """

    @@@@@@@@@@@@
    3. Get all ICD codes data from DIAGNOSES_ICD table = df_icd
       https://mimic.physionet.org/mimictables/diagnoses_icd/

       SELECT subject_id, hadm_id, seq_num, icd9_code
       FROM diagnoses_icd
       WHERE hadm_id IN hadm_ids
    @@@@@@@@@@@@
    """
    table = 'mimiciii.diagnoses_icd'
    col_psql = ['subject_id',HADM_ID,'seq_num','icd9_code']
    col_df = ['pt_id',HADM_ID,'icd_rank','icd_code']
    df_icd = context_extraction_helper(mimic_conn,table,col_psql,col_df,hadm_where_case)

    """
    @@@@@@@@@@@@
    4. Make df_icd into single rows for each admission, where one
     column is an ordered list of ICD codes for that admission
    @@@@@@@@@@@@
    """
    df_icd = df_icd.sort_values('icd_rank').groupby(HADM_ID).apply(lambda grp: grp['icd_code'].tolist())
    df_icd.name = 'icd_codes'
    df_icd = df_icd.reset_index()

    """
    @@@@@@@@@@@@
    Merging
    5. Merge df_pt and df_hadm on subject_id = demographics_df
    6. Merge demographics_df with df_icd on hadm_id = df_hadm_info
    @@@@@@@@@@@@
    """
    df_demographics = df_hadm.merge(df_pt,on='pt_id',how='left')
    df_hadm_info = df_demographics.merge(df_icd,on=HADM_ID,how='left')

    """
    @@@@@@@@@@@@
    Cleaning
    7. Remove all NA hadm_ids
    8. Add age column
    9. cast hadm_id to int
    @@@@@@@@@@@@
    """
    df_hadm_info = df_hadm_info.dropna(subset=[HADM_ID])
    df_hadm_info['age'] = df_hadm_info['start_dt']-df_hadm_info['dob']
    df_hadm_info[HADM_ID] = df_hadm_info[HADM_ID].astype(int)

    return df_hadm_info

def icu_data(mimic_conn,hadm_ids):

    table = 'mimiciii.icustays'
    col_psql = [HADM_ID,'icustay_id','dbsource','first_careunit','last_careunit','intime','outtime','los']
    col_df = [HADM_ID,'icustay_id','dbsource','first_icu','last_icu','intime','outtime','los']
    hadm_where_case = None if hadm_ids == ALL else 'hadm_id IN {}'.format(tuple(hadm_ids))
    df_icu = context_extraction_helper(mimic_conn,table,col_psql,col_df,hadm_where_case)


    """
    Cleaning
    - drop ICUSTAYS without hadm_id
    """
    df_icu = df_icu.dropna(subset=[HADM_ID])

    return df_icu

def context_extraction_helper(mimic_conn,table,col_psql,col_df,where_case=None):
    query = utils.simple_sql_query(table,col_psql,where_case)
    df = pd.read_sql_query(query,mimic_conn)
    rename_dict = dict(zip(col_psql,col_df))
    df.rename(index=str,columns=rename_dict,inplace=True)
    return df


def column_map():
    """
    Create column mapping
    """
    #definitions

    std_columns = [column_names.ID,column_names.DATETIME,column_names.VALUE,column_names.UNITS,'itemid']
    psql_col = ['hadm_id','charttime','value','valueuom','itemid']

    col_series = pd.Series(
        psql_col,
        index=std_columns
        )

    map_list = []
    col_series.name = 'chartevents'
    map_list.append(col_series)

    col_series = col_series.copy()
    col_series.name = 'labevents'
    map_list.append(col_series)

    col_series = col_series.copy()
    col_series.name = 'procedureevents_mv'
    map_list.append(col_series)

    col_series = col_series.copy()
    col_series.name = 'datetimeevents'
    map_list.append(col_series)

    col_series = col_series.copy()
    col_series.name = 'outputevents'
    map_list.append(col_series)

    psql_col = ['hadm_id','starttime','rate','rateuom','itemid']
    col_series = pd.Series(
        psql_col,
        index=std_columns
        )
    col_series.name = 'inputevents_mv'
    map_list.append(col_series)

    psql_col = ['hadm_id','endtime','amount','amountuom','itemid']
    col_series = pd.Series(
        psql_col,
        index=std_columns
        )
    col_series.name = 'inputevents_mv'
    map_list.append(col_series)

    psql_col = ['hadm_id','charttime','rate','rateuom','itemid']
    col_series = pd.Series(
        psql_col,
        index=std_columns
        )
    col_series.name = 'inputevents_cv'
    map_list.append(col_series)

    psql_col = ['hadm_id','charttime','amount','amountuom','itemid']
    col_series = pd.Series(
        psql_col,
        index=std_columns
        )
    col_series.name = 'inputevents_cv'
    map_list.append(col_series)


    return pd.DataFrame(map_list)

def item_defs(mimic_conn):

    df_items = pd.read_sql_query('SELECT * FROM mimiciii.d_items',mimic_conn)
    df_labitems = pd.read_sql_query('SELECT * FROM mimiciii.d_labitems',mimic_conn)
    df_labitems['linksto'] = 'labevents'

    df_all_items = pd.concat([df_labitems,df_items])
    return df_all_items

def items_for_components(item_map,components=ALL):
    if not (components == ALL):
        item_map = item_map[item_map.component.isin(components)]
    items = item_map.itemid.unique().astype(int).tolist()
    return items

def get_all_hadm_ids():
    conn = connect()
    all_ids = pd.read_sql_query('SELECT hadm_id from mimiciii.admissions',conn)['hadm_id']
    all_ids = all_ids[~pd.isnull(all_ids)]
    return all_ids.astype(int).sort_values().tolist()

def sample_hadm_ids(n,seed):
    all_ids = get_all_hadm_ids()
    random.seed(seed)
    sampled_ids = random.sample(all_ids,n)
    return sampled_ids

def connect(psql_username='postgres',psql_pass='123'):
    return utils.psql_connect(psql_username,psql_pass,'mimic')


"""
TRANSFORM Data extracted from MIMIC-III
"""
class clean_extract(BaseEstimator,TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        """
        FORMAT pre-unstack columns
        """
        df = df.replace(to_replace='', value=pd.np.nan)
        #drop NAN record_id, timestamps, or value
        df.dropna(subset=[column_names.ID,column_names.DATETIME,column_names.VALUE], how='any',inplace=True)

        #ID to integer
        df.loc[:,column_names.ID] = df.loc[:,column_names.ID].astype(int)

        #DATETIME to pd.DATETIME
        df.loc[:,column_names.DATETIME] = pd.to_datetime(df.loc[:,column_names.DATETIME],errors='raise')

        #set UOM to NO_UOM if not declared
        df.loc[:,column_names.UNITS] = df.loc[:,column_names.UNITS].fillna(NO_UNITS)

        df.rename(index=str,columns={ITEMID:column_names.DESCRIPTION},inplace=True)
        index_cols = [
                    column_names.ID,
                    column_names.DATETIME,
                    column_names.DESCRIPTION,
                    column_names.UNITS
                ]
        #Set up our row index
        df.set_index(index_cols,inplace=True)

        return df



class unstacker(transformers.safe_unstacker):

    def __init__(self):
        super(unstacker,self).__init__(column_names.UNITS,column_names.DESCRIPTION)

def transform_pipeline(first_component=None):
    return Pipeline([
    ('clean',clean_extract()),
    ('unstack',unstacker()),
    ('add_level',transformers.add_level(first_component,'component',axis=1)),
])

def standard_cleaners(data_dict):
    category_map = mimic_category_map(data_dict)
    ureg = units.MedicalUreg()
    return Pipeline([
        ('aggregate_same_datetime',transformers.same_index_aggregator(lambda grp:grp.iloc[0])),
        ('split_dtype',transformers.split_dtype()),
        ('standardize_columns',transformers.column_standardizer(data_dict,ureg)),
        ('standardize_categories',transformers.standardize_categories(data_dict,category_map)),
        ('split_bad_categories',transformers.split_bad_categories(data_dict)),
        # ('one_hotter',transformers.nominal_to_onehot()),
        ('drop_oob_values',transformers.oob_value_remover(data_dict))
    ])


def mimic_category_map(data_dict):
    return {
        data_dict.components.GLASGOW_COMA_SCALE_EYE_OPENING: {
            '1 No Response': 6,
            '2 To pain': 7,
            '3 To speech': 8,
            '4 Spontaneously': 9
        },
        data_dict.components.GLASGOW_COMA_SCALE_MOTOR: {
            '1 No Response': 0,
            '2 Abnorm extensn': 1,
            '3 Abnorm flexion': 2,
            '4 Flex-withdraws': 3,
            '5 Localizes Pain': 4,
            '6 Obeys Commands': 5
        },
        data_dict.components.GLASGOW_COMA_SCALE_VERBAL: {
            '1 No Response': 10,
            '1.0 ET/Trach': 10,
            '2 Incomp sounds': 11,
            '3 Inapprop words': 12,
            '4 Confused': 13,
            '5 Oriented':14
        }
    }





def ETL(extractor,
        components,
        data_dict,
        same_dt_aggregator,
        hdf5_fname=None,joined_path=None,
        hadm_ids=ALL,
        use_base_df=True,
        to_pandas=False,
        chunksize=500000):

    logger.log('***ETL***',new_level=True)
    logger.log('SETUP',new_level=True)

    category_map = mimic_category_map(data_dict)
    ureg = units.MedicalUreg()

    transformer = transform_pipeline()

    standard_clean_pipeline = Pipeline([
        ('aggregate_same_datetime',same_dt_aggregator),
        ('split_dtype',transformers.split_dtype()),
        ('standardize_columns',transformers.column_standardizer(data_dict,ureg)),
        ('standardize_categories',transformers.standardize_categories(data_dict,category_map)),
        ('split_bad_categories',transformers.split_bad_categories(data_dict)),
        # ('one_hotter',transformers.nominal_to_onehot()),
        ('drop_oob_values',transformers.oob_value_remover(data_dict))
    ])

    should_save = (hdf5_fname is not None)

    df_base = None

    if should_save & use_base_df:
        try:
            df_base = utils.open_df(hdf5_fname,joined_path)
        except:
            pass

    if df_base is not None:


        existing_components = df_base.columns.get_level_values(column_names.COMPONENT).unique().tolist()
        existing_ids = set(df_base.index.get_level_values(column_names.ID).tolist())
        requested_ids = hadm_ids if hadm_ids != ALL else get_all_hadm_ids()

        new_ids = [ID for ID in requested_ids if ID not in existing_ids]


        #case 1: new ids in existing columns, don't try to be smart with ALL unless not a lot of IDs
        if len(new_ids) > 0:
            df_addition = ETL(extractor,
                                existing_components,
                                data_dict,
                                same_dt_aggregator,
                                hadm_ids=new_ids,
                                to_pandas=True)
            if df_addition is not None:
                df_base = pd.concat([df_base,df_addition])
            #now we only need to load NEW components
            components = [comp for comp in components if comp not in existing_components]

        logger.log('Base DF to Dask')
        df_base = dd.from_pandas(df_base.reset_index(), chunksize=chunksize)


    df_all = df_base

    logger.log('BEGIN ETL for {} admissions and {} components: {}'.format(hadm_ids if hadm_ids == ALL else len(hadm_ids),
                                                                            len(components),
                                                                            components),new_level=True,end_level=True)
    for component in components:
        logger.log('{}: {}/{}'.format(component.upper(),components.index(component)+1,len(components)),new_level=True)

        """
        @@@@@@@@@@@@@@@
        ----EXTRACT----
        @@@@@@@@@@@@@@@
        """

        logger.log("Extracting...",new_level=True)
        df_extracted = extractor.extract_component(component,hadm_ids)

        if df_extracted.empty:
            print 'EMPTY Dataframe EXTRACTED for {}, n={} ids'.format(component,len(hadm_ids))
            logger.end_log_level()
            continue

        if should_save:
            logger.log('Save EXTRACTED DF = {}'.format(df_extracted.shape))
            utils.save_df(df_extracted,hdf5_fname,'extracted/{}'.format(component))
        logger.end_log_level()


        """
        @@@@@@@@@@@@@@@@@
        ----TRANSFORM----
        @@@@@@@@@@@@@@@@@
        """

        logger.log("Transforming... {}".format(df_extracted.shape),new_level=True)
        transformer.set_params(add_level__level_val=component)
        df_transformed = transformer.transform(df_extracted)

        print 'Data Loss (Extract > Transformed):',utils.data_loss(df_extracted.set_index(column_names.ID).value.to_frame(),df_transformed)

        if df_transformed.empty:
            print 'EMPTY Dataframe TRANSFORMED for {}, n={} ids'.format(component,len(hadm_ids))
            logger.end_log_level()
            continue

        if should_save:
            logger.log('Save TRANSFORMED DF = {}'.format(df_transformed.shape))
            utils.save_df(df_transformed,hdf5_fname,'transformed/{}'.format(component))
        logger.end_log_level()



        """
        @@@@@@@@@@@@@@@
        -----CLEAN-----
        @@@@@@@@@@@@@@@
        """

        logger.log("Cleaning... {}".format(df_transformed.shape),new_level=True)
        df = standard_clean_pipeline.transform(df_transformed)

        print 'Data Loss (Extract > Cleaned):', utils.data_loss(df_extracted.set_index(column_names.ID).value.to_frame(),df)

        if df.empty:
            print 'EMPTY Dataframe TRANSFORMED for {}, n={} ids'.format(component,len(hadm_ids))
            logger.end_log_level()
            continue

        if should_save:
            logger.log('Save CLEANED DF = {}'.format(df.shape))
            utils.save_df(df,hdf5_fname,'cleaned/{}'.format(component))
        logger.end_log_level()

        del df_extracted,df_transformed

        logger.log('Filter & sort - {}'.format(df.shape))

        df.sort_index(inplace=True)
        df.sort_index(inplace=True, axis=1)


        logger.log('Convert to dask - {}'.format(df.shape))
        df_dask = dd.from_pandas(df.reset_index(), chunksize=chunksize)
        del df

        logger.log('Join to big DF')

        if df_all is None: df_all = df_dask
        else :
            df_all = df_all.merge(df_dask,how='outer', on=['id','datetime'])
            del df_dask

        logger.end_log_level()
    logger.end_log_level()

    if df_all is None or not to_pandas:
        logger.end_log_level()
        return df_all

    logger.log('Dask DF back to pandas')
    df_pd = df_all.compute()
    del df_all
    df_pd.set_index(['id','datetime'], inplace=True)

    logger.log('SORT Joined DF')
    df_pd.sort_index(inplace=True)
    df_pd.sort_index(inplace=True, axis=1)

    if should_save:
        logger.log('SAVE Big DF')
        utils.save_df(df_pd,hdf5_fname,joined_path)
    logger.end_log_level()

    return df_pd
