import abc
import pandas as pd
import ast
from constants import column_names
import utils
import logger

class ETLManager(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self,data_dict,cleaners,hdf5_fname):
        self.data_dict = data_dict
        self.cleaners = cleaners
        self.hdf5_fname = hdf5_fname


    def etl(self,data_specs,panel_id=None,operator='and',save_steps=False,overwrite=False):

        components = self.data_dict.get_components(specs=data_specs,panel_id=panel_id,operator=operator)

        if not overwrite:
            components = self.get_unloaded_components(components)

        all_etl_info = []

        logger.log('BEGIN ETL for {} components: {}'.format(len(components),components),new_level=True)
        for component in components:
            logger.log('{}: {}/{}'.format(component.upper(),components.index(component)+1,len(components)),new_level=True)

            logger.log('Extract...',new_level=True)
            df_extracted = self.extract(component)
            logger.end_log_level()

            logger.log('Transform...',new_level=True)
            df_transformed = self.transform(df_extracted,component)
            logger.end_log_level()

            logger.log('Clean...',new_level=True)
            df = self.cleaners.fit_transform(df_transformed.copy())
            logger.end_log_level()

            logger.log('Save DataFrames...',new_level=True)
            if save_steps:
                logger.log('Save EXTRACTED DF: {}'.format(df_extracted.shape))
                df_extracted.to_hdf(self.hdf5_fname,'{}/{}'.format(component,'extracted'))

                logger.log('Save TRANSFORMED DF: {}'.format(df_extracted.shape))
                df_transformed.to_hdf(self.hdf5_fname,'{}/{}'.format(component,'transformed'))

            logger.log('Save FINAL DF: {}'.format(df_extracted.shape))
            deconstruct_and_write(df,self.hdf5_fname,component)
            logger.end_log_level()



            etl_info = self.get_etl_info(df_extracted,df_transformed,df)
            etl_info.name = component
            all_etl_info.append(etl_info)

            del df_extracted,df_transformed,df

            logger.end_log_level()



        logger.end_log_level()
        return pd.DataFrame(all_etl_info)


    def get_unloaded_components(self,components):
        store = pd.HDFStore(self.hdf5_fname)
        unloaded = [c for c in components if c not in store]
        store.close()
        return unloaded


    def open_df(self,component):
        #open dataframe, assume in root directory
        return read_and_reconstruct(self.hdf5_fname, component)

    def get_etl_info(self,df_extracted,df_transformed,df_cleaned):
        e_ids = self.extracted_ids(df_extracted)
        e_data_count = self.extracted_data_count(df_extracted)

        t_ids = df_transformed.index.get_level_values(column_names.ID).unique().tolist()
        t_data_count = df_transformed.apply(utils.smart_count).sum()

        c_ids = df_cleaned.index.get_level_values(column_names.ID).unique().tolist()
        c_data_count = df_cleaned.apply(utils.smart_count).sum()

        index = pd.MultiIndex.from_tuples([
            ('EXTRACTED','id_count'),
            ('EXTRACTED','data_count'),
            ('TRANSFORMED','id_count'),
            ('TRANSFORMED','data_count'),
            ('CLEANED','id_count'),
            ('CLEANED','data_count'),
        ], names=['stage','stat'])

        return pd.Series([
            len(e_ids),
            e_data_count,
            len(t_ids),
            t_data_count,
            len(c_ids),
            c_data_count
        ], index=index)




    @abc.abstractmethod
    def extract(self,componment):
        return

    @abc.abstractmethod
    def transform(self,df,component):
        return

    @abc.abstractmethod
    def extracted_ids(self,df_extracted):
        return

    @abc.abstractmethod
    def extracted_data_count(self,df_extracted):
        return

def read_and_reconstruct(hdf5_fname,component,path=None,where=[]):
    key = _make_key(component,path)
    data = pd.read_hdf(hdf5_fname,'{}/{}'.format(key,'data'))
    columns = pd.read_hdf(hdf5_fname,'{}/{}'.format(key,'columns'))
    return reconstruct_df(data,columns)


def reconstruct_df(data,columns):
    #reconstruct the columns from flattened tuples
    columns = columns.drop('dtype',axis=1)
    col_ix = data.columns
    col_arys = columns.loc[col_ix].T.values
    levels = columns.columns.tolist()

    col_index = pd.MultiIndex.from_arrays(col_arys,names=levels)
    df = data.copy()
    df.columns = col_index
    return df

def deconstruct_and_write(df,hdf5_fname,component,path=None):
    key = _make_key(component,path)
    data,columns = deconstruct_df(df)
    data.to_hdf(hdf5_fname,'{}/{}'.format(key,'data'),format='t')
    columns.to_hdf(hdf5_fname,'{}/{}'.format(key,'columns'),format='t')
    return

def deconstruct_df(df):
    columns = pd.DataFrame(map(list,df.columns.tolist()),columns=df.columns.names)
    columns['dtype'] = df.dtypes.values.astype(str)
    data = df.copy()
    data.columns = [i for i in range(df.shape[1])]
    return data,columns

def _make_key(tb_name,path):
    if path is not None: key = '{}/{}'.format(path,tb_name)
    else: key = tb_name
    return key
