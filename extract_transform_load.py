import abc
import pandas as pd
import ast
from constants import column_names
import utils
import logger

class ETLManager(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self,cleaners,hdf5_fname):
        self.cleaners = cleaners
        self.hdf5_fname = hdf5_fname


    def etl(self,components,save_steps=False,overwrite=False):
        if not overwrite:
            components = self.get_unloaded_components(components)
        if len(components) == 0: return None
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

                logger.log('Save TRANSFORMED DF: {}'.format(df_transformed.shape))
                df_transformed.to_hdf(self.hdf5_fname,'{}/{}'.format(component,'transformed'))

            logger.log('Save FINAL DF: {}'.format(df.shape))
            utils.deconstruct_and_write(df,self.hdf5_fname,path=component)
            logger.end_log_level()



            etl_info = self.get_etl_info(component,df_extracted,df_transformed,df)
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

    def open_df(self,component,ids=None):
        #open dataframe, assume in root directory
        where = None
        if ids is not None:
            ids = sorted(list(set(ids)))
            where = '{} in {}'.format(column_names.ID,ids)
        return utils.read_and_reconstruct(self.hdf5_fname, path=component, where=where)

    def get_etl_info(self,component,df_extracted,df_transformed,df_cleaned):
        e_ids = self.extracted_ids(df_extracted)
        e_data_count = self.extracted_data_count(df_extracted)

        t_ids = df_transformed.index.get_level_values(column_names.ID).unique().tolist()
        t_data_count = df_transformed.apply(utils.smart_count).sum()

        c_ids = df_cleaned.index.get_level_values(column_names.ID).unique().tolist()
        c_data_count = df_cleaned.apply(utils.smart_count).sum()

        return pd.Series({
            column_names.COMPONENT  : component,
            'EXTRACTED_id_count'    : len(e_ids),
            'EXTRACTED_data_count'  : e_data_count,
            'TRANSFORMED_id_count'  : len(t_ids),
            'TRANSFORMED_data_count': t_data_count,
            'CLEANED_id_count'      : len(c_ids),
            'CLEANED_data_count'    : c_data_count
        })




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

    @abc.abstractmethod
    def all_ids(self):
        return
