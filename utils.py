import sqlalchemy
import pandas as pd
from constants import variable_type,column_names,ALL
import logger
import numpy as np
import dask.dataframe as dd


def psql_connect(user, password, database='postgres', host='localhost', port=5432):
    '''Returns a connection and a metadata object'''
    # We connect with the help of the PostgreSQL URL
    # postgresql://federer:grandestslam@localhost:5432/tennis
    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(user, password, host, port, database)

    # The return value of create_engine() is our connection object
    connection = sqlalchemy.create_engine(url, client_encoding='utf8')

    return connection

def simple_sql_query(table,columns=['*'],where_condition=None):
    where = ''
    if where_condition is not None:
        where = ' WHERE {}'.format(where_condition)
    return 'SELECT {} FROM {}{}'.format(','.join(columns),table,where)

def save_df(df, hdf5_fname, path):
    store = pd.HDFStore(hdf5_fname)
    store[path] = df
    store.close()
    return df

def open_df(hdf5_fname, path):
    store = pd.HDFStore(hdf5_fname)
    df = store[path]
    store.close()
    return df


def is_categorical(var_type):
    return var_type in [variable_type.ORDINAL, variable_type.NOMINAL]

class Bunch(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

def add_subindex(df,subindex_name):
    df = df.sort_index()
    index = df.index
    df[subindex_name] = range(0,len(index))
    duplicated = index.duplicated(keep=False)
    df.loc[duplicated,subindex_name] = df.loc[duplicated].groupby(level=index.names)[subindex_name].apply(lambda x:x - x.min())
    df.loc[~duplicated,subindex_name] = 0
    df.loc[:,subindex_name] = df.loc[:,subindex_name].astype(int)
    df.set_index(subindex_name, append=True,inplace=True)
    return df

def add_same_val_index_level(df,level_val,level_name,axis=0):
    return pd.concat([df], keys=[level_val], names=[level_name],axis=axis)

def set_level_to_same_val(index,level,value):
    return index.set_labels([0]*index.size,level=level).set_levels([value],level=level)

def data_loss(df_start,df_end):
    data_loss = df_start.count().sum() - df_end.count().sum()
    admissions_start = len(df_start.index.get_level_values(column_names.ID).unique().tolist())
    admisstions_end = len(df_end.index.get_level_values(column_names.ID).unique().tolist())

    admission_loss = admissions_start - admisstions_end
    percent_loss = str(round(float(admission_loss)/admissions_start * 100,4))+'% records'

    return df_start.shape,df_end.shape,data_loss,admission_loss,percent_loss


def get_components(df):
    return df.columns.get_level_values('component').unique().tolist()

def filter_columns(df,level,value):
    return df.loc[:,df.columns.get_level_values(level) == value]

def append_to_description(old_desc,addition):
    return old_desc + '(' + addition + ')'

def get_first_dt(df_ts,df_context):
    ids = df_ts.index.get_level_values(column_names.ID).unique()
    admit_dt = get_admit_dt(df_context,ids)

    first_obs_dt = get_first_obs_dt(df_ts)
    first_dt = first_obs_dt.to_frame().join(admit_dt,how='left').apply(pd.np.min,axis=1)
    first_dt.name = column_names.DATETIME
    return first_dt

def get_admit_dt(df_context,ids=ALL):
    if not ids == ALL:
        df_context_filtered = df_context[df_context[column_names.ID].isin(ids)]
    else: df_context_filtered = df_context
    admit_dt = df_context_filtered.loc[:,['id','start_dt']].drop_duplicates(subset=['id']).set_index('id')
    return admit_dt

def get_first_obs_dt(df_ts):
    first_obs_dt = df_ts.groupby(level=column_names.ID).apply(lambda x:x.iloc[0].name[-1])
    first_obs_dt.name = 'start_dt_obs'
    return first_obs_dt

def flatten_index(df,join_char='_',suffix=None,axis=0):
    if axis==0:
        new_vals = []
        for row_name in df.index:
            if type(row_name) is tuple:
                row = join_char.join(map(str,row_name))
            else:
                row = str(row_name)

            if suffix is not None: row = row + join_char + suffix

            new_vals.append(row)

        df.index = pd.Index(new_vals)
    elif axis==1:
        new_vals = []
        for col_name in df.columns:
            if type(col_name) is tuple:
                col = join_char.join(map(str,col_name))
            else:
                col = str(col_name)
            if suffix is not None: col = col + join_char + suffix

            new_vals.append(col)

        df.columns = pd.Index(new_vals)
    else: pass
    return df

def dask_open_and_join(hdf5_fname,path,components,ids=ALL,chunksize=500000):

    df_all=None
    logger.log('DASK OPEN & JOIN n={} components: {}'.format(len(components),components),new_level=True)
    for component in components:
        logger.log('{}: {}/{}'.format(component.upper(),components.index(component)+1,len(components)),new_level=True)

        df_comp = open_df(hdf5_fname,'{}/{}'.format(path,component))
        df_comp.sort_index(inplace=True)
        df_comp.sort_index(inplace=True, axis=1)

        if not ids == ALL:
            df_comp = df_comp[df_comp.index.get_level_values(column_names.ID).isin(ids)]

        logger.log('Convert to dask - {}'.format(df_comp.shape))
        df_dask = dd.from_pandas(df_comp.reset_index(), chunksize=chunksize)
        del df_comp

        logger.log('Join to big DF')

        if df_all is None: df_all = df_dask
        else :
            df_all = df_all.merge(df_dask,how='outer', on=['id','datetime'])
            del df_dask
        logger.end_log_level()

    logger.log('Dask DF back to pandas')
    df_pd = df_all.compute()
    del df_all
    df_pd.set_index(['id','datetime'], inplace=True)

    logger.log('SORT Joined DF')
    df_pd.sort_index(inplace=True)
    df_pd.sort_index(inplace=True, axis=1)
    logger.end_log_level()
    return df_pd


# def partition_and_join(left,right,partition_size,index_level_for_output=0):
#     logger.log('PARTITION AND JOIN',new_level=True)
#
#
#
#     right.sort_index(inplace=True)
#     right.sort_index(inplace=True,axis=1)
#
#     #determine number of partitions
#     logger.log('Divide DataFrame {} in partitions of size = {}'.format(right.shape,partition_size))
#     num_partitions = int(np.ceil(right.shape[0]/float(partition_size)))
#     partitions = np.array_split(right,num_partitions)
#
#     #Join each partition, one at a time
#     logger.log('Join {} partitions to Dataframe shape = {}...'.format(num_partitions,left.shape),new_level=True)
#
#     while len(partitions) > 0 :
#
#         df_partition = partitions.pop()
#
#         #get information for logging
#         start = df_partition.iloc[0].name[index_level_for_output]
#         end = df_partition.iloc[-1].name[index_level_for_output]
#         partition_num = int(num_partitions-(len(partitions)))
#         logger.log('Partition {}/{}: {}, ({} -> {})'.format(partition_num,num_partitions,df_partition.shape,start,end),new_level=True)
#
#         #Drop NaN columns, will speed up join
#         #logger.log('Drop NaN')
#         #df_partition.dropna(axis=1,how='all',inplace=True)
#
#         #Perform complex join to left (can have overlapping row and column indicies)
#         in_index = df_partition.index.isin(left.index)
#         in_column = df_partition.columns.isin(left.columns)
#
#         logger.log('UPDATE existing indicies and existing columns')
#         left.update(df_partition.loc[in_index,in_column])
#
#         logger.log('JOIN existing indicies with new columns')
#         left = left.join(df_partition.loc[in_index,~in_column], how='outer')
#
#         logger.log('CONCAT new indicies')
#         left = pd.concat([left,df_partition.loc[~in_index,:]])
#
#         left.sort_index(inplace=True)
#         left.sort_index(inplace=True,axis=1)
#
#         logger.end_log_level()
#
#         #cleanup
#         del df_partition
#
#     logger.end_log_level()
#     logger.end_log_level()
#     return left
