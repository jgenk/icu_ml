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

def smart_count(col):
    var_type = col.name[2]
    if (var_type == variable_type.NOMINAL) and (col.dtype == pd.np.uint8):
        return col.sum()

    return col.count()


"""
Pytables/HDF5 I/O with axis deconstruction
"""

def read_and_reconstruct(hdf5_fname,path,where=None):
    # Get all paths for dataframes in store
    data_path,col_path = deconstucted_paths(path)

    data = pd.read_hdf(hdf5_fname,data_path,where=where)
    columns = pd.read_hdf(hdf5_fname,col_path)
    return reconstruct_df(data,columns)


def reconstruct_df(data,column_df):
    #reconstruct the columns from dataframe
    column_index = reconstruct_columns(column_df,data.columns)
    df = data.copy()
    df.columns = column_index
    return df

def reconstruct_columns(column_df,col_ix=None):
    column_df = column_df.drop('dtype',axis=1)
    if col_ix is None: col_ix = column_df.index
    col_arys = column_df.loc[col_ix].T.values
    levels = column_df.columns.tolist()

    return pd.MultiIndex.from_arrays(col_arys,names=levels)

def deconstruct_and_write(df,hdf5_fname,path,append=False):

    # Deconstruct the dataframe
    data,columns = deconstruct_df(df)

    # Get all paths for dataframes in store
    data_path,col_path = deconstucted_paths(path)

    #Open store and save df
    store = pd.HDFStore(hdf5_fname)
    store.put(data_path,data,append=append,format='t')
    if (not append) or col_path not in store:
        columns.to_hdf(hdf5_fname,col_path,format='t')
    store.close()
    return

def deconstruct_df(df):
    columns = pd.DataFrame(map(list,df.columns.tolist()),columns=df.columns.names)
    columns['dtype'] = df.dtypes.values.astype(str)
    data = df.copy()
    data.columns = [i for i in range(df.shape[1])]
    return data,columns

def deconstucted_paths(path):
    data_path = '{}/{}'.format(path,'data')
    col_path = '{}/{}'.format(path,'columns')
    return data_path,col_path

def complex_row_mask(df,specs,operator='or'):
    #init our mask datframe with row index of passed in dataframe
    df_mask = pd.DataFrame(index=df.index)

    #make sure our specs are a list
    if not isinstance(specs,list): specs = [specs]

    #if we have no specs, then we are not going to mask anything
    if (specs is None) or (len(specs) == 0):
        df_mask.loc[:,0] = True

    #apply specs to create mask
    for idx,spec in enumerate(specs):
        df_spec_mask = pd.DataFrame(index=df.index)
        for col_name,spec_info in spec.iteritems():
            if callable(spec_info):
                df_spec_mask.loc[:,col_name] = df.loc[:,col_name].apply(spec_info)
            else:
                if not isinstance(spec_info,list): spec_info = [spec_info]
                df_spec_mask.loc[:,col_name] = df.loc[:,col_name].isin(spec_info)
        df_mask.loc[:,idx] = df_spec_mask.all(axis=1)

    #if or, will will include rows from each spec.
    #   if and, only rows that meet criteria of EVERY spec ar included
    if operator == 'or' : mask = df_mask.any(axis=1)
    else: mask = df_mask.all(axis=1)
    return mask

def smart_join(hdf5_fname,paths,joined_path,ids,
                                        chunksize=5000,
                                        need_deconstruct=True,
                                        hdf5_fname_for_join=None,
                                        overwrite=True):

    logger.log('Smart join: n={}, {}'.format(len(ids),paths),new_level=True)

    if hdf5_fname_for_join is None: hdf5_fname_for_join=hdf5_fname

    store = pd.HDFStore(hdf5_fname_for_join)
    if (joined_path in store):
        if overwrite: del store[joined_path]
        else :
            store.close()
            logger.end_log_level()
            return hdf5_fname_for_join
    #sort ids, should speed up where clauses and selects
    ids = sorted(ids)

    #do chunked join
    logger.log('JOINING dataframes',new_level=True)
    for ix_start in range(0,len(ids),chunksize):
        ix_end = min(ix_start + chunksize,len(ids))
        id_slice = ids[ix_start:ix_end]

        where = '{id_col} in {id_list}'.format(id_col=column_names.ID,id_list=id_slice)

        logger.log('Slice & Join: {} --> {}, n={}'.format(id_slice[0], id_slice[-1],len(id_slice)),new_level=True)
        df_slice = None
        # for path in df_dict.keys():
        for path in paths:
            try:
                logger.log(path)
                if need_deconstruct: slice_to_add = read_and_reconstruct(hdf5_fname,path,where=where)
                else: slice_to_add = pd.read_hdf(hdf5_fname,path,where=where)
            except KeyError as err:
                logger.log(end_prev=True,start=False)
                print err
                continue

            if df_slice is None: df_slice = slice_to_add
            else:
                df_slice = df_slice.join(slice_to_add,how='outer')
                del slice_to_add

        logger.end_log_level()
        logger.log('Append slice')

        if need_deconstruct: deconstruct_and_write(df_slice,hdf5_fname_for_join,joined_path,append=True)
        else: df_slice.to_hdf(hdf5_fname_for_join,joined_path,append=True,format='t')

        del df_slice

    logger.end_log_level()
    logger.end_log_level()

    return hdf5_fname_for_join

def make_list_hash(l):
    #need to sort and make sure list are unique before hashing
    l = sorted(list(set(l)))

    #use a hash to make sure store this set of ids uniquely
    key = hash(''.join(map(str,l)))

    return key

"""
Dask intelligent join
"""

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


"""
Visualization
"""
import seaborn as sns
import matplotlib.pyplot as plt

def heatmap(df_ts):
    sns.set(context="paper", font="monospace")
    corrmat = df_ts.corr()
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(50, 50))
    # Draw the heatmap using seaborn
    sns.heatmap(corrmat, vmax=1, square=True)
    return
