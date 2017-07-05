import abc
import constants
import pandas as pd
import utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import logger
from transformers import filter_ids,DataNeedsFilter,do_nothing
from features import Featurizer
"""
Loading data
"""

class FilterBaseDF(TransformerMixin,BaseEstimator):

    def __init__(self,full_df,data_needs=constants.ALL):
        self.full_df=full_df
        self.data_needs = data_needs

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, ids):
        pipeline = Pipeline([
                ('row_filter',filter_ids(ids=ids)),
                ('dn_filter',DataNeedsFilter(self.data_needs))
        ])

        return pipeline.fit_transform(X=self.full_df, y=None)


class DFLoadAndFilter(FilterBaseDF):
    def __init__(self,hdf5_fname,path,data_needs=constants.ALL,load_at_init=False):
        self.hdf5_fname = hdf5_fname
        self.path = path
        self.load_at_init = load_at_init

        full_df = None
        if self.load_at_init:
            full_df = self.load_df(constants.ALL)

        super(DFLoadAndFilter,self).__init__(full_df,data_needs)

    def transform(self, ids):
        if not self.load_at_init:
            self.full_df = self.load_df(ids)
        return super(DFLoadAndFilter,self).transform(ids)

    def load_df(self,ids):
        return utils.open_df(self.hdf5_fname,self.path)


class ByComponentLoadAndFilter(DFLoadAndFilter):
    def __init__(self,hdf5_fname,path,data_needs,load_at_init=False,chunksize=500000):
        self.chunksize=chunksize
        super(ByComponentLoadAndFilter,self).__init__(hdf5_fname,path,data_needs,load_at_init)

    def load_df(self,ids):
        components = [dn[0] for dn in self.data_needs]
        return utils.dask_open_and_join(hdf5_fname=self.hdf5_fname,
                                    path=self.path,
                                    components=components,
                                    ids=ids,
                                    chunksize=self.chunksize)

class LoadAndSegment(TransformerMixin,BaseEstimator):
    def __init__(self,data_loader,segmenter):
        self.data_loader = data_loader
        self.segmenter=segmenter
        self.data_needs = data_loader.data_needs

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, y):
        if isinstance(y, pd.DataFrame):
            ids = y.index.get_level_values(constants.column_names.ID).unique().tolist()
        else: ids=y
        self.data_loader.data_needs = self.data_needs
        X = self.data_loader.fit_transform(ids)
        return self.segmenter.fit_transform(X=X, y=y)

"""
Features and Segments
"""

class SegmentFeaturizer(Featurizer):

    def __init__(self,loader,segmenter,
                    features=[],
                    pre_cleaners=do_nothing(),
                    post_cleaners=do_nothing()):
        post_cleaners = Pipeline([
            ('post_cleaner_arg',post_cleaners),
            ('drop_no_segments',DropNoSegments())
        ])
        super(SegmentFeaturizer, self).__init__(index_levels=[constants.column_names.ID,constants.SEG_ID],
                                                    loader=LoadAndSegment(loader,segmenter),
                                                    features=features,
                                                    pre_cleaners=pre_cleaners,
                                                    post_cleaners=post_cleaners)


"""
Segmenting
"""

class segmenter(BaseEstimator,TransformerMixin):
    __metaclass__ = abc.ABCMeta

    def __init__(self,end_first=False):
        self.end_first = end_first

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        logger.log('Segment df {}'.format(df.shape),new_level=True)

        logger.log('Get Segments')
        df_segments = self.__segment(df)
        logger.log('Apply n={} Segments to df.shape = {}'.format(df_segments.shape[0],df.shape))
        out_df = apply_segments(df,df_segments)
        logger.end_log_level()

        return out_df

    def __segment(self, df):


        if self.end_first:
            end_dt = self.__get_end_dt(df)
            start_dt = self.__get_start_dt(df,end_dt)
        else:
            start_dt = self.__get_start_dt(df)
            end_dt = self.__get_end_dt(df,start_dt)

        df_segments = self.__create_seg_df(start_dt,end_dt)
        return df_segments

    def __create_seg_df(self,start_dt,end_dt):
        return create_seg_df(start_dt,end_dt)

    @abc.abstractmethod
    def __get_start_dt(self,df_ts,end_dt=None):
        return

    @abc.abstractmethod
    def __get_end_dt(self,df_ts,start_dt=None):
        return

class static_end_date(segmenter):

    def __init__(self):
        super(static_end_date, self).__init__(end_first=True)
        return

    def fit(self, x, y=None, **fit_params):
        if y is None:
            self.end_dt = fit_params.get('end_dt',None)
        else:
            self.end_dt = y.reset_index(constants.column_names.DATETIME,drop=False).iloc[:,0]
        return self

    def _segmenter__get_end_dt(self,df_ts):
        return self.end_dt

class n_hrs_before(static_end_date):

    def __init__(self,n_hrs):
        super(n_hrs_before, self).__init__()
        self.__n_hrs = n_hrs
        return



    def _segmenter__get_start_dt(self,df_ts,end_dt):
        if self.__n_hrs == constants.ALL:
            return pd.Series([pd.NaT]*end_dt.size,index =end_dt.index)
        # n_before_dt = end_dt - pd.Timedelta(self.__n_hrs, unit='h')
        # first_obs_dt = utils.get_first_obs_dt(df_ts)
        # start_dt = first_obs_dt.to_frame().join(n_before_dt,how='left').apply(pd.np.max,axis=1)
        start_dt = end_dt - pd.Timedelta(self.__n_hrs, unit='h')
        return start_dt


class periodic(segmenter):

    def __init__(self,n_hrs,df_context=None):
        super(periodic, self).__init__()
        self.n_hrs = n_hrs
        self.df_context=df_context
        if self.df_context is not None:
            self.df_context = self.df_context.set_index(constants.column_names.ID)
        return

    def _segmenter__get_start_dt(self,df_ts,end_dt=None):
        grouped = df_ts.groupby(level=constants.column_names.ID)
        start_dt = grouped.apply(lambda x: self.__create_periods(x)).reset_index(level=1,drop=True)
        return start_dt.iloc[:,0]

    def _segmenter__get_end_dt(self,df_ts,start_dt):
        return start_dt + pd.Timedelta(self.n_hrs, unit='h')

    def __create_periods(self,seg):
        ID = seg.iloc[0].name[0]
        start = seg.iloc[0].name[-1]
        end = seg.iloc[-1].name[-1]
        if self.df_context is not None:
            start = min(start,self.df_context.loc[[ID],constants.START_DT].iloc[0])
            end = max(end,self.df_context.loc[[ID],constants.END_DT].iloc[0])

        return pd.Series(pd.date_range(start=start, end=end, freq='{}H'.format(self.n_hrs))).to_frame()


class DropNoSegments(BaseEstimator,TransformerMixin):


        def fit(self, x, y=None):
            return self

        def transform(self, df):
            return df[df.index.get_level_values(constants.SEG_ID) != constants.NO_SEGMENT]

def create_seg_df(start_dt,end_dt):


    start_dt.name = constants.START_DT
    start_dt = start_dt.to_frame()
    start_dt[constants.SEG_ID] = start_dt.groupby(level=constants.column_names.ID).cumcount()
    start_dt = start_dt.set_index(constants.SEG_ID,append=True).iloc[:,0]


    end_dt.name = constants.END_DT
    end_dt = end_dt.to_frame()
    end_dt[constants.SEG_ID] = end_dt.groupby(level=constants.column_names.ID).cumcount()
    end_dt = end_dt.set_index(constants.SEG_ID,append=True).iloc[:,0]

    df_segments = start_dt.to_frame()
    df_segments[end_dt.name] = end_dt
    df_segments.sort_index(inplace=True)



    return df_segments

def apply_segments(df_ts,df_segments):

    idx = pd.IndexSlice
    seg_to_add = {}

    #make a copy because we are going to be directly modifying this dataframe
    df_segmented = df_ts.copy()
    #we set all seg_id to "no segment" for now
    df_segmented[constants.SEG_ID] = constants.NO_SEGMENT
    #Iterate across all segments for a given ID
    for ID,id_segs in df_segments.groupby(level=constants.column_names.ID):
        has_data = ID in df_segmented.index

        if has_data: id_slice = df_segmented.loc[ID,:]
        #check segments within that ID, only compare to datetimes from that admission
        for seg_id,row in id_segs.loc[ID].iterrows():
            start_dt = row[constants.START_DT]
            end_dt = row[constants.END_DT]

            if has_data:
                #Datettime needs to be at or after start_dt, before end_dt
                #   If start_dt or end_dt is NaN, this signifies all before or all after
                #   respectively. as a result, if start_dt is nan, then any dt is "after start" etc.
                after_start = pd.isnull(start_dt) | (id_slice.index >= start_dt)
                before_end = pd.isnull(end_dt) | (id_slice.index < end_dt)
                in_seg = after_start & before_end

                #get the dt that should be in this segement
                dt_in_seg =  id_slice.loc[in_seg].index.tolist()

                if len(dt_in_seg) > 0:
                    df_segmented.loc[idx[ID,dt_in_seg],constants.SEG_ID] = seg_id
                    continue

            in_seg_dt = start_dt
            if pd.isnull(in_seg_dt):
                in_seg_dt = end_dt - pd.Timedelta(value=1,unit='s')
            seg_to_add[(ID,in_seg_dt)] = seg_id


    # create df for the empty segments & concat with existing dataframe
    if len(seg_to_add) > 0:
        empty_seg_index = pd.MultiIndex.from_tuples(seg_to_add.keys(),names=df_ts.index.names)
        df_empty_seg = pd.DataFrame(columns=df_ts.columns,index=empty_seg_index)
        df_empty_seg.loc[:,constants.SEG_ID] = seg_to_add.values()
        df_segmented = pd.concat([df_segmented,df_empty_seg])

    #format output dataframe
    df_segmented.set_index(constants.SEG_ID,append=True,inplace=True)
    df_segmented = df_segmented.reorder_levels([constants.column_names.ID,constants.SEG_ID,constants.column_names.DATETIME])
    df_segmented.sort_index(inplace=True)


    return df_segmented
