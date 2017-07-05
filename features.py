from sklearn.base import TransformerMixin,BaseEstimator,clone
from sklearn.pipeline import Pipeline,FeatureUnion
from constants import column_names,SEG_ID,NO_SEGMENT,ALL,CUSTOM_FILTER,FEATURE_LEVEL
import numpy as np
import utils
import transformers
import logger
import pandas as pd

class HasDataNeeds(object):

    def get_data_needs(self):
        return


"""
Aggregation
"""

class aggregator(TransformerMixin,BaseEstimator):

    def __init__(self,levels,agg_func,name=None):
        self.levels = levels
        self.agg_func = agg_func
        self.name = name

    def transform(self, df, **transform_params):
        if df.empty:
            drop_levels = [level for level in df.index.names if level not in self.levels]
            df = df.reset_index(drop_levels,drop=True)
            return pd.DataFrame(index=df.index,columns=df.columns)
        return df.groupby(level=self.levels).agg(self.agg_func)

    def fit(self, X, y=None, **fit_params):
        return self


class sample_aggregator(aggregator):
    def __init__(self,levels,random_state=None):
        agg_func = lambda x: x.sample(n=1,random_state=random_state)
        super(sample_aggregator,self).__init__(levels,agg_func,'SAMPLE')

class preserve_datetime_sample(sample_aggregator):

    def transform(self, df, **transform_params):
        datetimes = df.reset_index(column_names.DATETIME).iloc[:,0]
        datetimes.name = column_names.DATETIME
        sample_datetime = super(preserve_datetime_sample,self).transform(df=datetimes.to_frame()).set_index(column_names.DATETIME,append=True)
        return df.loc[sample_datetime.index.tolist()]

# class segment_aggregator(aggregator):
#
#     def __init__(self,agg_func,name):
#         super(segment_aggregator,self).__init__([column_names.ID,SEG_ID],agg_func,name)
#
#     def transform(self, df, **transform_params):
#         df = df[df.index.get_level_values(SEG_ID) != NO_SEGMENT]
#         return super(segment_aggregator,self).transform(df,**transform_params)
#
#     def fit(self, X, y=None, **fit_params):
#         return self


class mean_aggregator(aggregator):

    def __init__(self,levels):
        super(mean_aggregator,self).__init__(levels,np.mean, 'MEAN')

class std_aggregator(aggregator):

    def __init__(self,levels):
        super(std_aggregator,self).__init__(levels,np.std, 'STD')

class count_aggregator(aggregator):

    def __init__(self,levels):
        super(count_aggregator,self).__init__(levels,lambda x: x.count(), 'COUNT')

class last_aggregator(aggregator):

    def __init__(self,levels):
        super(last_aggregator,self).__init__(levels,last, 'LAST')

def last(col):
    col = col.dropna()
    if col.size > 0: return col.iloc[-1]
    return np.nan



class sum_aggregator(Pipeline):

    def __init__(self,levels):
        super(sum_aggregator,self).__init__(levels,np.sum, 'SUM')

class Feature(Pipeline,HasDataNeeds):
    def __init__(self,name,data_needs,col_filter,aggregator,fillna_method=transformers.do_nothing()):
        self.name=name
        self.data_needs=data_needs
        self.col_filter=col_filter
        self.aggregator=aggregator
        self.fillna_method=fillna_method
        super(Feature,self).__init__([
                                ('filter',self.col_filter),
                                ('agg_func',self.aggregator),
                                ('fillna',self.fillna_method),
                                ('add_feature_name',transformers.add_level(level_val=self.name,level_name=FEATURE_LEVEL,axis=1))
                            ])

    def get_data_needs(self):
        return self.data_needs



class Featurizer(TransformerMixin,BaseEstimator,HasDataNeeds):

    def __init__(self,index_levels,
                    loader,
                    features=[],
                    pre_cleaners=transformers.do_nothing(),
                    post_cleaners=transformers.do_nothing()):
        self.index_levels = index_levels
        self.loader=loader
        self.pre_cleaners = pre_cleaners
        self.post_cleaners = post_cleaners
        self.features = {}
        for f in features: self._add(f)

    def add_feature(self,name,data_needs,col_filter,agg_func,fillna_method=transformers.do_nothing):
        feature = Feature(
            name=name,
            data_needs=data_needs,
            col_filter=col_filter,
            fillna_method=fillna_method,
            aggregator=aggregator(levels=self.index_levels,agg_func=agg_func)
        )
        self._add(feature)
        return feature

    def _add(self,feature):
        dn_dict = self.features.get(tuple(feature.data_needs),{})
        dn_dict[feature.name] = feature
        self.features[tuple(feature.data_needs)] = dn_dict

    def remove_feature(self,feature):
        dn_dict = self.features.get(tuple(feature.data_needs),None)
        if dn_dict is None: return None
        feature = dn_dict.pop(feature.name,None)
        self.features[tuple(feature.data_needs)] = dn_dict
        return feature

    def get_data_needs(self):
        return list(set([dn for dn_list in self.features.keys() for dn in dn_list]))


    def fit_transform(self, X, y=None, **fit_params):
        return self._featurize(X=X,y=y,fit=True,**fit_params)

    def transform(self, X):
        return self._featurize(X,fit=False)

    def fit(self, X, y=None, **fit_params):
        self.pipeline_list = self._get_pipeline_list()
        return self

    def _featurize(self,X,y=None,fit=True,**fit_params):
        if fit: self.fit(X, y, **fit_params)
        logger.log('Featurizing...',new_level=True)
        df = None
        for dn,pipeline in self.pipeline_list:
            logger.log('{} - {}'.format(list(dn),', '.join([f.name for f in self.features[dn].values()])),new_level=True)
            if fit: df_feature = pipeline.fit_transform(X,y,**fit_params)
            else: df_feature = pipeline.transform(X)


            if df is None: df = df_feature
            elif not df_feature.empty:
                df = df.join(df_feature,how='outer')
            logger.end_log_level()

        logger.end_log_level()
        return df

    def _get_pipeline_list(self):
        pipeline_list = []
        for dn,feature_dict in self.features.iteritems():
            loader = clone(self.loader)
            loader.data_needs = dn
            pipeline = Pipeline([
                ('load_data',loader),
                ('pre-clean',clone(self.pre_cleaners)),
                ('featurize_{}'.format('-'.join(map(str,dn))),pdFeatureUnion(feature_dict.values())),
                ('post-clean',clone(self.post_cleaners)),
            ])
            pipeline_list.append((dn,pipeline))
        return pipeline_list

class pdFeatureUnion(TransformerMixin,BaseEstimator):

    def __init__(self,features):
        self.features = features

    def fit_transform(self, X, y=None, **fit_params):
        df= None
        logger.log('fit_transform features on DF {}'.format(X.shape),new_level=True)
        for feature in self.features:
            logger.log(feature.name)
            df_feature = feature.fit_transform(X,y,**fit_params)
            print X.head()
            print df_feature.head()
            if df is None: df = df_feature
            elif not df_feature.empty:
                df = df.join(df_feature,how='outer')
        logger.end_log_level()
        return df

    def fit(self, X, y=None, **fit_params):
        logger.log('fit features on DF {}'.format(X.shape),new_level=True)
        for feature in self.features:
            logger.log(feature.name)
            feature.fit(X,y,**fit_params)
        logger.end_log_level()
        return self

    def transform(self, X):
        df= None
        logger.log('transform features on DF {}'.format(X.shape),new_level=True)
        for feature in self.features:
            logger.log(feature.name)
            df_feature = feature.transform(X)

            if df is None: df = df_feature
            elif not df_feature.empty:
                df = df.join(df_feature,how='outer')

        logger.end_log_level()
        return df

# class featurizer(Pipeline,HasDataNeeds):
#
#     def __init__(self,aggregator,fillna,data_needs,name,preprocessor=None):
#
#         tuples = []
#         if preprocessor is not None:
#             """
#             Preprocessing includes filtering (i.e. only apply to these columns)
#             or calculation of a new column (i.e. get some columns, apply some transformation
#             directly to the timeseries, and then proceed with that new timeseries)
#
#             This could even be a Pipeline of transformations
#             """
#             tuples.append(('preprocessor',preprocessor))
#
#         tuples += [
#                 ('feature_aggregator',aggregator),
#                 ('feature_fillna',fillna),
#                 ('flatten_columns',transformers.flatten_index(axis=1,suffix=aggregator.name))
#             ]
#         super(featurizer,self).__init__(steps=tuples)
#         self.name = name
#         self.data_needs = data_needs
#
#     def get_data_needs(self):
#         return self.data_needs
#
#
# class simple_featurizer(featurizer):
#
#     def __init__(self,aggregator,fillna,slice_dict_list,name):
#         preprocessor= transformers.multislice_filter(slice_dict_list)
#         components = [slice_dict[column_names.COMPONENT] for slice_dict in slice_dict_list]
#         units = [slice_dict.get(column_names.UNITS,ALL) for slice_dict in slice_dict_list]
#         data_needs = zip(components,units)
#         super(simple_featurizer,self).__init__(aggregator,fillna,data_needs,name,preprocessor)


# def make_mapper(feature_tuples,df):
#     """
#     Each tuple is:
#     ('<TEXT NAME>',Featurizer,[columns to apply])
#     """
#
#     tuple_list = []
#     for name,featurizer,col_filter in feature_tuples:
#
#         pipeline_steps = [
#             ('{}_featurizer'.format(name),featurizer),
#             ('flatten_columns',transformers.flatten_index(axis=1,suffix=name))
#         ]
#
#         if col_filter != ALL:
#             pipeline_steps.insert(0,('{}_filter'.format(name),col_filter))
#
#         pipeline = Pipeline(pipeline_steps)
#
#
#
#         if col_details == ALL:
#             col_list = df.columns.tolist()
#         elif type(col_details) == tuple:
#             col_list = [col_details]
#         elif type(col_details) == list:
#             col_list = col_details
#         #this is a transformer, i.e. a custom filter
#         elif issubclass(type(col_details),TransformerMixin):
#             col_list = col_details.transform(df).columns.tolist()
#         else: #should be a dict
#             col_list = set()
#             for level,value in col_details.iteritems():
#                 col_list.update(utils.filter_columns(df,level,value).columns.tolist())
#             col_list = list(col_list)
#
#         tuple_list.append((name,featurizer,col_list))
#
#     mappings = []
#     for name,featurizer,col_list in tuple_list:
#         for col_name in col_list:
#             alias = {'alias':'_'.join(map(str,col_name) + [name])}
#             mappings.append((col_name,featurizer,alias))
#
#     return DataFrameMapper(mappings, df_out=True,input_df=True)
