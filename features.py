from sklearn.base import TransformerMixin,BaseEstimator,clone
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from constants import column_names,SEG_ID,NO_SEGMENT,ALL,CUSTOM_FILTER,FEATURE_LEVEL
import numpy as np
import utils
import transformers
import logger
import pandas as pd


"""
Feature Creation
"""

class Featurizer(TransformerMixin,BaseEstimator):
    def __init__(self,agg_func,resample_freq,
                    col_filter=transformers.do_nothing(),
                    pre_processor=transformers.do_nothing(),
                    post_processor=transformers.do_nothing(),
                    fillna_transformer=transformers.do_nothing(),
                    dropna=True
                    ):
        self.col_filter = col_filter
        self.agg_func = agg_func
        self.resample_freq = resample_freq
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        self.fillna_transformer = fillna_transformer
        self.dropna = dropna

    def _make_pipeline(self):
        dropna_transformer = transformers.do_nothing()
        if self.dropna: dropna_transformer = transformers.DropNaN(how='all')

        return Pipeline([
            ('col_filter',self.col_filter),
            ('pre_processor',self.pre_processor),
            ('aggregator',ResampleAggregator(self.agg_func,column_names.ID,column_names.DATETIME,self.resample_freq)),
            ('post_processor',self.post_processor),
            ('drop_na_rows',dropna_transformer),
            ('fill_na',self.fillna_transformer)
        ])


    def fit(self, X, y=None, **fit_params):
        self.pipeline = self._make_pipeline()
        return self.pipeline.fit(X, y, **fit_params)

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self,X, y=None, **fit_params):
        self.pipeline = self._make_pipeline()
        return self.pipeline.fit_transform(X, y, **fit_params)

class DataSpecsFeaturizer(Featurizer):
    def __init__(self,agg_func,resample_freq,
                    data_specs=[],
                    pre_processor=transformers.do_nothing(),
                    post_processor=transformers.do_nothing(),
                    fillna_transformer=transformers.do_nothing(),
                    dropna=True
                    ):
            self.data_specs = data_specs
            super(DataSpecsFeaturizer,self).__init__(agg_func,resample_freq,
                                                        col_filter=transformers.DataSpecFilter(data_specs),
                                                        pre_processor=pre_processor,
                                                        post_processor=post_processor,
                                                        fillna_transformer=fillna_transformer,
                                                        dropna=dropna
                                                    )

class ResampleAggregator(TransformerMixin,BaseEstimator):

    def __init__(self,agg_func,groupby_level=None,resample_level=None,resample_freq=None):
        self.agg_func=agg_func
        self.groupby_level=groupby_level
        self.resample_level=resample_level
        self.resample_freq=resample_freq

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        if self.groupby_level is not None:
            to_resample = X.groupby(level=self.groupby_level)
        else: to_resample = X

        if self.resample_level is not None:
            to_agg = to_resample.resample(rule=self.resample_freq,level=self.resample_level,label='right')
        else:
            to_agg = to_resample

        return to_agg.agg(self.agg_func)

class FeatureUnionDF(TransformerMixin,BaseEstimator):
    def __init__(self,featurizers,add_name_level=True):
        self.featurizers = featurizers
        self.add_name_level = add_name_level

    def fit(self, X, y=None, **fit_params):
        for f in self.featurizers:
            f[1].fit(X, y=None, **fit_params)
        return self

    def transform(self, X):
        return self.do_union(self,X,False)

    def fit_transform(self,X, y=None, **fit_params):
        return self.do_union(X, True, y, **fit_params)

    def do_union(self,X, is_fit, y=None, **fit_params):

        logger.log('Begin union for {} transformers'.format(len(self.featurizers)),new_level=True)
        df_features = None

        for f in self.featurizers:
            logger.log(f[0],new_level=True)
            
            if is_fit: df_ft = f[1].fit_transform(X)
            else: df_ft = f[1].transform(X)
            if self.add_name_level:
                df_ft = utils.add_same_val_index_level(df_ft,level_val=f[0],level_name=FEATURE_LEVEL,axis=1)
            if df_features is None: df_features = df_ft
            else: df_features = df_features.join(df_ft,how='outer')
            del df_ft

            logger.end_log_level()
        logger.end_log_level()
        return df_features


class DataSetFactory(TransformerMixin,BaseEstimator):

    def __init__(self,
                 featurizers,
                 resample_freq,
                 components,
                 etl_manager,
                 pre_processor=transformers.do_nothing(),
                 post_processor=transformers.do_nothing(),
                 should_fillna=True):
        self.featurizers = featurizers
        self.resample_freq = resample_freq
        self.components = components
        self.etl_manager=etl_manager
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        self.should_fillna=should_fillna
        return

    def fit(self,X,y=None, **fit_params):
        self.fit_transform(X, y, **fit_params)
        return self

    def transform(self, X):
        return self.make_feature_set(self,X,False)

    def fit_transform(self,X, y=None, **fit_params):
        return self.make_feature_set(X, True, y, **fit_params)

    def make_feature_set(self, ids, fit, y=None, **fit_params):
        logger.log("Make Feature Set. id_count={}, #features={}, components=".format(len(ids),len(self.featurizers),self.components),new_level=True)
        if fit:
            self.comp_preprocessors = [(c,self.preprocessor_pipeline(c)) for c in self.components]

        adjusted_featurizers = [(ft_name,self.adjust_featurizer(ft)) for ft_name,ft in self.featurizers]

        pipeline_steps = [
            ('pre_processors',FeatureUnionDF(self.comp_preprocessors, add_name_level=False)),
            ('feature_union',FeatureUnionDF(adjusted_featurizers)),
            ('post_processor',self.post_processor),
        ]

        if self.should_fillna:
            pipeline_steps.append(('fillna',LocAndFillNaN(self.featurizers)))

        ft_union_pipeline = Pipeline(pipeline_steps)
        if fit: df = ft_union_pipeline.fit_transform(ids, y, **fit_params)
        else: df = ft_union_pipeline.transform(ids)

        logger.end_log_level()
        return df

    def adjust_featurizer(self,ft):
        return Featurizer(ft.agg_func,
                            resample_freq=self.resample_freq,
                            col_filter=ft.col_filter,
                            pre_processor=ft.pre_processor,
                            post_processor=ft.post_processor,
                            dropna=False
                        )

    def preprocessor_pipeline(self,comp):
        return Pipeline([
            ('data_loader',ComponentDataLoader(comp, self.etl_manager)),
            ('pre_processor',clone(self.pre_processor))
        ])

class LocAndFillNaN(TransformerMixin,BaseEstimator):

    def __init__(self,featurizers):
        self.featurizers = featurizers

    def transform(self, df):
        df = df.copy()
        for ft_name,ft in self.featurizers:
            df[ft_name] = ft.fillna_transformer.transform(df[ft_name])
        return df

    def fit(self, df, y=None, **fit_params):
        for ft_name,ft in self.featurizers:
            ft.fillna_transformer.fit(df[ft_name],y,**fit_params)
        return self

class ComponentDataLoader(TransformerMixin,BaseEstimator):

    def __init__(self,component,etl_manager):
        self.component = component
        self.etl_manager = etl_manager

    def transform(self, X):
        logger.log('Load data from component: {}'.format(self.component.upper()),new_level=True)
        if isinstance(X,pd.DataFrame) or isinstance(X,pd.Series):
            X = X.index
        if isinstance(X, pd.Index):
            ids=X.get_level_values(column_names.ID).unique().tolist()
        else: ids=X

        df_component = self.etl_manager.open_df(self.component,ids=ids)

        logger.end_log_level()

        return df_component

    def fit(self, X, y=None, **fit_params):
        return self
