
ALL = 'all'
NO_UNITS = 'no_units'
START_DT = 'start_dt'
END_DT = 'end_dt'
SEG_ID = 'seg_id'
CUSTOM_FILTER = 'custom'
NO_SEGMENT = -1
FEATURE_LEVEL = 'feature'

class variable_type(object):
    QUANTITATIVE = 'qn'
    ORDINAL = 'ord'
    NOMINAL = 'nom'

class clinical_source(object):
    INTERVENTION = 'intervention'
    OBSERVATION = 'observation'

class column_names:
    UNITS='units'
    VALUE='value'
    ID='id'
    DATETIME='datetime'
    DESCRIPTION='description'
    COMPONENT='component'
    VAR_TYPE = 'variable_type'
