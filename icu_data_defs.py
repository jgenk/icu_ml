import pandas as pd
import utils
from constants import variable_type,clinical_source,NO_UNITS,column_names

class data_dictionary(object):

    def __init__(self,xls_fname):

        self.load(xls_fname)

    def load(self,xls_fname):
        xls = pd.ExcelFile(xls_fname)
        obj_dict = {}
        obj_dict['xls_fname'] = xls_fname
        df_tables={}
        df_names={}
        for sheet_name in xls.sheet_names:
            df_tables[sheet_name] = xls.parse(sheet_name,index_col=0)
            df_names[sheet_name] = sheet_name
        obj_dict['tables'] = utils.Bunch(**df_tables)
        obj_dict['table_names'] = utils.Bunch(**df_names)

        self.__dict__.update(**obj_dict)
        self.__refresh_components()

    def __refresh_components(self):
        components = map(str,self.tables.definitions.component.unique().tolist())
        keys = map(lambda component: str.upper(component.replace(' ','_')),components)
        self.__dict__['components'] = utils.Bunch(**dict(zip(keys,components)))

    def save(self,xls_fname=None):
        if xls_fname is None: xls_fname = self.xls_fname
        writer = pd.ExcelWriter(xls_fname, engine='xlsxwriter')
        for table_name,table in self.tables.__dict__.iteritems():
            table.to_excel(writer,table_name)
        writer.save()
        return

    def add_definition(self,component,units=NO_UNITS,
                       variable_type=variable_type.QUANTITATIVE,
                       clinical_source=clinical_source.OBSERVATION,
                       lower_limit=pd.np.nan,
                       upper_limit=pd.np.nan,
                       list_id=pd.np.nan):
        new_id = _next_id(self.tables.definitions)
        self.tables.definitions.loc[new_id] = [component,units,variable_type,clinical_source,lower_limit,upper_limit,list_id]
        self.__refresh_components()
        return new_id

    def add_panel(self,panel_name,panel_map):
        """
        panel map: {table_name:id}
        """
        new_panel_id = _next_id(self.tables.panels)
        new_list_id = _next_id(self.tables.lists)
        self.tables.panels.loc[new_panel_id] = [panel_name,new_list_id]
        for ref_table,ref_id in panel_map:
            self.add_item_to_panel(new_panel_id,ref_table,ref_id)
        return new_panel_id

    def add_item_to_panel(self,panel_id,ref_table,ref_id):
        list_id = self.tables.panels.loc[panel_id,'list_id']
        return self.__add_list_item(list_id,ref_table,ref_id,pd.np.nan)

    def __add_list_item(self,list_id,ref_table,ref_id,seq_num):
        orig_index_name = self.tables.lists.index.name
        list_df = self.tables.lists.reset_index(drop=False)
        new_id = _next_id(list_df)
        list_df.loc[new_id] = [list_id,ref_table,ref_id,seq_num]
        list_df.set_index(orig_index_name,inplace=True)
        self.tables.lists = list_df
        return new_id

    def add_category(self,val_numeric,val_text):
        new_id = _next_id(self.tables.categories)
        self.tables.categories.loc[new_id] = [val_numeric,val_text]
        return new_id

    def add_category_list(self,categories,is_ordered=False):
        new_list_id = _next_id(self.tables.lists)
        for i,category_id in enumerate(categories):
            self.__add_list_item(new_list_id,
                                 self.table_names.categories,
                                 category_id,
                                 i if is_ordered else pd.np.nan)
        return new_list_id

    def get_panel_defintions(self,panel_id):
        list_id = self.tables.panels.loc[panel_id,'list_id']
        def_list = []
        for index, row in self.tables.lists.loc[list_id].iterrows():
            table = row['table']
            id_ = row['id']
            if table == 'panels':
                defs = self.get_panel_defintions(id_)
            else: defs = self.tables.__dict__[table].loc[[id_]]

            def_list.append(defs)

        return pd.concat(def_list)

    def get_categories(self,component):
        joined = self.tables.definitions.merge(self.tables.lists, left_on='list_id',right_index=True)
        joined = joined.merge(self.tables.categories,left_on='id',right_index=True)
        filtered = joined.loc[joined.component == component]
        if filtered.shape[0] == 0: return None
        out_df = filtered[['seq_num','val_numeric','val_text']].set_index('seq_num').sort_index()
        return out_df

    def defs_for_component(self,component):
        return self.get_defs({column_names.COMPONENT : component})

    def get_clinical_source(self,component):
        return self.defs_for_component(component).loc[:,'clinical_source'].iloc[0]

    def get_variable_type(self,component):
        return self.defs_for_component(component).loc[:,'variable_type'].iloc[0]

    def get_defs(self,data_specs,operator='and'):
        return _filter_defs(self.tables.definitions,data_specs)

    def get_components(self,specs={},panel_id=None,operator='and'):
        if panel_id is not None:
            defs = self.get_panel_defintions(panel_id)
        else:
            defs = self.tables.definitions
        return _filter_defs(defs,specs,operator).component.unique().tolist()

def _filter_defs(defs,specs,operator='and'):
    df_mask = pd.DataFrame(index=defs.index)
    if len(specs) == 0: df_mask.loc[:,0] = True

    for col_name,vals in specs.iteritems():
        if not isinstance(vals,list): vals = [vals]
        df_mask.loc[:,col_name] = (defs.loc[:,col_name].isin(vals))

    if operator == 'or': mask = df_mask.any(axis=1)
    else: mask = df_mask.all(axis=1)

    return defs.loc[mask]

def _next_id(df):
    return max(df.index.tolist())+1
