from typing import List

import pandas as pd
import numpy as np


from DataInterface.DataInterface import DataInterface
from DataPreprocessing.AdminData.Data import AdminDataComplete


class AdminData(DataInterface):
    def __init__(self):
        super().__init__()
        self._admin_data_complete = AdminDataComplete()
        self.data = self._admin_data_complete.data
        self.pop_metadata = self._admin_data_complete.pop_metadata
        self.equip_metadata = self._admin_data_complete.equip_metadata
        self.selected_pop_vars = list(self._admin_data_complete.selected_pop_vars['COD_VAR'])

    def get_admin_data(self, subset: List[str] = None, only_iris_habitat: bool = True, coarsened_equip: bool = True, selected_pop_vars: bool = True):
        if subset is not None:
            iris_in_intersection = np.intersect1d(subset, self.data.index)
            data = self.data.loc[iris_in_intersection].copy()
        else:
            data = self.data.copy()

        if only_iris_habitat:
            data = self._get_habitat_iris_data(data=data)
        if coarsened_equip:
            data = self._get_admin_data_with_coarsened_equipment_data(data=data,
                                                                      equip_metadata=self.equip_metadata)
        if selected_pop_vars:
            data = self._reduce_data_to_selected_population_variables_plus_equipment_variables(data=data)
        return data

    def get_variable_description(self, var_name: str):
        if var_name.startswith('EQUIP_'):
            short_var_name = var_name.replace('EQUIP_', '')
            if len(short_var_name) == 2:
                return self.equip_metadata[self.equip_metadata['SDOM'] == short_var_name]['SDOM_TITLE'].values[
                    0]
            else:
                return \
                self.equip_metadata[self.equip_metadata['TYPEQU'] == short_var_name]['TYPEQU_TITLE'].values[0]
        else:
            return self.pop_metadata[self.pop_metadata['COD_VAR'] == var_name]['LIB_VAR_LONG'].values[0]

    def _reduce_data_to_selected_population_variables_plus_equipment_variables(self, data):
        selected_equip_variables = list(data.columns[data.columns.str.contains('EQUIP')])
        vars_to_keep = self.selected_pop_vars + selected_equip_variables
        data = data[vars_to_keep]
        return data

    @staticmethod
    def _get_habitat_iris_data(data):
        data = data[data['TYP_IRIS'] == 'H'].copy()
        data.drop(columns=['TYP_IRIS'], inplace=True)
        return data

    @staticmethod
    def _get_admin_data_with_coarsened_equipment_data(data, equip_metadata):
        equip_data = data[[c for c in data.columns if c.startswith('EQUIP_')]].T
        coarsening = {f"EQUIP_{row['TYPEQU']}": f"EQUIP_{row['SDOM']}" for _, row in equip_metadata.iterrows()}
        equip_data.rename(index=coarsening, inplace=True)
        equip_data = equip_data.groupby(equip_data.index).sum().T
        all_admin_data = pd.concat([data[[c for c in data.columns if not c.startswith('EQUIP_')]], equip_data],
                                   axis=1)
        return all_admin_data