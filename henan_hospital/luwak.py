# encoding=utf8
from config import *
from convert import FeatureConvertProxy
import os
from csv_io import IO


class LuwakFlow(object):
    def __init__(self, raw_csv_file, mining_flow, output_path):
        self.raw_csv_file = raw_csv_file
        self.mining_flow = mining_flow
        self.output_path = output_path

    def __setattr__(self, key, value):
        if key == 'mining_flow':
            if isinstance(value, list):
                object.__setattr__(self, key, value)
            else:
                raise TypeError("feature_processor is not list type")
        object.__setattr__(self, key, value)

    def _flow_file_name(self, index):
        return str(index) + "_flow"

    def flow_execute_engine(self):
        if hasattr(self, 'mining_flow'):
            for index, flow in enumerate(self.mining_flow):
                if index == 0:
                    self._execute_single_flow(index, flow, self.raw_csv_file)
                else:
                    self._execute_single_flow(index, flow, self.output_path + self._flow_file_name(index - 1))
        else:
            raise AttributeError("instance has no attribute \'mining_flow\'")

    def _execute_single_flow(self, index, flow, last_flow_file):
        output_path = self.output_path + self._flow_file_name(index)
        if os.path.isfile(output_path):
            os.remove(output_path)
        fc_proxy = FeatureConvertProxy(last_flow_file, flow)
        with open(output_path, 'a') as f_output:
            _header = True
            for df in IO.read_from_csv(last_flow_file):
                df = fc_proxy.execute_all(df)
                IO.write_to_csv(df, f_output, header=_header)
                _header = False


if __name__ == '__main__':
    raw_csv_file = './data/input/C13.csv'
    output_path = './data/output/'
    henan_mining_flow = []
    henan_mining_flow.append([
        (
            'two_to_one_convert', [('PECR_CheckDate', 'PAPAT_DE_Dob', DERIVE_AGE_FROM_DATE, 'DER_AGE')]
        ),
        (
            'negative_or_positive_convert',
            [('C-14_碳14吹气试验', 'one'), ('C-13_碳13吹气试验', 'two'), ('PAPAT_DE_SexCode', 'three')]
        )
    ])
    henan_mining_flow.append([
        (
            'two_to_one_convert',
            [('C-14_碳14吹气试验', 'C-13_碳13吹气试验', MERGE_POS_NEG, 'COMPREHENSIVE_NP')]
        ),
    ])
    henan_mining_flow.append([
        (
            'remain_columns',
            ['C-14_碳14吹气试验', 'C-13_碳13吹气试验', 'COMPREHENSIVE_NP', 'PAPAT_DE_SexCode']
        ),
    ])
    lf = LuwakFlow(raw_csv_file, henan_mining_flow, output_path)
    lf.flow_execute_engine()
    print 'exit'
