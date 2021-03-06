# encoding=utf8

import os
from luwak_flow.convert.feature_convert import FeatureConvertFactory
from luwak_flow.utils.luwak_io import IO


class LuwakFlow(object):
    def __init__(self, luwak_feed):
        self.raw_csv_file = luwak_feed['raw_csv_file']
        self.output_path = luwak_feed['output_path']
        self.mining_flow = luwak_feed['flow']

    def __setattr__(self, key, value):
        if key == 'raw_csv_file':
            if os.path.isfile(value):
                object.__setattr__(self, key, value)
            else:
                raise IOError("file \'%s\' not found" % value)
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
        fc_factory = FeatureConvertFactory(last_flow_file, flow)
        with open(output_path, 'a') as f_output:
            _header = True
            for df in IO.read_from_csv(last_flow_file):
                df = fc_factory.execute_all(df)
                IO.write_to_csv(df, f_output, header=_header)
                _header = False
