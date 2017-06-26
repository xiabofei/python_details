# encoding=utf8
import pandas as pd


class IO(object):

    DEFAULT_CHUNK_SIZE = 5000
    DEFAULT_QUOTE_CHAR = '"'
    DEFAULT_SEP = ','
    DEFAULT_HEADER = 0
    DEFAULT_DTYPE = object
    DEFAULT_INDEX_COL = False
    DEFAULT_INDEX = False

    @classmethod
    def read_from_csv(cls,
                      input_path,
                      chunk_size = DEFAULT_CHUNK_SIZE,
                      quote_char=DEFAULT_QUOTE_CHAR,
                      sep = DEFAULT_SEP,
                      header = DEFAULT_HEADER,
                      dtype = DEFAULT_DTYPE,
                      index_col = DEFAULT_INDEX_COL
                      ):
        return pd.read_csv(
            input_path,
            chunksize=chunk_size,
            quotechar=quote_char,
            sep=sep,
            header=header,
            dtype=dtype,
            index_col=index_col
        )

    @classmethod
    def write_to_csv(cls,
                     df,
                     output_path,
                     header,
                     quote_char = DEFAULT_QUOTE_CHAR,
                     index = DEFAULT_INDEX,
                     ):
        df.to_csv(
            output_path,
            header = header,
            quotechar= quote_char,
            index=index,
        )
