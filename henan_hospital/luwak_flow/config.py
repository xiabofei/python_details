# encoding=utf8

CHUNK_SIZE_THRESHOLD = 50000
FEATURE_NAME_CONNECT = '_'
MIN_FEATURE_CATEGORY_NUM = 1
MAX_FEATURE_CATEGORY_NUM = 100

# filter action
NA_FILTER = 'na_filter'
COLUMN_FILTER = 'column_filter'

# one to one convert action
HML_CONVERT = 'hml_convert'
BINARY_CONVERT = 'binary_convert'
NORMALIZATION_CONVERT = 'normalization_convert'

USER_DEFINED_ONE_TO_ONE = 'user_defined_one_to_one'
POSITIVE = '+'
NEGATIVE = '-'

# one to N action
ONE_HOT_ENCODER = 'one_hot_encoder'

# N to one action
TWO_TO_ONE_CONVERT = 'two_to_one_convert'

# action type
PLUS = '+'
MINUS = '-'
MULTIPLY = '*'
DERIVE_AGE_FROM_DATE = 'derive_age_from_date'
MERGE_POS_NEG = 'merge_pos_neg'
ACTION_TYPE = [PLUS, MINUS, MULTIPLY, DERIVE_AGE_FROM_DATE,MERGE_POS_NEG]
