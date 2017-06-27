# encoding=utf8
from convert import ColumnFilterConvert, RowFilterConvert, OneToOneConvert, OneHotConvert, NToOneConvert

xiehe_mining_flow = []
xiehe_mining_flow.append([
    (
        ColumnFilterConvert.remain_columns,
        ['P', 'Q', 'R', 'S', 'T', 'U', 'V', 'AE', 'AF', 'AG', 'AH', 'AK',
         'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BN', 'BP', 'BQ',
         'DS', 'IY', 'IZ', 'JA', 'JB', 'JC', 'JF']
    ),
    (
        RowFilterConvert.filter_row_contain_na_value,
        ['P', 'Q', 'R', 'S', 'T', 'U', 'V', 'AE', 'AF', 'AG', 'AH', 'AK',
         'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BN', 'BP', 'BQ', 'DS', 'IY']
    ),
])
xiehe_mining_flow.append([
    (
        OneToOneConvert.normalization_convert,
        [
            ('AK', 26, 81), ('BJ', 36, 862), ('BK', 0.2, 70), ('DS', 0, 98)
        ]
    ),
    (
        OneToOneConvert.H_M_L_convert,
        [
            ('BD', 1.98, 23), ('BE', 0.23, 82.4), ('BF', 0.46, 6.03),
            ('BG', 50, 386), ('BH', 210, 667), ('BI', 5, 101)
        ]
    ),
    (
        OneToOneConvert.binary_convert,
        [
            ('IZ', "有"), ('JA', "有"), ('JB', "有"), ('JC', "有"), ('JF', "有")
        ]
    ),
])
xiehe_mining_flow.append([
    (
        OneHotConvert.one_hot_convert,
        [
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'AE', 'AF', 'AG', 'AH', 'BN', 'BP', 'BQ', 'IY'
        ]
    ),
])

henan_mining_flow = []
henan_mining_flow.append([
    (
        NToOneConvert.two_to_one_convert,
        [('PECR_CheckDate', 'PAPAT_DE_Dob', 'derive_age_from_date', 'DER_AGE')]
    ),
    (
        OneToOneConvert.binning_age,
        [('DER_AGE',)]
    ),
])
henan_mining_flow.append([
    (
        OneHotConvert.one_hot_convert,
        ['DER_AGE']
    )
])
