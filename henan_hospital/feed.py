# encoding=utf8
from convert import ColumnFilterConvert, RowFilterConvert, OneToOneConvert, OneHotConvert, NToOneConvert

"""
每个flow参数如何定义:
1. 需要传参的 用(列名, 参数1, 参数2,... )
2. 不需要传参的 直接传列名即可
"""

xiehe_luwak_feed = {
    'raw_csv_file': './data/input/xiehe.csv',
    'output_path': './data/output/',
    'flow': []
}
xiehe_luwak_feed['flow'].append([
    (
        ColumnFilterConvert.remain_columns,
        ['AA','AB','AC', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'AE', 'AF', 'AG', 'AH', 'AK',
         'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BN', 'BP', 'BQ',
         'DS', 'IY', 'IZ', 'JA', 'JB', 'JC', 'JF']
    ),
    (
        RowFilterConvert.filter_row_contain_na_value,
        ['P', 'Q', 'R', 'S', 'T', 'U', 'V', 'AE', 'AF', 'AG', 'AH', 'AK',
         'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BN', 'BP', 'BQ', 'DS', 'IY']
    ),
])
xiehe_luwak_feed['flow'].append([
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
    (
        OneToOneConvert.plus_one,
        ['AA']
    ),
    (
        OneHotConvert.one_hot_convert,
        [
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'AE', 'AF', 'AG', 'AH', 'BN', 'BP', 'BQ', 'IY'
        ]
    ),
    (
        NToOneConvert.find_largest,
        [
            ('AA','AB','AC','AA_AB_AC')
        ]
    ),
    (
        ColumnFilterConvert.remain_columns,
        ['AA', 'AB', 'AC', 'AA_AB_AC']
    ),
])

henan_luwak_feed = {
    'raw_csv_file': './data/input/tijian.csv',
    'output_path': './data/output/',
    'flow': []
}
henan_luwak_feed['flow'].append([
    (
        ColumnFilterConvert.remain_columns,
        [
            'PAPAT_PatientID', 'PAPAT_DE_Name', 'PECR_CheckDate', 'PAPAT_DE_Dob', 'PAPAT_DE_SexCode',
        ]
    ),
    (
        NToOneConvert.derive_age_from_date,
        [
            ('PECR_CheckDate', 'PAPAT_DE_Dob', 'DER_AGE')
        ]
    ),
])
henan_luwak_feed['flow'].append([
    (
        OneHotConvert.one_hot_convert,
        ['PAPAT_DE_SexCode']
    ),
])
