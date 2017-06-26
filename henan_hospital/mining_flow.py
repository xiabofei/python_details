# encoding=utf8

xiehe_mining_flow = []
xiehe_mining_flow.append([
    (
        'remain_columns',
        ['P', 'Q', 'R', 'S', 'T', 'U', 'V', 'AE', 'AF', 'AG', 'AH', 'AK',
         'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BN', 'BP', 'BQ',
         'DS', 'IY', 'IZ', 'JA', 'JB', 'JC', 'JF']
    ),
    (
        'filter_row_contain_na_value',
        ['P', 'Q', 'R', 'S', 'T', 'U', 'V', 'AE', 'AF', 'AG', 'AH', 'AK',
         'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BN', 'BP', 'BQ', 'DS', 'IY']
    ),
])
xiehe_mining_flow.append([
    (
        'normalization_convert',
        [('AK', 26, 81), ('BJ', 36, 862), ('BK', 0.2, 70), ('DS', 0, 98)]
    ),
    (
        'H_M_L_convert',
        [
            ('BD', 1.98, 23), ('BE', 0.23, 82.4), ('BF', 0.46, 6.03),
            ('BG', 50, 386), ('BH', 210, 667), ('BI', 5, 101)
        ]
    ),
    (
        'binary_convert',
        [('IZ', "有"), ('JA', "有"), ('JB', "有"), ('JC', "有"), ('JF', "有")]
    ),
])
xiehe_mining_flow.append([
    (
        'one_hot_convert',
        ['P', 'Q', 'R', 'S', 'T', 'U', 'V', 'AE', 'AF', 'AG', 'AH', 'BN', 'BP', 'BQ', 'IY']
    ),
])


henan_mining_flow = []
henan_mining_flow.append([
    (
        'two_to_one_convert', [('PECR_CheckDate', 'PAPAT_DE_Dob', 'derive_age_from_date', 'DER_AGE')]
    ),
    (
        'negative_or_positive_convert',
        [('C-14_碳14吹气试验', 'one'), ('C-13_碳13吹气试验', 'two'), ('PAPAT_DE_SexCode', 'three')]
    )
])
henan_mining_flow.append([
    (
        'two_to_one_convert',
        [('C-14_碳14吹气试验', 'C-13_碳13吹气试验', 'merge_pos_neg', 'COMPREHENSIVE_NP')]
    ),
])
henan_mining_flow.append([
    (
        'remain_columns',
        ['C-14_碳14吹气试验', 'C-13_碳13吹气试验', 'COMPREHENSIVE_NP', 'PAPAT_DE_SexCode']
    ),
])

