# encoding=utf8
from luwak_flow.convert.feature_convert import ColumnFilterConvert
from luwak_flow.convert.feature_convert import RowFilterConvert
from luwak_flow.convert.feature_convert import OneToOneConvert
from luwak_flow.convert.feature_convert import OneHotConvert
from luwak_flow.convert.feature_convert import NToOneConvert
from luwak_flow.engine.luwak import LuwakFlow

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
        ['AA', 'AB', 'AC', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'AE', 'AF', 'AG', 'AH', 'AK',
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
            ('AA', 'AB', 'AC', 'AA_AB_AC')
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

mri_luwak_feed = {
    'raw_csv_file': '../data/input/MRI.csv',
    'output_path': '../data/output/',
    'flow': []
}
mri_luwak_feed['flow'].append([
    (
        ColumnFilterConvert.remain_columns,
        [
            'PAADM_RowID',
            # 'PAPMI_No',
            # 'PAPMI_Medicare',
            # 'RAR_RegDate',
            # 'rar_RegTime',
            # 'DRPT_RowID',
            # 'DRPT_PAADM_DR',
            # 'DRPT_ReportDate',
            # 'drpt_ReportTime',
            # 'DRPT_VerifyDate',
            # 'drpt_VerifyTime',
            # 'ARCIM_Desc',
            # 'ARCIC_Desc',
            # 'ORCAT_Desc',
            # 'ExamDescExResult',
            'ResultDescExResult'
        ]
    ),
])
mri_luwak_feed['flow'].append([
    (
        OneToOneConvert.transform_punctuation,
        ['ResultDescExResult']
    )
])

c13_luwak_feed = {
    'raw_csv_file': './data/input/C13_data5.csv',
    'output_path': './data/output/',
    'flow': []
}
c13_luwak_feed['flow'].append([
    (
        OneToOneConvert.binning_numeric,
        [ ('Age',10,100,10) ]
    ),
])

c13_luwak_feed['flow'].append([
    (
        OneHotConvert.one_hot_convert,
        [ 'Age']
    ),
])

if __name__ == '__main__':
    lf = LuwakFlow(c13_luwak_feed)
    lf.flow_execute_engine()
