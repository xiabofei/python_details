#encoding=utf8

from config import FACTOR_DESCRIPTION, TRANSFORM_RELATION
from config import RELATION_1, RELATION_2, RELATION_3, RELATION_4, RELATION_5
from config import TARGET_DESCRIPTION, TARGET_FACTORS_GROUP
from config import CATEGORY_NAME
from config import FACTOR_DEFAULT_VALUE, TARGET_RISK_FACTOR_RADAR, TARGET_RISK_FACTOR_POLAR
from pprint import pprint
from collections import OrderedDict

import random
import cPickle


class rating_engine(object):

    # Each _risk_factors[TRANSFORM_RELATION] will be converted from 'list of tuple' to 'OrderDict' in __new__
    # when the first instance is created
    _risk_factors = {
        'P':{
            FACTOR_DESCRIPTION: u'腹泻',
            TRANSFORM_RELATION: RELATION_1,
            FACTOR_DEFAULT_VALUE: 2

        },
        'Q':{
            FACTOR_DESCRIPTION: u'直肠炎',
            TRANSFORM_RELATION: RELATION_1,
            FACTOR_DEFAULT_VALUE: 2
        },
        'R':{
            FACTOR_DESCRIPTION: u'尿频',
            TRANSFORM_RELATION: RELATION_1,
            FACTOR_DEFAULT_VALUE: 1
        },
        'S':{
            FACTOR_DESCRIPTION: u'恶心',
            TRANSFORM_RELATION: RELATION_1,
            FACTOR_DEFAULT_VALUE: 1
        },
        'T':{
            FACTOR_DESCRIPTION: u'呕吐',
            TRANSFORM_RELATION: RELATION_1,
            FACTOR_DEFAULT_VALUE: 1
        },
        'U':{
            FACTOR_DESCRIPTION: u'腹痛',
            TRANSFORM_RELATION: RELATION_1,
            FACTOR_DEFAULT_VALUE: 1
        },
        'V':{
            FACTOR_DESCRIPTION: u'乏力',
            TRANSFORM_RELATION: RELATION_1,
            FACTOR_DEFAULT_VALUE: 1
        },
        'AE':{
            FACTOR_DESCRIPTION: u'白细胞毒性',
            TRANSFORM_RELATION: RELATION_2,
            FACTOR_DEFAULT_VALUE: 1
        },
        'AF':{
            FACTOR_DESCRIPTION: u'中性粒细胞毒性',
            TRANSFORM_RELATION: RELATION_2,
            FACTOR_DEFAULT_VALUE: 1
        },
        'AG':{
            FACTOR_DESCRIPTION: u'血红蛋白毒性',
            TRANSFORM_RELATION: RELATION_2,
            FACTOR_DEFAULT_VALUE: 1
        },
        'AH':{
            FACTOR_DESCRIPTION: u'血小板毒性',
            TRANSFORM_RELATION: RELATION_2,
            FACTOR_DEFAULT_VALUE: 1
        },
        'AK':{
            FACTOR_DESCRIPTION: u'年龄[1-100]',
            TRANSFORM_RELATION: None,
            FACTOR_DEFAULT_VALUE: u'52'
        },
        'IY':{
            FACTOR_DESCRIPTION: u'分期',
            TRANSFORM_RELATION: RELATION_3,
            FACTOR_DEFAULT_VALUE: 1
        },
        'IZ':{
            FACTOR_DESCRIPTION: u'腹主动脉旁淋巴结转移',
            TRANSFORM_RELATION: RELATION_4,
            FACTOR_DEFAULT_VALUE: 1
        },
        'JA':{
            FACTOR_DESCRIPTION: u'盆腔淋巴结转移',
            TRANSFORM_RELATION: RELATION_4,
            FACTOR_DEFAULT_VALUE: 1
        },
        'BD':{
            FACTOR_DESCRIPTION: u'白细胞[1.98-25.45]',
            TRANSFORM_RELATION: None,
            FACTOR_DEFAULT_VALUE: u'6.7'
        },
        'BE':{
            FACTOR_DESCRIPTION: u'中性粒细胞[0.23-89.2]',
            TRANSFORM_RELATION: None,
            FACTOR_DEFAULT_VALUE: u'4.275'
        },
        'BF':{
            FACTOR_DESCRIPTION: u'淋巴细胞[0.32-30.5]',
            TRANSFORM_RELATION: None,
            FACTOR_DEFAULT_VALUE: u'1.82'
        },
        'BG':{
            FACTOR_DESCRIPTION: u'血红蛋白[47-386]',
            TRANSFORM_RELATION: None,
            FACTOR_DEFAULT_VALUE: u'129'
        },
        'BH':{
            FACTOR_DESCRIPTION: u'血小板[30-731]',
            TRANSFORM_RELATION: None,
            FACTOR_DEFAULT_VALUE: u'251'
        },
        'BI':{
            FACTOR_DESCRIPTION: u'ALT[2-335]',
            TRANSFORM_RELATION: None,
            FACTOR_DEFAULT_VALUE: u'15'
        },
        'BJ':{
            FACTOR_DESCRIPTION: u'Cr[34-1589]',
            TRANSFORM_RELATION: None,
            FACTOR_DEFAULT_VALUE: u'60'
        },
        'BK':{
            FACTOR_DESCRIPTION: u'SCC[0.2-70]',
            TRANSFORM_RELATION: None,
            FACTOR_DEFAULT_VALUE: u'3.65'
        },
        'BM':{
            FACTOR_DESCRIPTION: u'HBsAg',
            TRANSFORM_RELATION: RELATION_5,
            FACTOR_DEFAULT_VALUE: 1
        },
        'BN':{
            FACTOR_DESCRIPTION: u'HBsAb',
            TRANSFORM_RELATION: RELATION_5,
            FACTOR_DEFAULT_VALUE: 1
        },
        'BO':{
            FACTOR_DESCRIPTION: u'HBeAg',
            TRANSFORM_RELATION: RELATION_5,
            FACTOR_DEFAULT_VALUE: 1
        },
        'BP':{
            FACTOR_DESCRIPTION: u'HbeAb',
            TRANSFORM_RELATION: RELATION_5,
            FACTOR_DEFAULT_VALUE: 1
        },
        'BQ':{
            FACTOR_DESCRIPTION: u'HBcAb',
            TRANSFORM_RELATION: RELATION_5,
            FACTOR_DEFAULT_VALUE: 1
        },
        'BR':{
            FACTOR_DESCRIPTION: u'HCV',
            TRANSFORM_RELATION: RELATION_5,
            FACTOR_DEFAULT_VALUE: 1
        },
        'BS':{
            FACTOR_DESCRIPTION: u'TP',
            TRANSFORM_RELATION: RELATION_5,
            FACTOR_DEFAULT_VALUE: 1
        },
        'DK':{
            FACTOR_DESCRIPTION: u'先期化疗',
            TRANSFORM_RELATION: RELATION_4,
            FACTOR_DEFAULT_VALUE: 1
        },
        'DS':{
            FACTOR_DESCRIPTION: u'放疗持续时间[0-112]',
            TRANSFORM_RELATION: None,
            FACTOR_DEFAULT_VALUE: u'51'
        },
        'JB':{
            FACTOR_DESCRIPTION: u'高血压既往史',
            TRANSFORM_RELATION: RELATION_4,
            FACTOR_DEFAULT_VALUE: 1
        },
        'JC':{
            FACTOR_DESCRIPTION: u'糖尿病既往史',
            TRANSFORM_RELATION: RELATION_4,
            FACTOR_DEFAULT_VALUE: 1
        },
        'JF':{
            FACTOR_DESCRIPTION: u'各类手术既往史',
            TRANSFORM_RELATION: RELATION_4,
            FACTOR_DEFAULT_VALUE: 1
        },
        'JG':{
            FACTOR_DESCRIPTION: u'药物过敏既往史',
            TRANSFORM_RELATION: RELATION_4,
            FACTOR_DEFAULT_VALUE: 1
        }
    }

    _clinical_target = {
        'IT': {
            TARGET_DESCRIPTION: u'复发转移',
            TARGET_FACTORS_GROUP:OrderedDict([
                (CATEGORY_NAME['CU_FEN_QI'],['IY']),
                (CATEGORY_NAME['ZHONGLIU_BIAOJIWU'],['BK']),
                (CATEGORY_NAME['ZHUAN_YI'],['IZ', 'JA']),
                (CATEGORY_NAME['FANG_HUA_LIAO'],['DK', 'DS']),
                (CATEGORY_NAME['FU_FAN_YING'],['P', 'Q', 'R', 'S', 'T', 'U', 'V']),
                (CATEGORY_NAME['JI_WANG_SHI'],['JB', 'JC', 'JF', 'JG']),
                (CATEGORY_NAME['XUE_YE_DU_XING'],['AE', 'AF', 'AG', 'AH', 'AK']),
                (CATEGORY_NAME['XUE_CHANG_GUI'],['BD', 'BE', 'BF', 'BG', 'BH']),
                (CATEGORY_NAME['GAN_SHEN_GONG'],['BI', 'BJ']),
                (CATEGORY_NAME['GAN_RAN'],['BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'BS'])
            ]),
            TARGET_RISK_FACTOR_RADAR:[
                'BK', # SCC
                'BP', # HbeAb
                # 'DS', # 放疗持续时间
                'IZ', # 腹主动脉旁淋巴结转移
                'AG', # 血红蛋白毒性
                'JF', # 各种手术既往史
                'IY', # 分期
                'U',  # 腹痛
                'S'  # 恶心
                # 'V',  # 乏力
                # 'Q'   # 直肠炎
            ],
            TARGET_RISK_FACTOR_POLAR:[]
        },
        'IV': {
            TARGET_DESCRIPTION: u'有肿瘤残留',
            TARGET_FACTORS_GROUP:{},
            TARGET_RISK_FACTOR_RADAR:[],
            TARGET_RISK_FACTOR_POLAR:[]
        },
        'IW': {
            TARGET_DESCRIPTION: u'死亡',
            TARGET_FACTORS_GROUP:{},
            TARGET_RISK_FACTOR_RADAR:[],
            TARGET_RISK_FACTOR_POLAR:[]
        }
    }

    _addressed_relation_ship = False
    _check_risk_radar_data = False

    def __new__(cls, factors, target):

        if cls._addressed_relation_ship==False:
            for k,v in cls._risk_factors.items():
                v[TRANSFORM_RELATION] = \
                    OrderedDict(v[TRANSFORM_RELATION]) if v[TRANSFORM_RELATION] else v[TRANSFORM_RELATION]
            cls._addressed_relation_ship = True

        if cls._check_risk_radar_data == False:
            for k,v in cls._clinical_target.items():
                all_columns = []
                for columns in v[TARGET_FACTORS_GROUP].values():
                    all_columns += columns
                for column in v[TARGET_RISK_FACTOR_RADAR]:
                    if column not in all_columns:
                        raise Exception("column \"%s\" not in \"%s\" factor groups" % (column, k))
            cls._check_risk_radar_data = True

        return object.__new__(cls, factors, target)

    def __init__(self, factors, target):
        self.target = target
        self.factors = factors

    def __setattr__(self, key, value):
        if key=='target':
            assert value in ['IT','IV','IW'], 'clinical target "%s" not match' % (value)
            object.__setattr__(self, key, value)

    def produce_risk_factor_radar_data(self):
        """
        All the risk factors that occurs in the
        :return: list
                 [ column1,column2,..., ]
        """
        return self._clinical_target[self.target][TARGET_RISK_FACTOR_RADAR]

    def produce_factor_category_list(self):
        """
        Factor category info for front page display
        :return: list
                  [
                      {
                          category:'副反应',
                          factors:
                          [
                              {
                                  'factor_col':'P',
                                  'factor_name':'腹泻',
                                  'opts':['无', I, II, III, IV]
                              }
                              ...
                          ]

                      }
                      ...
                  ]
        """
        ret = []
        def _get_factor_opts(transform_relation):
            return { 'text':transform_relation.keys(),'value':transform_relation.values() } \
                if isinstance(transform_relation, OrderedDict) else transform_relation
        for k,v in self._clinical_target[self.target][TARGET_FACTORS_GROUP].iteritems():
            factor_category = {}
            factor_category['category_name'] = k
            factor_category['factors'] = []
            for factor in v:
                tmp = {
                    'factor_col':factor,
                    'factor_name':self._risk_factors[factor][FACTOR_DESCRIPTION],
                    'opts':_get_factor_opts(self._risk_factors[factor][TRANSFORM_RELATION]),
                    'default':self._risk_factors[factor][FACTOR_DEFAULT_VALUE]
                }
                factor_category['factors'].append(tmp)
            ret.append(factor_category)
        return ret

    def transform_factors(self, factors):
        """
        transform factors then it can be used by trained model
        :param factors: dict,
               {
                  'U_val' : '无',
                  'JF_val' : '有',
                  ...
               }
        :return: dict, can be used by trained model
        """
        ret = {}
        for k,v in factors.items():
            ret[k] = \
                v if (not self._risk_factors[k][TRANSFORM_RELATION]) else self._risk_factors[k][TRANSFORM_RELATION][v]
        ret = { k:float(v) for k,v in ret.items() }
        return ret


if __name__ == '__main__':
    engine = rating_engine({}, 'IT')
    print engine.factors
    print engine.target
