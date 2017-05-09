#encoding=utf8

from config import FACTOR_DESCRIPTION, TRANSFORM_RELATION
from config import RELATION_1, RELATION_2, RELATION_3, RELATION_4, RELATION_5
from config import TARGET_DESCRIPTION, TARGET_FACTORS_GROUP
from config import CATEGORY_NAME

import random

import cPickle


class rating_engine(object):

    _risk_factors = {
        'P':{
            FACTOR_DESCRIPTION: u'腹泻',
            TRANSFORM_RELATION: RELATION_1
        },
        'Q':{
            FACTOR_DESCRIPTION: u'直肠炎',
            TRANSFORM_RELATION: RELATION_1
        },
        'R':{
            FACTOR_DESCRIPTION: u'尿频',
            TRANSFORM_RELATION: RELATION_1
        },
        'S':{
            FACTOR_DESCRIPTION: u'恶心',
            TRANSFORM_RELATION: RELATION_1
        },
        'T':{
            FACTOR_DESCRIPTION: u'呕吐',
            TRANSFORM_RELATION: RELATION_1
        },
        'U':{
            FACTOR_DESCRIPTION: u'腹痛',
            TRANSFORM_RELATION: RELATION_1
        },
        'V':{
            FACTOR_DESCRIPTION: u'乏力',
            TRANSFORM_RELATION: RELATION_1
        },
        'AE':{
            FACTOR_DESCRIPTION: u'白细胞毒性',
            TRANSFORM_RELATION: RELATION_2
        },
        'AF':{
            FACTOR_DESCRIPTION: u'中性粒细胞毒性',
            TRANSFORM_RELATION: RELATION_2
        },
        'AG':{
            FACTOR_DESCRIPTION: u'血红蛋白毒性',
            TRANSFORM_RELATION: RELATION_2
        },
        'AH':{
            FACTOR_DESCRIPTION: u'血小板毒性',
            TRANSFORM_RELATION: RELATION_2
        },
        'AK':{
            FACTOR_DESCRIPTION: u'年龄',
            TRANSFORM_RELATION: None
        },
        'IY':{
            FACTOR_DESCRIPTION: u'粗略分期',
            TRANSFORM_RELATION: RELATION_3
        },
        'IZ':{
            FACTOR_DESCRIPTION: u'腹主动脉旁淋巴结转移',
            TRANSFORM_RELATION: RELATION_4
        },
        'JA':{
            FACTOR_DESCRIPTION: u'盆腔淋巴结转移',
            TRANSFORM_RELATION: RELATION_4
        },
        'BD':{
            FACTOR_DESCRIPTION: u'白细胞',
            TRANSFORM_RELATION: None
        },
        'BE':{
            FACTOR_DESCRIPTION: u'中性粒细胞',
            TRANSFORM_RELATION: None
        },
        'BF':{
            FACTOR_DESCRIPTION: u'淋巴细胞',
            TRANSFORM_RELATION: None
        },
        'BG':{
            FACTOR_DESCRIPTION: u'血红蛋白',
            TRANSFORM_RELATION: None
        },
        'BH':{
            FACTOR_DESCRIPTION: u'血小板',
            TRANSFORM_RELATION: None
        },
        'BI':{
            FACTOR_DESCRIPTION: u'ALT',
            TRANSFORM_RELATION: None
        },
        'BJ':{
            FACTOR_DESCRIPTION: u'Cr',
            TRANSFORM_RELATION: None
        },
        'BK':{
            FACTOR_DESCRIPTION: u'SCC',
            TRANSFORM_RELATION: None
        },
        'BM':{
            FACTOR_DESCRIPTION: u'HBsAg',
            TRANSFORM_RELATION: RELATION_5
        },
        'BN':{
            FACTOR_DESCRIPTION: u'HBsAb',
            TRANSFORM_RELATION: RELATION_5
        },
        'BO':{
            FACTOR_DESCRIPTION: u'HBeAg',
            TRANSFORM_RELATION: RELATION_5
        },
        'BP':{
            FACTOR_DESCRIPTION: u'HbeAb',
            TRANSFORM_RELATION: RELATION_5
        },
        'BQ':{
            FACTOR_DESCRIPTION: u'HBcAb',
            TRANSFORM_RELATION: RELATION_5
        },
        'BR':{
            FACTOR_DESCRIPTION: u'HCV',
            TRANSFORM_RELATION: RELATION_5
        },
        'BS':{
            FACTOR_DESCRIPTION: u'TP',
            TRANSFORM_RELATION: RELATION_5
        },
        'DK':{
            FACTOR_DESCRIPTION: u'先期化疗',
            TRANSFORM_RELATION: RELATION_4
        },
        'DS':{
            FACTOR_DESCRIPTION: u'放疗持续时间',
            TRANSFORM_RELATION: None
        },
        'JB':{
            FACTOR_DESCRIPTION: u'高血压既往史',
            TRANSFORM_RELATION: RELATION_4
        },
        'JC':{
            FACTOR_DESCRIPTION: u'糖尿病既往史',
            TRANSFORM_RELATION: RELATION_4
        },
        'JF':{
            FACTOR_DESCRIPTION: u'各种手术既往史',
            TRANSFORM_RELATION: RELATION_4
        },
        'JG':{
            FACTOR_DESCRIPTION: u'药物过敏既往史',
            TRANSFORM_RELATION: RELATION_4
        }
    }

    _clinical_target = {
        'IT': {
            TARGET_DESCRIPTION: u'复发转移',
            TARGET_FACTORS_GROUP:{
                CATEGORY_NAME['FU_FAN_YING']:['P', 'Q', 'R', 'S', 'T', 'U', 'V'],
                CATEGORY_NAME['XUE_YE_DU_XING']:['AE', 'AF', 'AG', 'AH', 'AK'],
                CATEGORY_NAME['CU_FEN_QI']:['IY'],
                CATEGORY_NAME['ZHUAN_YI']:['IZ','JA'],
                CATEGORY_NAME['XUE_CHANG_GUI']:['BD', 'BE', 'BF', 'BG', 'BH'],
                CATEGORY_NAME['GAN_SHEN_GONG']:['BI', 'BJ'],
                CATEGORY_NAME['ZHONGLIU_BIAOJIWU']:['BK'],
                CATEGORY_NAME['GAN_RAN']:['BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'BS'],
                CATEGORY_NAME['FANG_HUA_LIAO']:['DK','DS'],
                CATEGORY_NAME['JI_WANG_SHI']:['JB', 'JC', 'JF', 'JG']
            }
        },
        'IV': {
            TARGET_DESCRIPTION: u'有肿瘤残留',
            TARGET_FACTORS_GROUP:{}
        },
        'IW': {
            TARGET_DESCRIPTION: u'死亡',
            TARGET_FACTORS_GROUP:{}
        }
    }

    def __init__(self, factors, target, predictor):
        self.target = target
        self.factors = factors
        self.predictor = predictor

    def __setattr__(self, key, value):
        if key=='target':
            assert value in ['IT','IV','IW'], 'clinical target "%s" not match' % (value)
            object.__setattr__(self, key, value)
        if key=='predictor':
            object.__setattr__(self, key, value)

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
            if isinstance(transform_relation, dict):
                return transform_relation.keys()
            return transform_relation
        for k,v in self._clinical_target[self.target][TARGET_FACTORS_GROUP].iteritems():
            factor_category = {}
            factor_category['category_name'] = k
            factor_category['factors'] = []
            for factor in v:
                tmp = {
                    'factor_col':factor,
                    'factor_name':self._risk_factors[factor][FACTOR_DESCRIPTION],
                    'opts':_get_factor_opts(self._risk_factors[factor][TRANSFORM_RELATION])
                }
                factor_category['factors'].append(tmp)
            ret.append(factor_category)
        return ret



    def calculate_risk(self):
        return random.randint(0,100)

    def _logistic_regression_model(self):
        factor = self.factors
        predictor = self.predictor
        return 0.5

    def _tree_model(self):
        factor = self.factors
        predictor = self.predictor
        return 0.5


if __name__ == '__main__':
    engine = rating_engine({}, 'IT', './data/support/lr.pkl')
    print engine.factors
    print engine.target
