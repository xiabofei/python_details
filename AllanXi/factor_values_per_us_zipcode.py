#coding=utf8
import requests
import re
import json
import sys

__all__ = ['get_population_density', 'get_racial_homogeneity']

zipcode_density_dict = {}
with open('./Zipcode-ZCTA-Population-Density-And-Area-Unsorted.csv','r') as f:
    for l in f.readlines():
        items = l.strip().split(',')
        if len(items)==4:
            zipcode_density_dict[items[0]] = items[3]

def get_population_density(zipcode):
    """Get population density according to the Zipcode-ZCTA-Population-Density

    Parameters
    ----------
    zipcode : str
              The input zipcode

    Returns
    -------
    float
        population density corresponding to the input zipcode
    """
    return zipcode_density_dict.get(str(zipcode),'not valid zipcode')


def get_racial_homogeneity(zipcode):
    """Get raical homogeneity from http://zipwho.com
    
    Parameters
    ----------
    zipcode : str
              The input zipcode

    Returns
    -------
    json
        input 94577 return 
        {"median_income': {"val": 50888, "national_percentage_rank": 76}, "cost_of_living_index" : {"val": 256.1, "national_percentage_rank": 93}, ... , }
    """
    ret = {}
    # target key mapping 
    SPLIT = '.'
    VALUE = SPLIT + 'val'
    RANK = SPLIT + 'national_percentage_rank'
    key_mapping_dict = {
            'AsianPercent':'asian'+VALUE,
            'AsianRank':'asian'+RANK,
            'AverageHouseholdSize':'average_household_size'+VALUE,
            'AverageHouseholdSizeRank':'average_household_size'+RANK,
            'BlackPercent':'black'+VALUE,
            'BlackRank':'black'+RANK,
            'CollegeDegreePercent':'college_degree'+VALUE,
            'CollegeDegreeRank':'college_degree'+RANK,
            'CostOfLivingIndex':'cost_of_living_index'+VALUE,
            'CostOfLivingRank':'cost_of_living_index'+RANK,
            'DivorcedPercent':'divorced'+VALUE,
            'DivorcedRank':'divorced'+RANK,
            'HispanicEthnicityPercent':'hispanic_ethnicity'+VALUE,
            'HispanicEthnicityRank':'hispanic_ethnicity'+RANK,
            'MaleToFemaleRatio':'male_to_female'+VALUE,
            'MaleToFemaleRank':'male_to_female'+RANK,
            'MarriedPercent':'married'+VALUE,
            'MarriedRank':'married'+RANK,
            'MedianAge':'median_age'+VALUE,
            'MedianAgeRank':'median_age'+RANK,
            'MedianIncome':'median_income'+VALUE,
            'MedianIncomeRank':'median_income'+RANK,
            'MedianMortgageToIncomeRatio':'median_mortagage_to_income_ratio'+VALUE,
            'MedianMortgageToIncomeRank':'median_mortagage_to_income_ratio'+RANK,
            'MedianRoomsInHome':'median_rooms_in_home'+VALUE,
            'MedianRoomsInHomeRank':'median_rooms_in_home'+RANK,
            'OwnerOccupiedHomesPercent':'owner_occupied_homes'+VALUE,
            'OwnerOccupiedHomesRank':'owner_occupied_homes'+RANK,
            'Population':'population'+VALUE,
            'PopulationRank':'population'+RANK,
            'ProfessionalPercent':'professional'+VALUE,
            'ProfessionalRank':'professional'+RANK,
            'WhitePercent':'white'+VALUE,
            'WhiteRank':'white'+RANK,
            'city':'city',
            'state':'state',
            'zip':'zip'
            }
    # regex for extract js returun value
    reg = re.compile(r"getData\(\)\{(.*?)\}")
    url = "http://zipwho.com/?zip="+str(zipcode)+"&city=&filters=--_--_--_--&state=&mode=zip"
    # proxies for anti-web-crawler
    proxies = {
            "http":"http://40.114.5.134:8118/",
            "http":"http://47.88.8.215:8118/",
            "http":"http://221.195.55.182:8080/"
            }
    proxies = {}
    response = requests.get(url,proxies=proxies)
    if response.status_code==200:
        text = response.text.replace("\r","").\
                replace("\n","").\
                replace("\t","")
        context = reg.search(text)
        if context != None:
            context = reg.search(text).group(1).\
                    replace("\"","").\
                    replace("\\n","#").\
                    replace("return","").\
                    replace(";","")
            keys_values = context.strip().split('#')
            if len(keys_values)==2:
                keys = context.strip().split('#')[0].split(',')
                values = context.strip().split('#')[1].split(',')
                assert len(keys)==len(values),\
                        "length of keys %s and values %s not match "%(len(keys),len(values))
                for k,v in zip(keys,values):
                    tmp = key_mapping_dict[k].split('.')
                    if len(tmp)==2:
                        if tmp[0] not in ret.keys():
                            ret[tmp[0]] = {}
                        ret[tmp[0]][tmp[1]] = v
                    else:
                        ret[k] = v
    return json.dumps(ret, indent=4, sort_keys=True)

if __name__ == '__main__':
    assert len(sys.argv)==2, "sys.args length %s not match"%(len(sys.argv))
    print get_population_density(sys.argv[1])
    print get_racial_homogeneity(sys.argv[1])


