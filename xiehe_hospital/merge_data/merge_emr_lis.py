#encoding=utf8
from collections import defaultdict
from ipdb import set_trace as st
import json

SPLIT_AMONG_FIELD = '\t'
CRLF = '\n'
SPLIT_INNER_FIELD = '+'

INPUT_PATH = './extracted_data/lis_emr.dat'
OUTPUT_PATH = './extracted_data/merged_lis_emr.dat'

OUTPUT_JSON_A_LINE= 'on'

if __name__ == '__main__':
    # 0954532+10644475    LIS 10473265||3+全血细胞分析    血小板  237
    # 0954532+7478271 EMR 门(急)诊诊断(西医诊断)-疾病编码 宫颈癌
    with open(INPUT_PATH, 'r') as f_input, open(OUTPUT_PATH, 'w') as f_output:

        last_patientID_caseID = ""
        line_content_to_write = {
            'EMR' : {},
            'LIS' : defaultdict(dict)
        }

        for l in f_input.xreadlines():
            try:
                # process fields in the current line
                fields = l.split(SPLIT_AMONG_FIELD)
                patientID_caseID = fields[0].strip()
                lis_or_emr = fields[1].strip()
                # same patientID
                if patientID_caseID==last_patientID_caseID:
                    # if EMR
                    if lis_or_emr=='EMR':
                        emr_item,emr_value = fields[2].strip(),fields[3].strip()
                        line_content_to_write['EMR'][str(emr_item)] = str(emr_value)
                    # if LIS
                    elif lis_or_emr=='LIS':
                        lis_category,lis_item,lis_value = fields[2].strip(),fields[3].strip(),fields[4].strip()
                        line_content_to_write['LIS'][lis_category][str(lis_item)] = str(lis_value)
                    # bad case
                    else:
                        print 'error'
                # new patientID occurs
                else:
                    # if last_patientID_caseID is not empty, then write the last patient's info
                    if last_patientID_caseID!="":
                        f_output.write(last_patientID_caseID+SPLIT_AMONG_FIELD+json.dumps(line_content_to_write, ensure_ascii=False)+CRLF)
                    # update the last_patientID_caseID
                    last_patientID_caseID = patientID_caseID
                    # clear the line_content_to_write
                    line_content_to_write = {
                        'EMR' : {},
                        'LIS' : defaultdict(dict)
                    }
                    # address this line
                    if lis_or_emr=='EMR':
                        emr_item,emr_value = fields[2].strip(),fields[3].strip()
                        line_content_to_write['EMR'][str(emr_item)] = str(emr_value)
                    elif lis_or_emr=='LIS':
                        lis_category,lis_item,lis_value = fields[2].strip(),fields[3].strip(),fields[4].strip()
                        line_content_to_write['LIS'][lis_category][str(lis_item)] = str(lis_value)
                    else:
                        print 'error'
            except Exception,e:
                st(context=21)
                print e

        f_output.write(last_patientID_caseID+SPLIT_AMONG_FIELD+json.dumps(line_content_to_write, ensure_ascii=False)+CRLF)




