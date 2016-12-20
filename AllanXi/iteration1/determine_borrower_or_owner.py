import sys
import json

# Mapping original code in ./Ascii/DAYV2PUB.CSV to '1' or '0' 
# '1' for 'YES'
# '0' for 'NO'
HHMEMDRV = {
        '01' : '1',
        '02' : '0'
        }
DRVR_FLG = {
        '01' : '1',
        '02' : '0'
        }
TRPHHVEH = {
        '01' : '1',
        '02' : '0'
        }
# bit code explain info
BITCODE_EXPLAIN = {
        '000' : 'HHMEMDRV:NO & DRVR_FLG:NO  & TRPHHVEH:NO',
        '001' : 'HHMEMDRV:NO & DRVR_FLG:NO  & TRPHHVEH:YES',
        '010' : 'HHMEMDRV:NO & DRVR_FLG:YES  & TRPHHVEH:NO',
        '011' : 'HHMEMDRV:NO & DRVR_FLG:YES  & TRPHHVEH:YES',
        '100' : 'HHMEMDRV:YES & DRVR_FLG:NO  & TRPHHVEH:NO',
        '101' : 'HHMEMDRV:YES & DRVR_FLG:NO  & TRPHHVEH:YES',
        '110' : 'HHMEMDRV:YES & DRVR_FLG:YES  & TRPHHVEH:NO',
        '111' : 'HHMEMDRV:YES & DRVR_FLG:YES  & TRPHHVEH:YES'
        }
# The dict defined following serves as a global data structure for statistic distribution
# Three bits code key indicate whether "HHMEMDRV" "DRVR_FLG" "TRPHHVEH" is 'YES' or 'NO' respectively
# e.g
#   keys in dict
#       '100' = HHMEMDRV is 'YES' & DRVR_FLG is 'NO'  & TRPHHVEH is 'NO'
#       '010' = HHMEMDRV is 'NO'  & DRVR_FLG is 'YES' & TRPHHVEH is 'NO'
#       '001' = HHMEMDRV is 'NO'  & DRVR_FLG is 'NO'  & TRPHHVEH is 'YES'
#   values in dict
#       the total sum of trips of each situation
DISTRIBUTION_SUM = {
        '000' : 0,
        '001' : 0,
        '010' : 0,
        '011' : 0,
        '100' : 0,
        '101' : 0,
        '110' : 0,
        '111' : 0
        }
# The dict defined following serves as a reference dictionary
# Three bits code key mapping whether DRVR_TYPE is 'borrower' or 'owner'
# e.g
#   keys in dict
#   '100' = HHMEMDRV is 'YES' & DRVR_FLG is 'NO'  & TRPHHVEH is 'NO'
#   '010' = HHMEMDRV is 'NO'  & DRVR_FLG is 'YES' & TRPHHVEH is 'NO'
#   '001' = HHMEMDRV is 'NO'  & DRVR_FLG is 'NO'  & TRPHHVEH is 'YES'
#   values in dict
#       0 : owner
#       1 : borrower
BORROWER_OR_OWNER = {
        '000' : 0,
        '001' : 1,
        '010' : 0,
        '011' : 0,
        '100' : 0,
        '101' : 1,
        '110' : 1,
        '111' : 0
        }

def determine_DRVR_TYPE(hhmemdrv, drvr_flg, trphhveh):
    """
    Description
    -----------
        1. Determine whether the trip was completed in an owned or borrowed vehicle
        2. Update the DISTRIBUTION_SUM

    Parameter
    ---------
        hhmemdrv:
            str
            category type string
            '01' : YES
            '02' : NO
            others : IGNORE

        drvr_flg:
            str
            category type string 
            '01' : YES
            '02' : NO
            others : IGNORE
        
        trphhveh:
            str
            category type string
            '01' : YES
            '02' : NO
            others : IGNORE
    
    Return
    ------
        int
        category type string
        0  : Borrower
        1  : Owner
        -1 : Ignore
        -2 : Exception
    """
    try:
        bits_code = []
        if (hhmemdrv in HHMEMDRV.keys()) and (drvr_flg in DRVR_FLG.keys()) and (trphhveh in TRPHHVEH.keys()):
            bits_code.append(HHMEMDRV[hhmemdrv])
            bits_code.append(DRVR_FLG[drvr_flg])
            bits_code.append(TRPHHVEH[trphhveh])
            bits_code = "".join(bits_code)
            DISTRIBUTION_SUM[bits_code] += 1;
            return BORROWER_OR_OWNER[bits_code]
        return -1
    except Exception,e:
        return -2


if __name__ == '__main__':
    HHMEMDRV_COLNUM = 42
    DRVR_FLG_COLNUM = 38
    TRPHHVEH_COLNUM = 73
    CSV_SPLIT = ','
    with open('./Ascii/DAYV2PUB.CSV','r') as f_input, open('dataset_output.csv', 'w') as f_output:
        for line in f_input:
            items = line.strip().split(CSV_SPLIT)
            try:
                hhmemdrv = items[HHMEMDRV_COLNUM].strip()
                drvr_flg = items[DRVR_FLG_COLNUM].strip()
                trphhveh = items[TRPHHVEH_COLNUM].strip()
                drvr_type = determine_DRVR_TYPE(hhmemdrv, drvr_flg, trphhveh)
                # ignore those rows do not have 'YES' or 'NO' response
                if drvr_type==1 or drvr_type==0:
                    items.append(str(drvr_type))
                    f_output.write(CSV_SPLIT.join(items)+'\n')
                else:
                    # column name row
                    if hhmemdrv=='HHMEMDRV' and drvr_flg=='DRVR_FLG' and trphhveh=='TRPHHVEH':
                        items.append('DRVR_TYPE')
                        f_output.write(CSV_SPLIT.join(items)+'\n')
            except Exception,e:
                continue
    with open('statistic_output.dat', 'w') as f_statistic_output:
        statistic_content = {}
        for k,v in BITCODE_EXPLAIN.items():
            statistic_content[k] = { DISTRIBUTION_SUM[k]:v }
        f_statistic_output.write(json.dumps(statistic_content, indent=4 ,sort_keys=True))

