#!/usr/bin/env bash
BASE_PATH='./协和导出数据'
OUTPUT_PATH='./extracted_data/'
PYTHON_BIN='/Users/xiabofei/anaconda2/bin/python'

# replace chinese punctuation 
function replace_chinese_punctuation()
{
    sed -e 'y/（），？：；。、/(),?:;.,/'
}

# 1. extract (PAPMI_Medicare,Adm_Id, Adm_AdmDate) from ADM.txt 
time awk 'FNR>1 {print $1, $2, $4}' ${BASE_PATH}/ADM.txt > ${OUTPUT_PATH}tmp 

# 2. convert 'Adm_AdmDate' to yyyymmdd format and sorted by PAPMI_Medicare then Adm_AdmDate then Adm_Id 
time cat ${OUTPUT_PATH}tmp | awk '{ split($3, arr, "-"); if(length(arr[2])<2){arr[2]="0"arr[2]}; if(length(arr[3])<2){arr[3]="0"arr[3]}; print $1,arr[1]""arr[2]""arr[3],$2}' OFS='\t' | sort -k1 -k2 | awk '{print $1"+"$3,$2}' OFS='\t' | sed '$ d' > ${OUTPUT_PATH}extracted_ADM.dat
echo 'ADM done...'

# 3. extract (PAPMI_Medicare, Adm_Id, LisI_OrdItemID, LisI_ItemName, LisI_LisResult, OrdL_TestSetName ) from LisItem.txt
time cat ${BASE_PATH}/LisItem.txt | tr '\n' ' ' | tr '\r' '\n'| awk 'FNR>1 {gsub(/ /,"", $9);if($3!=""){print $1"+"$3,"LIS",$4"+"$11,$7,$9}}' FS='\t' OFS='\t' | sed -e 's/*//g' -e 's/ //g' | replace_chinese_punctuation > ${OUTPUT_PATH}extracted_LisItem.dat
echo 'LisItem done...'

# 4. extract (PAPMI_Medicare,PAADM_RowID, Name, GetDataByGlossaryCategoryResult) from EMRData.txt
time cat ${BASE_PATH}/EMRData.txt | awk 'FNR>1 {gsub(/^[ \t]+|[ \t]+$/, "", $3); if($3!="-" && $3!="─"){print $4"+"$1, "EMR", $2, $3}}' FS='\t' OFS='\t'| sed -e 's/─//g'| replace_chinese_punctuation > ${OUTPUT_PATH}extracted_EMRData.dat
echo 'EMRData done...'

# 5. merge emr and lis data
time cat ${OUTPUT_PATH}extracted_LisItem.dat ${OUTPUT_PATH}extracted_EMRData.dat| sort -k1 > ${OUTPUT_PATH}lis_emr.dat
echo 'merge LisItem and EMRData done...'

# 6. collect emr and lis of one case of each patient
time ${PYTHON_BIN} ./merge_emr_lis.py
echo 'collect EMR and LIS of one case of each patient done...'

# 7. join adm and merged emr and lis data
time join -a1 -t$'\t' ${OUTPUT_PATH}extracted_ADM.dat ${OUTPUT_PATH}merged_lis_emr.dat > ${OUTPUT_PATH}merged_adm_lis_emr.dat
echo 'join adm date with lis and emr done...'

# 8. produce result data
time ${PYTHON_BIN} ./merge_date.py
echo 'merge date info done...'

# clear up tmp file
rm ${OUTPUT_PATH}tmp
