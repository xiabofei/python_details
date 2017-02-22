#encoding=utf8
"""
http://stackoverflow.com/questions/4770297/python-convert-utc-datetime-string-to-local-datetime
search 'python dateutil parse utc to pst'
"""
from datetime import datetime
from dateutil import tz

def convert_utc_to_pst(origin_datetime, input_format, output_format):
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('US/Pacific')
    utc = datetime.strptime(origin_datetime, input_format)
    utc = utc.replace(tzinfo=from_zone)
    pst = utc.astimezone(to_zone)
    return pst.strftime(output_format)

if __name__=='__main__':
    ret = convert_utc_to_pst(\
            '2017-02-19T03:35:51',\
            '%Y-%m-%dT%H:%M:%S',\
            '%m/%d/%Y %H:%M:%S')
    print ret 
    



