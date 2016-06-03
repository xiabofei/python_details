#encoding=utf8
"""
测试Python logging的用法
"""
import logging as lg
# 生成LOG输出信息接口
LOG1 = lg.getLogger('b.c')
LOG2 = lg.getLogger('d.e')
# 处理信息输出
filehandler = lg.FileHandler('test.log','a')
# 信息输出格式
formatter = lg.Formatter('%(name)s %(asctime)s %(levelname)s %(message)s')
filehandler.setFormatter(formatter)
# 信息输出过滤
# filter = lg.Filter('b')
# filehandler.addFilter(filter)
LOG1.addHandler(filehandler)
LOG2.addHandler(filehandler)
LOG1.setLevel(lg.INFO)
LOG2.setLevel(lg.DEBUG)
# 测试LOG1五个级别的输出
LOG1.debug('it is a debug message for log1')
LOG1.info('norma info message for log1')
LOG1.warning('warning message for log1:b.c')
LOG1.error('error message for log1:abcd')
LOG1.critical('critical message for log1:not worked')
# 测试LOG2五个级别的输出
LOG2.debug('it is a debug message for log2')
LOG2.info('norma info message for log2')
LOG2.warning('warning message for log2')
LOG2.error('error:b.c')
LOG2.critical('critical')
