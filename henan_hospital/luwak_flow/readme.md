自动化挖掘框架-数据特征预处理部分

1. 框架代码文件定位
    ./luwak.py : 主执行引擎
    ./convert.py : 集成了项目中常见的数据特征转换的功能(1 to 1, 1 to N, N to 1, one hot encoder, column filter, row filter)
    ./eat_io.py : 集成了文件流的读写功能
    ./config.py : 定义了全局常值变量
    ./utils.py : 工具函数

2. 框架使用
    1) 使用框架高效地完成挖掘工作的数据特征预处理, 只需要按照框架规范执行简单的配置即可, 尽可能减少重复工作
    2) 请参照./feed.py中的两个demo('协和宫颈癌'和'河南体检')
