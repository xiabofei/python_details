# encoding=utf8

domain_dict_dir = '../../data/input/domain_dict/'

CLRF = '\n'

class AddressDomainDict(object):
    def __init__(self,
                 zhenduan_dict_path,
                 shoushu_dict_path,
                 jianyan_dict_path,
                 yaopin_dict_path,
                 addressed_suffix
                 ):
        self.zhenduan_dict_path = zhenduan_dict_path
        self.shoushu_dict_path = shoushu_dict_path
        self.jianyan_dict_path = jianyan_dict_path
        self.yaopin_dict_path = yaopin_dict_path
        self.addressed_suffix = addressed_suffix

    def _wrapper(self, input_path, output_path, address):
        with open(input_path, 'r') as f_i, open(output_path, 'w') as f_o:
            output_contents = []
            for l in f_i.readlines():
                address(output_contents, l)
            for word in sorted(set(output_contents)):
                f_o.write(word + CLRF)

    def address_zhenduan(self):
        def _zhenduan(output_contents, l):
            for i in l.split(','):
                if i.strip():
                    output_contents.append(i.strip())
        self._wrapper(self.zhenduan_dict_path, self.zhenduan_dict_path + self.addressed_suffix, _zhenduan)

    def address_shoushu(self):
        def _shoushu(output_contents, l):
            for i in l.split(','):
                if i.strip():
                    output_contents.append(i.strip())
        self._wrapper(self.shoushu_dict_path, self.shoushu_dict_path + self.addressed_suffix, _shoushu)

    def address_jianyan(self):
        def _jianyan(output_contents, l):
            output_contents.append(l.strip())
        self._wrapper(self.jianyan_dict_path, self.jianyan_dict_path + self.addressed_suffix, _jianyan)

    def address_yaopin(self):
        def _yaopin(output_contents, l):
            output_contents.append(l.strip())
        self._wrapper(self.yaopin_dict_path, self.yaopin_dict_path + self.addressed_suffix, _yaopin)


if __name__ == '__main__':
    zhenduan_dict = domain_dict_dir + '3_zhenduan'
    shoushu_dict = domain_dict_dir + '4_shoushu'
    jianyan_dict = domain_dict_dir + '5_jianyan'
    yaopin_dict = domain_dict_dir + '6_yaopin'
    addressed_suffix = '_addressed'

    aDD = AddressDomainDict(zhenduan_dict, shoushu_dict, jianyan_dict, yaopin_dict, addressed_suffix)

    aDD.address_zhenduan()
    aDD.address_shoushu()
    aDD.address_jianyan()
    aDD.address_yaopin()
