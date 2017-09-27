#encoding=utf8


from ipdb import set_trace as st
st(context=21)

class iter_test(object):
    def __iter__(self):
        return iter('okay')
        # for i in 'okay':
            # yield i

for i in iter_test():
    print i
