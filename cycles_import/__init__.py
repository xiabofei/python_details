#encoding=utf8
_submodule_exports = {
        '.foo' : ['Foo'],
        '.bar' : ['Bar']
        }

_submodule_by_name = {
        name: modulename 
        for modulename in _submodule_exports
        for name in _submodule_exports[modulename] }

print id(_submodule_by_name)

import types
import sys
import importlib


class OnDemandModule(types.ModuleType):
    def __getattr__(self, name):
        print dir(self)
        print id(self._submodule_by_name)
        modulename = self._submodule_by_name.get(name)
        if modulename:
            print '__package__:'+str(__package__)
            print 'self.__packge__:'+self.__package__
            module = self.importlib.import_module(modulename, self.__package__)
            print('Loaded', name)
            value = getattr(module, name)
            setattr(self, name ,value)
            return value
        raise AttributeError('No attribute %s' % name)

# 添加一个原有module的引用 避免garbage collection
# 猜测这里应该是保留原有module中定义的变量
# 避免更新sys.modules带来的将原有module有关的变量都置成None的side-effect
old_ref = sys.modules[__name__]
# print id(sys.modules[__name__])
# print id(old_ref)

# 这里生成一个对象的示例 因此globals()中的变量都update到了实例中
# 因此_submodule_by_name虽然还是原来的那个变量(id没有变) 但是已经
# 不能用全局变量的形式访问了 得用self.xxx的形式访问
newmodule = OnDemandModule(__name__)
print id(globals()['_submodule_by_name'])
newmodule.__dict__.update(globals())
# newmodule.__all__ = list(_submodule_by_name)
# 执行完这一步之后 原来产生的module已经被取代了 访问原有module的各种变量都是None
# 具体可以看stackoverflow的解释:
# http://stackoverflow.com/questions/5365562/why-is-the-value-of-name-changing-after-assignment-to-sys-modules-name
sys.modules[__name__] = newmodule
# maintain_module = sys.modules[__name__]
