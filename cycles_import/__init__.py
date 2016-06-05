#encoding=utf8
_submodule_exports = {
        '.foo' : ['Foo'],
        '.bar' : ['Bar']
        }

_submodule_by_name = {
        name: modulename 
        for modulename in _submodule_exports
        for name in _submodule_exports[modulename] }

import types
import sys
import importlib


class OnDemandModule(types.ModuleType):
    def __getattr__(self, name):
        modulename = self._submodule_by_name.get(name)
        if modulename:
            module = self.importlib.import_module(modulename, self.__package__)
            print('Loaded', name)
            value = getattr(module, name)
            setattr(self, name ,value)
            return value
        raise AttributeError('No attribute %s' % name)
# old_ref = sys.modules[__name__]
newmodule = OnDemandModule(__name__)
newmodule.__dict__.update(globals())
sys.modules[__name__] = newmodule
