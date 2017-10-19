# encoding=utf8

from ipdb import set_trace as st

def reset_parameter(**kwargs):
    def callback(env):
        new_parameters = {}
        for key, value in kwargs.items():
            if key in ['num_class', 'boosting_type', 'metric']:
                raise RuntimeError("cannot reset {} during training".format(repr(key)))
            if isinstance(value, list):
                if len(value) != env.end_iteration - env.begin_iteration:
                    raise ValueError("Length of list {} has to equal to 'num_boost_round'.".format(repr(key)))
                new_param = value[env.iteration - env.begin_iteration]
            else:
                new_param = value(env.iteration - env.begin_iteration)
            if new_param != env.params.get(key, None):
                new_parameters[key] = new_param
        if new_parameters:
            env.model.reset_parameter(new_parameters)
            env.params.update(new_parameters)

    callback.before_iteration = True
    callback.order = 10
    return callback
