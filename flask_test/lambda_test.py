#encoding=utf8

from werkzeug import Local, LocalProxy, LocalStack
# _request_ctx_stack = LocalStack()
l = Local()
print isinstance(l, Local)
request = l('request')
print isinstance(l, Local)
# _request_ctx_stack.push(l)
# request = LocalProxy(lambda: _request_ctx_stack.top.request)

