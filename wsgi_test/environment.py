#encoding=utf8
"""
用python自带的wsgi框架, 并返回环境变量信息
"""
from ipdb import set_trace as st
from wsgiref import simple_server

def application(envrion, start_response):
    response_body = [
            '%s: %s' % (k,v) for k,v in sorted(envrion.items())
            ]
    response_body = '\n'.join(response_body)

    status = '200 OK'
    response_header = [
            ('Content-Type', 'text/plain'),
            ('Content-Length', str(len(response_body)))
            ]
    start_response(status, response_header)
    return [response_body]

st(context=21)
httpd = simple_server.make_server('127.0.0.1', 5000, application)

httpd.handle_request()
httpd.handle_request()
print 'end'
