import subprocess
import os
import sys

def f():
    sys.exit(3)

if os.environ.get('A_NEW_ENVIRON') == 'true':
    f()
else:
    args = [sys.executable] + sys.argv
    new_environ = os.environ.copy()
    new_environ['A_NEW_ENVIRON'] = 'true'
    exit_code = subprocess.call(args, env=new_environ)
    print "exit code is %s" % exit_code
