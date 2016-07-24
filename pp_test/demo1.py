#encoding=utf8
import sys, time
import pp

def sum_primes(n):
    print "hh"
    f(n*1000)

def f(n):
    print n

ppservers = ()

if len(sys.argv)>1:
    ncpus = int(sys.argv[1])
    job_server = pp.Server(ncpus, ppservers=ppservers)
else:
    job_server = pp.Server(ppservers=ppservers)

print "Starting pp with", job_server.get_ncpus(), "workers"

start_time = time.time()
# from ipdb import set_trace as st
# st(context=21)
inputs = (100000, 100100, 100200, 100300, 100400)
jobs = [(input, job_server.submit(sum_primes,(input,), (f,), ())) for input in inputs]
for input, job in jobs:
    print "Sum of primes below", input, "is", job()
print "Time elapsed: ", time.time() - start_time, "s"
job_server.print_stats()
