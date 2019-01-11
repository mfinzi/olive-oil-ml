import sys,dill
f = dill.loads(sys.argv[1])
datain = dill.loads(sys.argv[2])
f(datain)