import sys,dill
f = dill.loads(sys.argv[0])
datain = dill.loads(sys.argv[1])
f(datain)