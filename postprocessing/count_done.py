import os 
import sys


path = "../results_blackbox"
if len(sys.argv) > 1:
  path = sys.argv[1]


os.chdir(path)
c = 0
for d in os.listdir("."):
  for f in os.listdir(d):
    #print(d)
    c += 1

print(c)
