import sys
from utils import *
from pyswip import Prolog

prolog = Prolog()

complete_filename = sys.argv[1]

try:
    prolog.consult(complete_filename)
    consult_ans = list(prolog.query('solve(X)'))[0]['X']
    consult_ans = int(consult_ans)
except:
    print({'message': SYNTAXERROR_CODE})
else:
    print({'message': NOERROR_CODE, 'consult_ans': consult_ans})