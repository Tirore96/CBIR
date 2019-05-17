#from features import *
from evaluator import *
from MLMethod import *
cm = ConvolutionMethod(load_prev_model=True)
    
e_ML = Evaluator(method=cm,load_as_volume=False)

print(e_ML.MAP(2,15,["ishealthy"]))