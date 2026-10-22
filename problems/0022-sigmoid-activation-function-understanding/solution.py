import math

def sigmoid(z: float) -> float:
	#Your code here
	expn=math.exp(-z)
	result=1/(1+expn)
	return result 