import math

def softmax(scores: list[float]) -> list[float]:
    m=max(scores)

    exp_e=[math.exp(s-m) for s in scores]


    tot=sum(exp_e)


    return [i/tot for i in exp_e]
