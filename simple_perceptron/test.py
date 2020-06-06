from forward_pass import fp

def test(Xin, Yexp, weights):
    correct = 0
    
    for i in range(len(Xin)):
        output, hot_one = fp(Xin[i], weights)
        if hot_one == Yexp[i]:
            correct = correct + 1

    print(f"In {len(Xin)} tests, there is an accuracy of {correct / len(Xin)}")
