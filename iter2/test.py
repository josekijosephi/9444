from forwardpass import forwardPass
from numpy import array_equal

def test(Xin, Yexp, weights, splice):
    count = 0

    for i in range(splice, splice + len(Xin)):
        output, hot_one = forwardPass(Xin.loc[i], weights)
        if array_equal(hot_one, Yexp.loc[i]):
            count += 1

    print(f"In {len(Xin)} tests, {count / len(Xin)}% accuracy.")
