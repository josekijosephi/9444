def gradientDescent(Xin, weights, Ycal, Yexp, LR=0.25):
    for i in range(len(weights)):
        for j in range(len(weights[0])):
            # format:                               difference out          sigmoid derivative        Input
            weights[i][j] = weights[i][j] - LR * (Ycal[j] - Yexp[j]) * ( Ycal[j] * ( 1 - Ycal[j] ) ) * Xin[i]
