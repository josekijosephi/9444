import csv

def get_csv(fname):
    Xinputs = []
    Yout = []
    with open(fname) as f:
        reader = csv.reader(f)
        for line in reader:
            try:
                Xinputs.append([float(line[2]),float(line[3])])
                Yout.append(line[4])
            except:
                pass

    return Xinputs, Yout
                                                                        
