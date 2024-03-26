import numpy as np

eV = 27.211

def NiceArr(X, NCol=8):
    N = len(X)
    NRow = int(np.ceil(N/NCol))
    if N<=NCol:
        return" ".join(["%7.2f"%(x) for x in X])
    
    Str = ""
    for k in range(NRow):
        if k==(NRow-1): Str += NiceArr(X[(k*NCol):N])
        else: Str += NiceArr(X[(k*NCol):((k+1)*NCol)]) + "\n"
    return Str

def ShowOrbs(XHelp, kmax = 0, Expand=True):
    kl = XHelp.kl
    kmin = max(XHelp.kh-2, 0)
    kmax = max(kl+4, kmax)
    print("Occupied epsilon =")
    print("%s [eV]"%(NiceArr(eV*XHelp.epsilonE[kmin:kl])))
    print("Unoccupied epsilon =")
    print("%s [eV]"%(NiceArr(eV*XHelp.epsilonE[kl:kmax])), flush=True)

    if Expand:
        Over = (XHelp.CE.T).dot(XHelp.Engine.S_ao).dot(XHelp.C0)**2*100.
        for k in range(kmin, kmax):
            Str = "|%3d_E> = "%(k)
            SS = []
            if (k == XHelp.kh+1): print('-'*16)
            for kp in range(kmin, kmax):
                if Over[k,kp]>0.1: SS += ["%5.1f%% |%3d>"%(Over[k,kp], kp)]
            print(Str + " + ".join(SS), flush=True)
