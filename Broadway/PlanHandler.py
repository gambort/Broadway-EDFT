import numpy as np
import scipy.linalg as la

zero_round = 1e-5

#############################################################################
# Show occupations nicely
def Showf(f):
    if np.sum(np.abs(f - np.round(f)))<zero_round:
        return "[" + ";".join(["%d"%(int(x)) for x in np.round(f)]) + "]"
    else:
        return "[" + ";".join(["%.2f"%(x) for x in f]) + "]"

# Show a plan nicely
def ShowPlan(Plan):
    Str = "1RDM : %s\nkTo: %s\n"%(Showf(Plan['1RDM']), Showf(Plan['kTo']))
    for Key in ('Hx', 'xcDFA'):
        if not(Key in Plan): continue

        if len(Plan[Key])==0:
            continue
        
        Str += "%s:\n"%(Key)
        for WE, FE, f in Plan[Key]:
            Str += "  (%7.4f, %7.4f, %s)\n"%(WE, FE, Showf(f))

    if not('Extra' in Plan) \
       or (Plan['Extra'] is None) or len(Plan['Extra'])==0: return Str
        
    Str += "Extra:\n"
    for Kind_ in ('J', 'K', 'K_w'):
        for WE, FE, k1, k2, Kind in Plan['Extra']:
            if Kind == Kind_:
                Str += "  (%7.4f, %7.4f, %3d, %3d, \'%s\')\n"%(WE, FE, k1, k2, Kind)
    return Str

#############################################################################
# Created a promoted occ
# NDbl is either int (number double occupied orbitals)
# or an f matrix
def PromoteOcc(NDbl, kFrom, kTo, S=1., N=None):
    # N is the target length, if larger than kTo+1
    if N is None: 
        f = np.zeros((kTo+1,))
    else:
        f = np.zeros((max(N, kTo+1),))

    try:
        f[:len(NDbl)] = NDbl
    except:
        f[:NDbl] = 2.
        
    f[kFrom] -= S
    f[kTo] += S

    return f

#############################################################################
# Call this to tidy up a plan
def TidyPlan(Plan, Reverse=False):
    return MixPlans(1., Plan, Reverse=Reverse)

#############################################################################
# Call this to make a plan for energy only
# * note, expands out k1, k2 and k2, k1
def EnergyPlan(Plan_):
    # Get rid of excess reversals
    Plan = TidyPlan(Plan_, Reverse=True)

    # Prepart the output
    PlanOut = {}
    # kTo is meaningless for energy
    PlanOut['kTo']=()
    # 1RDM stays the same
    PlanOut['1RDM']=Plan['1RDM']
    # Singlet stays the same
    PlanOut['Singlet']=Plan['Singlet']

    
    # Copy out the Hx and xcDFA parts but nullify
    for Key in ('Hx', 'xcDFA'):
        PlanOut[Key] = []
        for WE, _, fE in Plan[Key]:
            PlanOut[Key] += [(WE, 0., fE)]

    # Split every Extra of k1\neq k2 into two parts k1->k2 and k2->k1
    PlanOut['Extra'] = []
    for WE, _, k1, k2, Kind in Plan['Extra']:
        if k1==k2:
            PlanOut['Extra'] += [(WE, 0., k1, k2, Kind)]
        else:
            PlanOut['Extra'] += [(WE/2, 0., k1, k2, Kind)]
            PlanOut['Extra'] += [(WE/2, 0., k2, k1, Kind)]

    return PlanOut
            
    

#############################################################################
# Call this to form ensembles
def MixPlans(alpha1, Plan1, alpha2=0., Plan2=None, Reverse=False):
    if Plan2 is None:
        Plan2 = {'kTo':(), 'Singlet':False,
                 '1RDM':[0.,], 'Hx':[], 'xcDFA':[], 'Extra':[]}
    
    # Add f of different lengths
    def Addf(x1, x2):
        if len(x1)<len(x2): return Addf(x2,x1)
        elif len(x1)==len(x2): return x1+x2

        xr = 1.*x1
        xr[:len(x2)] += x2
        return xr

    # Check if two f are same
    def Samef(x1, x2):
        return np.sum(np.abs(Addf(x1, -x2)))<zero_round

    # Scale an f based list
    def Scale(alpha, EFfList):
        return [ (x[0]*alpha, x[1]*alpha, x[2]) for x in EFfList ]

    # Scale an extra list
    def ScaleExtra(alpha, Extra):
        if Extra is None: return []
        return [ (x[0]*alpha, x[1]*alpha, x[2], x[3], x[4]) for x in Extra ]

    # Remvoe duplicates from an f based list
    def Condense(EFfList):
        Pairs = {}
        Done = []
        for k in range(len(EFfList)):
            Pairs[k] = []
            for kp in range(k, len(EFfList)):
                if kp in Done: continue
                
                f  = EFfList[k][2]
                fp = EFfList[kp][2]
                if Samef(f, fp):
                    Pairs[k] += [kp,]
                    Done += [kp,]

        Out = []
        for k in Pairs:
            if len(Pairs[k])==0: continue
            
            EW, FW = 0., 0.
            f = EFfList[k][2]
            for kp in Pairs[k]:
                EW += EFfList[kp][0]
                FW += EFfList[kp][1]
            Out += [(EW, FW, f)]
        return Out          

    # Remvoe duplicates from an Extra based list
    def CondenseExtra(Extra):
        Pairs = {}
        Done = []
        for k in range(len(Extra)):
            Pairs[k] = []
            for kp in range(k, len(Extra)):
                if kp in Done: continue

                AreSame = (Extra[k][4] == Extra[kp][4]) \
                    and (Extra[k][2] == Extra[kp][2]) \
                    and (Extra[k][3] == Extra[kp][3])

                if Reverse:
                    AreSameR = (Extra[k][4] == Extra[kp][4]) \
                        and (Extra[k][2] == Extra[kp][3]) \
                        and (Extra[k][3] == Extra[kp][2])

                    AreSame = AreSame or AreSameR
                
                if AreSame:
                    Pairs[k] += [kp,]
                    Done += [kp,]

        Out = []
        for k in Pairs:
            if len(Pairs[k])==0: continue
            
            EW, FW = 0., 0.
            for kp in Pairs[k]:
                EW += Extra[kp][0]
                FW += Extra[kp][1]
            Out += [(EW, FW, Extra[k][2], Extra[k][3], Extra[k][4])]
        return Out

    Plan = {}

    # Do kTo
    # Use set to ensure uniqeness
    Plan['kTo'] = tuple(set(list(Plan1['kTo']) + list(Plan2['kTo'])))

    # 1RDM is a trivial average
    Plan['1RDM'] = Addf(alpha1*np.atleast_1d(Plan1['1RDM']),
                        alpha2*np.atleast_1d(Plan2['1RDM']))

    # Singlet is forced if one ingredient is singlet
    Plan['Singlet'] = Plan1['Singlet'] or Plan2['Singlet']

    Plan['Hx'] = Condense( Scale(alpha1, Plan1['Hx'])
                           + Scale(alpha2, Plan2['Hx']) )
    Plan['xcDFA'] = Condense( Scale(alpha1, Plan1['xcDFA'])
                               + Scale(alpha2, Plan2['xcDFA']) )
    
    # 'Extra': [(PreE, PreF, k1, k2, Kind), ...] or None
    Plan['Extra'] = CondenseExtra( ScaleExtra(alpha1, Plan1['Extra'])
                                   + ScaleExtra(alpha2, Plan2['Extra']) )

    return Plan
        

#############################################################################
# Form a plan for orbital k
def PlanDeriv(Plan_, k, IsEnergy=False):
    if not(IsEnergy):
        Plan = EnergyPlan(Plan_)
    else:
        Plan = Plan_
        
    # Check if k is unoccupied (either too large of value of 0)
    if (k>=len(Plan['1RDM'])) \
       or (Plan['1RDM'][k] == 0.):
        print("Cannot take a derivative on an unoccupied orbitaql")
        return None

    fk = Plan['1RDM'][k]

    Plan_fk = {}
    Plan_fk['1RDM'] = Plan['1RDM']
    Plan_fk['kTo']  = Plan['kTo']
    Plan_fk['Singlet'] = Plan['Singlet']

    # Functionals of f are 1RDM
    for Key in ('Hx', 'xcDFA'):
        Plan_fk[Key] = []
        for WEP, _, fP in Plan[Key]:
            if k>=len(fP): fPk = 0.
            else: fPk = fP[k]
        
            FEP = WEP * fPk/fk
            Plan_fk[Key] += [(WEP, FEP, fP)]

    
    # Extra must be handled carefully
    Plan_fk['Extra'] = []
    for PreE, _, k1, k2, Kind in Plan['Extra']:
        if k1 == k2:
            if (k1==k): PreF = 2.*PreE / fk
            else: PreF = 0.
            Plan_fk['Extra'] += [(PreE, PreF, k1, k2, Kind)]
        else:
            if k2==k:
                Plan_fk['Extra'] += [(PreE, 2.*PreE / fk, k1, k2, Kind)]
            else:
                Plan_fk['Extra'] += [(PreE, 0., k1, k2, Kind)]

    #return Plan_fk
    return Plan_fk

#############################################################################
# The PlanHandler class automates some aspects of ensemble
# generation.
#############################################################################

class PlanHandler(dict):
    def __init__(self, Content=None, NEl=2, ):
        if not(Content is None):
            self.Content = Content           
        else:
            if np.abs(NEl - np.round(NEl))<zero_round:
                NEl = int(np.round(NEl))
                if NEl%2==1:
                    print("Odd electron number not implemented")
                    quit()
                else:
                    f = 2.*np.ones((NEl//2,))
                    self.Content = { 'kTo': (NEl//2,),
                                     'Singlet': True,
                                     '1RDM': f, 'Hx': [[1., 1., f],],
                                     'xcDFA': [[1., 1., f],], 'Extra':None }
            else:
                print("Fractional electron number not implemented")
                quit()

    def __getitem__(self, ID):
        if ID in self.Content: return self.Content[ID]
        else: return None

    def SetExtra(self, Extra, WithHalf=True):
        if WithHalf:
            self.Content['Extra'] = Extra
        else:
            self.Content['Extra']  = []
            for WE, WF, k1, k2, Kind in Extra:
                self.Content['Extra'] += [(WE/2., WF/2., k1, k2, Kind)]

        return self

    def Tidy(self):
        self.Content = EnergyPlan(self.Content)
        return self

    def Show(self):
        return ShowPlan(self.Content)


    def FromOcc(self, f, Triplet=False,
                epsilon=None):
        # Total number of electrons and as doubly occupied
        NTot = np.sum(f)
        NDOcc = int(np.floor(NTot/2.))
        
        # Get the number of occupied orbitals and trim
        ft = np.array(f, dtype=float)
        NOcc = 0
        for k in range(len(ft)):
            if ft[k]>zero_round: NOcc = k+1
        ft = ft[:NOcc]

        self.Content['1RDM'] = ft*1.
        self.Content['kTo'] = (NOcc-1,)
        
        # Build the Hx and xcDFA approximation
        fRegular = []
        fr = 1.*ft
        for AllOcc in range(NOcc, NDOcc-1, -1):
            funp = 0.*ft
            funp[ :AllOcc ] = 1.
            NRem = NDOcc * 2 - AllOcc
            funp[ :NRem ] = 2.

            W = fr[AllOcc-1]/funp[AllOcc-1]
            fr -= W*funp

            if np.abs(W)>zero_round:
                fRegular += [[W, 0., funp*1.]]

        if np.abs(np.sum(fr))>zero_round:
            print("Warning! The occupation handler isn't designed for this much virtual")
            fRegular += [[-1., 0., fr*1.]]

        for Key in ('Hx', 'xcDFA'):
            self.Content[Key] = []
            for H in fRegular:
                self.Content[Key] += [H]

    

        if epsilon is None:
            print("Note, cannot build extra without epsilon")
            self.Content['Extra'] = None
            return self

        # Automation only possible for double occupations or
        # two unpaired orbitals

        # Count the number unpaired
        Unpaired = np.sum(ft * (2-ft))


        # Get a list of all 'available' orbitals
        kAvailable = np.argwhere(ft<(2.-zero_round)).reshape((-1,))

        print("Note, extras not yet implemented")
        self.Content['Extra'] = None
        return self

        
        if Unpaired == 2.:
            # Two unpaired orbitals
            1
        elif Unpaired == 0.:
            # No unpaired orbitals
            1
        else:
            print("Note, can only build extra for paired or two unpaired orbitals")
            self.Content['Extra'] = None
            return self
            


        return self
        

        

        
        
        

#############################################################################

if __name__ == "__main__":
    print("="*72)
    kh = 2
    kl = 3

    f_gs = 2.*np.ones((kh+1,))
    f_ts = PromoteOcc(f_gs, kh, kl, 1.)

    Plan = {
        '1RDM' : f_ts,
        'kTo': (kl,),
        'Hx'   : [ (1., 0., f_ts), ],
        'xcDFA': [ (1., 0., f_ts), ],
        'Extra': [ ( 2., 0., kh , kl , 'K'), ]
    }

    print("Original plan")
    print(ShowPlan(Plan))
    print("Tidied plan (Reverse=False)")
    print(ShowPlan(TidyPlan(Plan, Reverse=False)))
    print("Energy plan")
    print(ShowPlan(EnergyPlan(Plan)))
    
    for k in range(kl+1):
        print("Plan_Deriv (for k=%d)"%(k))
        print(ShowPlan(PlanDeriv(Plan, k)))
    
    print("="*72)
    kh = 2
    kl1 = 3
    kl2 = 4

    f_gs = 2.*np.ones((kh+1,))
    f_ts1 = PromoteOcc(f_gs, kh, kl1, 1., N=kl2+1)
    f_ts2 = PromoteOcc(f_gs, kh, kl2, 1., N=kl2+1)
    f_dx = PromoteOcc(f_ts1, kh, kl2, 1.)

    Plan = {
        '1RDM' : f_dx,
        'kTo': (kl1,kl2),
        'Hx'   : [ (1., 0., f_ts1), (1., 0., f_ts2), (-1., 0., f_gs) ],
        'xcDFA': [ (1., 0., f_ts1), (1., 0., f_ts2), (-1., 0., f_gs) ],
        'Extra': [ ( 1., 0., kh , kh , 'J'),
                   ( 1., 0., kl1, kl2, 'J'),
                   (-1., 0., kh , kl1, 'J'),
                   (-1., 0., kh , kl2, 'J'),
                   
                   ( 2., 0,  kl1, kl2, 'K'),
                   ( 1., 0,  kh , kl1, 'K'),
                   ( 1., 0,  kh , kl2, 'K'),
        ]                   
    }

    print("Original plan")
    print(ShowPlan(Plan))
    print("Tidied plan (Reverse=False)")
    print(ShowPlan(TidyPlan(Plan, Reverse=False)))
    print("Energy plan")
    print(ShowPlan(EnergyPlan(Plan)))
    
    Plan_D1 = PlanDeriv(Plan, kl1)
    Plan_D2 = PlanDeriv(Plan, kl2)

    Plan_D = MixPlans(0.5, Plan_D1, 0.5, Plan_D2)

    print("Plan_D1 (for k=%d)"%(kl1))
    print(ShowPlan(Plan_D1))
    print("Plan_D2 (for k=%d)"%(kl2))
    print(ShowPlan(Plan_D2))


    #####################################
    print("="*72)
    NewPlan = PlanHandler(Plan)
    print(NewPlan['1RDM'])

    print( PlanHandler().FromOcc(f=[2,2,2,0,2]).Show() )
    print( PlanHandler().FromOcc(f=[2,2,2,1,1]).Show() )
    print( PlanHandler().FromOcc(f=[2,1,2,0,1]).Show() )
