import numpy as np
import scipy.linalg as la

from Broadway.LDAFits import *
from Broadway.PlanHandler import *

eV = 27.211

zero_round = 1e-5

###############################################################################
# Top level class for EDFT calculations
#
# This class does not have an optimizer, but provides the routines
# that are used by the different optimizers
#
# It is inherited directly or indirectly by all solver classes
###############################################################################
class CoreExcitationHelper:
    def __init__(self, Engine,
                 xi = None,
                 Report = 0,
                 **kwargs):
        # Level of reporting
        self.Report = Report

        self.SetEngine(Engine)

        # Can ignore xi - is for a future feature
        self.Setxi(xi)

        # This is the general properties
        self.Props = {}

        # These routines can be overriden in inherited classes 
        self.SetProps(**kwargs)
        self.SetMix(**kwargs)

        # These routines should not be tampered with
        self.SetFrom(**kwargs)
        self.SetTo(**kwargs)

        # These routines can be overriden in inherited classes 
        self.SetExtras(**kwargs)

    # This deals with class-specific extras
    def SetExtras(self, **kwargs):
        1

    # Initialise the engine and derived quantities
    def SetEngine(self, Engine):
        self.Engine = Engine

        self.NAtom = Engine.NAtom

        self.f = Engine.f_Occ
        self.kh = Engine.kh
        self.kl = Engine.kl

        self.C0 = Engine.C*1.
        self.epsilon0 = Engine.epsilon*1.

        # Use the initial orbitals
        self.epsilonE = Engine.epsilon*1.
        self.CE = Engine.C*1.

        # By default it hasn't converged
        self.Converged = False

    # Set default xi
    def Setxi(self, xi):
        # This is the scaling factor for sr density-driven correlations
        if xi is None:
            self.xi = -0.32
        else:
            self.xi = xi

        
    # Set the internal properties
    def SetProps(self, MaxIter = 200, ShowIter = 20,
                 DECut = 1e-7, DepsCut = 1e-6,
                 Fail = True,
                 **kwargs):
        self.Props['MaxIter'] = MaxIter
        self.Props['ShowIter'] = ShowIter
        self.Props['DECut']   = DECut
        self.Props['DepsCut'] = DepsCut
        self.Props['Fail']    = Fail

    # Update the internal properties
    def UpdateProps(self, **kwargs):
        for v in self.Props:
            if v in kwargs: self.Props[v]  = kwargs[v]

    # Set the mixing properties
    def SetMix(self, MixC = 0.5, Mix = None, Mix2 = None,
               **kwargs):
        self.Props['MixC'] = MixC
        self.Props['Mix']  = Mix
        if Mix2 is None:
            self.Props['Mix2'] = Mix
        else:
            self.Props['Mix2'] = Mix2

    # Convert kFrom or kTo
    def Process_k(self, k='HOMO'):
        if k is None: # Handle None
            return None 
        elif isinstance(k, str): # Handle strings
            Txt = k.upper().replace('HOMO', 'H').replace('LUMO', 'L')
            ID = Txt[0]
            if len(Txt)>1:
                Delta = int(Txt[1:])
            else: Delta = 0

            return {'H':self.kh, 'L':self.kl}[ID]+Delta
        elif hasattr(k, '__iter__'): # Handle iterables
            kout = []
            for kk in k:
                kout += [self.Process_k(kk),]
            return kout
        else: # If it's an integer leave it alone
            return k

    # Set the 'from' orbital by number (<=kh) or symmetry
    def SetFrom(self, k=None, Sym=None, **kwargs):
        k = self.Process_k(k)
        
        if not(Sym is None):
            # By symmetry
            self.kFrom = self.Engine.kh
            for q in range(self.Engine.kh):
                k = self.Engine.kh - q
                if self.Engine.Sym_k[k] == Sym:
                    self.kFrom = k
                    break
        elif (k is None) or (k>self.Engine.kh):
            self.kFrom = self.Engine.kh
        else:
            self.kFrom = k

        if self.Report>=3:
            print("k_From = %3d with sym %2d"\
                  %(self.kFrom, self.Engine.Sym_k[self.kFrom]))

        return self.kFrom

    # Set the 'to' orbital by number (>kh) or symmetry
    def SetTo(self, k=None, Sym=None, **kwargs):
        k = self.Process_k(k)
        
        if not(Sym is None):
            # By symmetry
            self.kTo = self.Engine.kl
            for k in range(self.Engine.kl, self.Engine.nbf):
                if self.Engine.Sym_k[k] == Sym:
                    self.kTo = k
                    break
        elif (k is None) or (k<self.Engine.kl):
            self.kTo = self.Engine.kl
        else:
            self.kTo = k

        if self.Report>=3:
            print("k_To   = %3d with sym %2d"\
                  %(self.kTo  , self.Engine.Sym_k[self.kTo  ]))

        return self.kTo

    # Reset the current orbitals to their initial values
    def ResetOrbitals(self):
        self.CE = self.C0*1.
        self.epsilonE = self.epsilon0*1.

    # Reset the orbitals to the input natural orbitals
    def ResetNaturialOrbitals(self):
        SI = la.inv(self.Engine.S_ao)
        D = self.Engine.Da + self.Engine.Db

        # Get the NO of D
        f, SC = la.eigh(-self.Engine.D , b=SI)
        CE = SI.dot(SC)

        # Project onto the Fock operator
        E, F = self.GetEnergy(Plan, self.C0)
        eps = np.einsum('pq,pk,qk->k', F, CE, CE)
        k = np.argsort(eps)

        self.CE = CE[:,k]

    # Quick call to solve for the GS
    #
    # Note, PlanOnly returns the GS plan
    def SolveGS(self, xi=None,
                UseNO = False,
                kFrom = None, kTo = None,
                PlanOnly=False,
                **kwargs):
        # xi is not used
        if kFrom is None: kFrom = self.kFrom
        if kTo   is None: kTo   = self.kTo

        
        kFrom = self.Process_k(kFrom)
        kTo   = self.Process_k(kTo  )

        Plan = {
            'kTo': (kTo,),
            'Singlet': True,
            '1RDM': self.f, # Weights only for 1-RDM
            'Hx': [ (1., 1., self.f), ], # (EWeight, FWeight, Occ)
            'xcDFA': [ (1., 1., self.f), ], # (EWeight, FWeight, Occ)
            'Extra': None, # [(PreE, PreF, k1, k2, Kind), ...]
        }

        if PlanOnly: return Plan
        
        if UseNO:
            self.ResetNaturalOrbitals()
            return self.Solver(Plan, Reset = False, **kwargs)
        else:
            return self.Solver(Plan, **kwargs)

    def SolvePol(self, Pol = 0, Nup = None, Ndn = None,
                 UseNO = False,
                 xi=None,
                 kFrom = None, kTo = None,
                 PlanOnly=False,
                 **kwargs):
        # xi is not used
        # kFrom and kTo are not used
        if not(Nup is None) and not(Ndn is None):
            Na, Nb = max(Nup, Ndn), min(Nup, Ndn)
        else:
            if Pol%2==0:
                Na = 1+self.kh + Pol//2
                Nb = 1+self.kh - Pol//2
            else:
                Na = 1+self.kh + (Pol-1)//2
                Nb = 1+self.kh - (Pol+1)//2

        f_pol = np.ones(Na)
        f_pol[:Nb] += 1.
        kTo = Na-1
            
        Plan = {
            'kTo': (kTo,),
            'Singlet': False,
            '1RDM': f_pol, # f only
            'Hx': [ (1., 1., f_pol), ], # (EWeight, FWeight, Occ)
            'xcDFA': [ (1., 1., f_pol), ], # (EWeight, FWeight, Occ)
            'Extra': None, # [(PreE, PreF, k1, k2, Kind), ...]
        }

        if PlanOnly: return Plan

        # Use the natural orbitals of D (broken sym) as better starting point
        if UseNO:
            self.ResetNaturalOrbitals()
            return self.Solver(Plan, Reset = False, **kwargs)
        else:
            return self.Solver(Plan, **kwargs)
            
    def SolveTS(self, xi=None,
                UseNO = False,
                kFrom = None, kTo = None,
                PlanOnly=False,
                **kwargs):
        # xi is not used
        if kFrom is None: kFrom = self.kFrom
        if kTo   is None: kTo   = self.kTo

        kFrom = self.Process_k(kFrom)
        kTo   = self.Process_k(kTo  )

        f_ts = PromoteOcc(self.f, kFrom, kTo)
            
        Plan = {
            'kTo': (kTo,),
            'Singlet': False,
            '1RDM': f_ts, # f only
            'Hx': [ (1., 1., f_ts), ], # (EWeight, FWeight, Occ)
            'xcDFA': [ (1., 1., f_ts), ], # (EWeight, FWeight, Occ)
            'Extra': None, # [(PreE, PreF, k1, k2, Kind), ...]
        }

        if PlanOnly: return Plan

        if UseNO:
            self.ResetNaturalOrbitals()
            return self.Solver(Plan, Reset = False, **kwargs)
        else:
            return self.Solver(Plan, **kwargs)

    # For use in singlet calculations
    def ExtraEST(self, kFrom, kTo, xi, fl=1., s=1.):
        # Need to handle xi properly
        # Use RS if xi is positive and has RS
        if self.Engine.Has_RS and self.xi>=0.:
            cK = 2.-2.*np.abs(xi)
            cK_w = 2.*np.abs(xi)
        else: # Oterwise, use it only on  full K
            cK = 2.-2.*np.abs(xi)
            cK_w = 0.

        if fl is None:
            # No contribution to Fock operator
            Extra = [ (s*cK, 0., kFrom, kTo, 'K'), ]
            if (np.abs(cK_w)>0.):
                Extra += [ (s*cK_w, 0., kFrom, kTo, 'K_w')]
        else:
            Extra = [ (s*cK, s*cK/fl, kFrom, kTo, 'K'), ]
            if (np.abs(cK_w)>0.):
                Extra += [ (s*cK_w, s*cK_w/fl, kFrom, kTo, 'K_w')]
                
        return Extra

    def SolveSX(self, **kwargs):
        return self.SolveSS(**kwargs)
    def SolveSS(self, xi=None,
                UseNO = False,
                kFrom = None, kTo = None,
                PlanOnly=False,
                **kwargs):
        if xi is None: xi = self.xi
        if kFrom is None: kFrom = self.kFrom
        if kTo   is None: kTo   = self.kTo

        kFrom = self.Process_k(kFrom)
        kTo   = self.Process_k(kTo  )

        Plan = self.SolveTS(xi=xi, kFrom=kFrom, kTo=kTo,
                            PlanOnly=True, **kwargs)

        Plan['Singlet'] = True
        Plan['Extra'] = self.ExtraEST(kFrom, kTo, xi, fl=1.)

        if PlanOnly: return Plan

        if UseNO:
            self.ResetNaturalOrbitals()
            return self.Solver(Plan, Reset = False, **kwargs)
        else:
            return self.Solver(Plan, **kwargs)   
        
    # Solves for a double excitation from kFrom to kTo without using combination laws
    def SolveDX_NC(self, xi=None,
                   UseNO = False,
                   kFrom = None, kTo = None,
                   PlanOnly=False,
                   flScale=1.,
                   **kwargs):
        if xi is None: xi = self.xi
        if kFrom is None: kFrom = self.kFrom
        if kTo   is None: kTo   = self.kTo

        kFrom = self.Process_k(kFrom)
        kTo   = self.Process_k(kTo  )

        f_gs = 1.*self.f
        f_ts = PromoteOcc(self.f, kFrom, kTo)
        f_dx = PromoteOcc(self.f, kFrom, kTo, 2.)

        Plan = {
            'kTo': (kTo,),
            'Singlet': True,
            '1RDM': f_dx, # f only
            'Hx'   : [ ( 2., 1., f_ts),
                       (-1., 0., f_gs),], # (EWeight, FWeight, Occ)
            'xcDFA': [ ( 1., 1., f_dx), ],
        }

        # Extra Hartree part
        Plan['Extra'] = [
            ( 1.,-1., kFrom, kFrom, 'J'),
            (-2., 0., kFrom, kTo  , 'J'),
            ( 1., 1., kTo  , kTo  , 'J'),
        ]

        # ST term with DD correlation correction
        Plan['Extra'] += self.ExtraEST(kFrom, kTo, xi, fl=2.*flScale)
        
        if PlanOnly: return Plan

        if UseNO:
            self.ResetNaturalOrbitals()
            return self.Solver(Plan, Reset = False, **kwargs)
        else:
            return self.Solver(Plan, **kwargs)

    # Solves for a double excitation from kFrom to kTo
    def SolveDX(self, xi=None,
                UseNO = False,
                kFrom = None, kTo = None,
                PlanOnly=False,
                flScale=1.,
                AllowSplit = True, # Set false to force pure double
                **kwargs):
        if xi is None: xi = self.xi
        if kFrom is None: kFrom = self.kFrom
        if kTo   is None: kTo   = self.kTo

        kFrom = self.Process_k(kFrom)
        kTo   = self.Process_k(kTo  )

        if AllowSplit and \
           np.abs(self.epsilon0[kTo] - self.epsilon0[kTo+1])<zero_round:
            print("Degenerate LUMO -- doing split double excitation!")
            return self.SolveDXDbl(xi=xi,
                                   kFrom = kFrom, kTo = (kTo, kTo+1),
                                   PlanOnly = PlanOnly,
                                   flScale1=flScale, flScale2=flScale,
                                   **kwargs)

        f_gs = 1.*self.f
        f_ts = PromoteOcc(self.f, kFrom, kTo)
        f_dx = PromoteOcc(self.f, kFrom, kTo, 2.)

        Plan = {
            'kTo': (kTo,),
            'Singlet': True,
            '1RDM': f_dx, # f only
            'Hx': [ ( 2., 1., f_ts),
                    (-1., 0., f_gs),], # (EWeight, FWeight, Occ)
            'xcDFA': [ ( 2., 1., f_ts),
                       (-1., 0., f_gs),], # (EWeight, FWeight, Occ)
        }

        # Extra Hartree part
        Plan['Extra'] = [
            ( 1.,-1., kFrom, kFrom, 'J'),
            (-2., 0., kFrom, kTo  , 'J'),
            ( 1., 1., kTo  , kTo  , 'J'),
        ]

        # ST term with DD correlation correction
        Plan['Extra'] += self.ExtraEST(kFrom, kTo, xi, fl=2.*flScale)
        
        if PlanOnly: return Plan

        if UseNO:
            self.ResetNaturalOrbitals()
            return self.Solver(Plan, Reset = False, **kwargs)
        else:
            return self.Solver(Plan, **kwargs)

    # Solves for a double excitation from kFrom to (kTo1, kTo2)
    def SolveDXDbl(self, xi=None,
                   UseNO = False,
                   kFrom = None, kTo = None,
                   PlanOnly=False,
                   flScale1=1., flScale2=1.,
                   **kwargs):
        if xi is None: xi = self.xi
        if kFrom is None: kFrom = self.kFrom
        if kTo   is None: kTo   = (self.kTo, self.kTo+1)

        kFrom = self.Process_k(kFrom)
        kTo   = self.Process_k(kTo  )

        kTo1, kTo2 = tuple(kTo)

        f_gs = 1.*self.f
        f_ts1 = PromoteOcc(self.f, kFrom, kTo1)
        f_ts2 = PromoteOcc(self.f, kFrom, kTo2)
        f_dx  = PromoteOcc(f_ts1 , kFrom, kTo2)

        Plan = {
            'kTo': (kTo1, kTo2),
            'Singlet': True,
            '1RDM': f_dx, # f only
            'Hx': [ ( 1., 0.5, f_ts1),
                    ( 1., 0.5, f_ts2),
                    (-1., 0. , f_gs),], # (EWeight, FWeight, Occ)
            'xcDFA': [ ( 1., 0.5, f_ts1),
                       ( 1., 0.5, f_ts2),
                       (-1., 0. , f_gs),], # (EWeight, FWeight, Occ)
        }

        # Extra Hartree part
        Plan['Extra'] = [
            ( 1.,-1. , kFrom, kFrom, 'J'), # -1
            (-1., 0.5, kTo1 , kFrom, 'J'), # -0.5
            (-1., 0.5, kTo2 , kFrom, 'J'), # -0.5
            ( 1., 0. , kTo1 , kTo2 , 'J'),
        ]

        # ST term with DD correlation correction
        Plan['Extra'] += self.ExtraEST(kFrom, kTo1, xi, fl=2.*flScale1, s=0.5)
        Plan['Extra'] += self.ExtraEST(kFrom, kTo2, xi, fl=2.*flScale1, s=0.5)
        Plan['Extra'] += self.ExtraEST(kTo1 , kTo2, xi, fl=None, s=0.5)
        Plan['Extra'] += self.ExtraEST(kTo2 , kTo1, xi, fl=None, s=0.5)

        Plan = TidyPlan(Plan)
        
        if PlanOnly: return Plan

        return self.Solver(Plan, kTo=(kTo1,kTo2), **kwargs)

    # The routine evaluates the energy and default Fock matrix for a given
    # plan
    #   C defaults to self.CE
    #   StoreParts=True gets it to store compnents of the energy
    def GetEnergy(self, Plan, C=None,
                  StoreParts = False):
        if C is None: C = self.CE

        f = Plan['1RDM']
        nf = len(f)
        CT = C[:,:nf]
        D = np.einsum('pk,qk,k->pq', CT, CT, f)

        FTV = self.Engine.T_ao + self.Engine.V_ao
        ETV = np.vdot(FTV, D)

        self.LastEns = {'ETV':ETV,}

        # Initialise the energy and (up) Fock matrix using
        # the trivial kinetic (T) and external potential (V)        
        E = ETV
        F = FTV
        if (self.Report>=20):
            print("W = %.3f, ETV = %10.5f"%(1., ETV))

        # Add the Hartree-exchange terms from the weighted
        # sum of existing DFAs (with appropriate scaling of x)
        self.LastEns['Hx'] = 0.
        self.LastEns['Hx Parts'] = []
        for WE, WF, f in Plan['Hx']:
            fa = np.minimum(f, 1.)
            fb = f - fa

            nf = len(f)
            CT = C[:,:nf]

            Ca = CT[:,np.abs(fa-1.)<zero_round]
            if np.mean(np.abs(fa-fb))>zero_round:
                Cb = CT[:,np.abs(fb-1.)<zero_round]
                EHx, FHx, FHx_d = self.Engine.GetHx(Ca=Ca, Cb=Cb, BothSpin=True)
            else:
                EHx, FHx = self.Engine.GetHx(Ca=Ca)
                FHx_d = FHx

            if (self.Report>=20):
                print("WE = %.3f, WF = %.3f, EHx = %10.5f"%(WE, WF, EHx))

            E += WE*EHx
            F += WF*FHx

            self.LastEns['Hx'] += WE*EHx
            if StoreParts:
                self.LastEns['Hx Parts'] += [(EHx, FHx*1., FHx_d*1.)]
            
        # Add the DFA terms from the weighted sum of
        # existing DFAs (with appropriate scaling of x)
        self.LastEns['xcDFA'] = 0.
        self.LastEns['xcDFA Parts'] = []
        for WE, WF, f in Plan['xcDFA']:
            fa = np.minimum(f, 1.)
            fb = f - fa

            nf = len(fa)
            CT = C[:,:nf]
            Da = np.einsum('pk,qk,k', CT, CT, fa, optimize=True)
            Db = np.einsum('pk,qk,k', CT, CT, fb, optimize=True)

            ExcDFA, FxcDFA, FxcDFA_d = self.Engine.GetDFA(Da=Da, Db=Db, BothSpin=True)

            if Plan['Singlet'] and False:
                kk = np.argwhere(np.round(f*(2-f))==1.).reshape((-1,))
                fC = np.floor(f/2.+0.0001) # Get the core
                if len(kk)==0:
                    ab = []
                if len(kk)==2:
                    ab = [ ([kk[0],], [kk[1],]), ([kk[1],], [kk[0],]) ]

                ExcNew = []
                for a, b in ab:
                    fa = fC*1.
                    fb = fC*1.
                    fa[a] = 1.
                    fb[b] = 1.
                    Da = np.einsum('pk,qk,k', CT, CT, fa, optimize=True)
                    Db = np.einsum('pk,qk,k', CT, CT, fb, optimize=True)
                    Exc_ab, _, _ = self.Engine.GetDFA(Da=Da, Db=Db, BothSpin=True)
                    ExcNew += [Exc_ab]

                if len(ab)>0:
                    ExcDFA  = np.mean(ExcNew)

            if (self.Report>=20):
                print("WE = %.3f, WF = %.3f, ExcDFA = %10.5f"%(WE, WF, ExcDFA))
                
            E += WE*ExcDFA
            F += WF*FxcDFA

            self.LastEns['xcDFA'] += WE*ExcDFA
            if StoreParts:
                self.LastEns['xcDFA Parts'] += [(ExcDFA, FxcDFA*1., FxcDFA_d*1.)]

        # Add the extra terms (of J and K form) that come from
        # EST' as well as anything missed in the Hartree term
        #
        # Note, the energy and Fock can be treated inconsistently
        # here to accommodate approximations
        self.LastEns['Extra'] = 0.
        self.LastEns['Extra Parts'] = []
        if ('Extra' in Plan) and not(Plan['Extra'] is None):
            if StoreParts:
                EEx, FEx = 0., 0.
                for P in Plan['Extra']:
                    Pm = (1., 1., P[2], P[3], P[4]) # [(PreE, PreF, k1, k2, Kind), ...]
                    EEx_P, FEx_P = self.Engine.GetExtra(C, [Pm,])
                    EEx += P[0] * EEx_P
                    FEx += P[1] * FEx_P
                    self.LastEns['Extra Parts'] += [(EEx_P, FEx_P*1.)]
            else:
                EEx, FEx = self.Engine.GetExtra(C, Plan['Extra'])

            E += EEx
            F += FEx

            self.LastEns['Extra'] += EEx

        # Finally, add the nuclear-nuclear term to the energy
        E += self.Engine.Enn

        # Return E and F(up)
        return E, F

    # The routine evaluates the energy and orbieal-resolved Fock matrix
    # for a given plan
    #   C defaults to self.CE
    #   Raw=True returns raw Fock matrices (do not use!)
    # 
    # Note, computes different Fock matrices for different orbitals.
    # The way this is handled is a mess and will be fully revamped
    # in future releases.
    def GetEnergyFocks(self, Plan_, C=None,
                       Raw=False):
        if C is None: C = self.CE

        # Return all Nones if no Plan
        if Plan_ is None: return None, None, None

        # This makes sure the plan is suitable for energies by
        # combining equal ERIs like [pp|qq] and [qq|pp]
        Plan = EnergyPlan(Plan_)
        E, _ = self.GetEnergy(Plan, C=C, StoreParts=True)

        # Initialise the map
        # - all core are treated the same way
        # - all unoccupied are treated the same way

        NOrb = C.shape[1]
        N_f = len(Plan['1RDM'])
        f = Plan['1RDM']

        Core = []
        Holes = []
        Other = []
        for k in range(NOrb):
            if k>=N_f:
                Holes += [k,]
            elif f[k]==2.:
                # Doubly occupied are safe unless in Extra
                Extra = False
                for _, _, k1, k2, _ in Plan['Extra']:
                    Extra = Extra or (k==k1) or (k==k2)

                # No extra - set to core, otherwise in other
                if not(Extra): Core += [k,]
                else: Other += [k,]
            elif f[k]==0.:
                Holes += [k,]
            else: Other += [k,]

        All = set(Core + Other + Holes)
        if not(len(All) == NOrb):
            print("Something has gone wrong - wrong number orbitals")
            print(Core)
            print(Other)
            print(Holes)
            print(All)
            quit()

        # Handle the special case of no core electrons
        if len(Core)>0:
            kLow = np.min(Core)
        else: kLow = 0
        
        if len(Other)>0:
            kHi = np.max(Other)+1
        elif len(Core)>0:
            kHi = np.max(Core)+1
        else:
            kHi = 1

        if len(Core)>0:
            FockMap = { kLow:[], kHi:[] }
            kUnique = [kLow, kHi,]
        else:
            FockMap = { kLow:[kLow,], kHi:[] }
            kUnique = [kLow, kHi,]
            
        for k in range(NOrb):
            if k in Core:
                # Doubly occupied are assigned the lowest possible
                FockMap[kLow] += [k,]
            elif k in Holes:
                # Doubly occupied are assigned the lowest possible
                FockMap[kHi] += [k,]
            else:
                FockMap[k] = [k,]
                kUnique += [k,]

        # Contains effective Fock operators
        # F_k = 1/fk d/dphi_k^* E[{phi}]
        # Note the normalization using fk
        
        FockList = {}
        for k in kUnique:
            # Skip when f[k] is zero (undefined == 0)
            if k>=N_f: continue
            if f[k]==0.: continue
            
            FockList[k] = self.Engine.T_ao + self.Engine.V_ao # Start with TV

            # Get the plan for a derivative with respect to orbital k
            FPlan = PlanDeriv(Plan, k, IsEnergy=True)

            # Process the 1RDM sums
            for Key in ('Hx', 'xcDFA'):
                # Note, we use the energy prefactor here
                for K, (WE, _, fF) in enumerate(FPlan[Key]):
                    if k<len(fF):
                        # Up occupation in density
                        tu = min(fF[k], 1.) # At most one up electron
                        # Down occupation in density
                        td = (fF[k] - tu) # Remainder are down

                        # Save time if unoccupied
                        if np.abs(tu) + np.abs(td)<2*zero_round:
                            continue

                        if self.Report>=10:
                            print("%d %3d | WF = %6.3f, tu = %d, td = %d, wu = %6.3f, wd = %6.3f"\
                                  %( K, k, WE, tu, td, WE*tu/f[k], WE*td/f[k]))

                        # Fock contrinution from up [1] and down [2]
                        FockList[k] += WE * (tu * self.LastEns[Key+' Parts'][K][1]
                                             + td * self.LastEns[Key+' Parts'][K][2]) / f[k]

            # Handle the extra continbutions
            for K, (_, WF, _, _, _) in enumerate(FPlan['Extra']):
                FockList[k] += WF * self.LastEns['Extra Parts'][K][1]

        if Raw:
            # Return the unprcoessed quantities (not recommended!)
            # Return:
            # the energy E
            # unique Fock operators in FockList
            # map from k to FockList
            return E, FockList, FockMap

        # Do some additional processing to:
        # * Deal with the unoccupied orbitals
        # * Pad any holes
        # * Convert things to dictionaries for easy use
        
        # Project Fock operators onto orbitals
        MapBack = np.zeros((NOrb,), dtype=int)
        for k in FockList:
            MapBack[FockMap[k]] = k
            FockList[k] = (C.T).dot(FockList[k]).dot(C)

        # Use kTo by default for unoccupied orbitals
        if not('kTo' in Plan) or len(Plan['kTo'])==0:
            # If kTo is unspecified use the Fock operator for the highest orbital
            # for unoccupied
            FUnocc = FockList[MapBack[kHi-1]]*1.
        else:
            # If kTo is specified use their average Fock operator for unoccupied
            w = 1/len(Plan['kTo'])
            FUnocc = 0.
            for k in Plan['kTo']:
                FUnocc += w * FockList[k]

        # Fock operator for unoccupied
        FockList[kHi] = FUnocc

        # Tidy up the maps to reduce accidental dupes and sort
        for k in FockMap:
            FockMap[k] = np.array(sorted(list(set(FockMap[k]))))

        # Return:
        # the energy E
        # unique Fock operators in a FockList dictionary
        # map from k to FockList in a FockMap dictionary
        return E, FockList, FockMap



    # Generic solver routine - overwritten in inherited classes
    def Solver(self, Plan, kFrom=None, kTo=None,
               Dipole = False, # Return the dipole as well
               **kwargs):
        return self.SolverFrozen(Plan, kFrom=kFrom, kTo=kTo,
                                 Dipole=Dipole,
                                 **kwargs)

    # Frozen solver routine using current orbitals
    # This evaluates EDFT@DFT if called before orbital updates
    # or EDFT@Last run if called later
    #
    # Useful for obtaining the dipole if not asked the first
    # time.
    def SolverFrozen(self, Plan,
                     Dipole = False, # Return the dipole as well
                     **kwargs):
        epsE = self.epsilonE
        CE = self.CE

        E, F = self.GetEnergy(Plan, CE)
        
        if Dipole:
            mu = np.tensordot(self.Engine.Dip_ao, self.LastD,
                              axes=((1,2),(0,1)))
            return E, mu
        else:
            return E
        
    # Da, Db = GetRDM()
    # Setting pyscf to True gets it in pyscf style as a (2,NBas,NBas) tensor
    def GetRDM(self, f = None, C = None,
               pyscf=False):
        if f is None: f = self.Lastf
        if C is None: C = self.CE

        fa = np.minimum(f, 1.)
        fb = f - fa
        
        nf = len(fa)
        CT = C[:,:nf]
        Da = np.einsum('pk,qk,k', CT, CT, fa, optimize=True)
        Db = np.einsum('pk,qk,k', CT, CT, fb, optimize=True)

        if pyscf:
            D = np.zeros((2, Da.shape[0], Da.shape[0]))
            D[0,:,:] = Da
            D[1,:,:] = Db
            return D

        return Da, Db


    
if __name__ == "__main__":
    1
