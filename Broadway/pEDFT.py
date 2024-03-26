from Broadway.EDFT import CoreExcitationHelper
from Broadway.LDAFits import *

import numpy as np
import scipy.linalg as la

eV = 27.211

zero_round = 1e-5

class pExcitationHelper(CoreExcitationHelper):
    # Generic solver routine - currently can only call SolverpEDFT
    def Solver(self, Plan, kFrom=None, kTo=None,
               Dipole = False, # Return the dipole as well
               **kwargs):
        return self.SolverpEDFT(Plan, **kwargs)
        
    # Solve the pEDFT equations iteratively
    def SolverpEDFT(self, Plan,
                    Dipole = False, # Return the dipole as well
                    **kwargs):
        kTo   = Plan['kTo']

        # Allow for multiple orbital changes
        kToArr = np.atleast_1d(kTo)

        
        self.UpdateProps(**kwargs)
        self.SetMix(**kwargs)

        # Initialise to the current orbitals to take advantage
        # of earlier runs
        epsE = self.epsilonE
        CE = self.CE

        # This is an orbital-level mixer to improve convergence
        def MixOrbital(CE, CE_Old):
            for k_ in kToArr:
                k = k_ - self.kl

                # Flip CE if direction wrong
                x = (CE[:,k]).dot(self.Engine.S_ao).dot(CE_Old[:,k])
                if x<0.: CE[:,k] *= -1.

                Cp = CE[:,k]*(1. - self.Props['MixC']) + CE_Old[:,k]*self.Props['MixC']
                Cp /= np.sqrt( (Cp).dot(self.Engine.S_ao).dot(Cp) )

                CE[:,k] = Cp
            return CE
            
        
        E_Old = 1e10
        epsTo_Old = 1e10
        CE__Old = CE[:,self.kl:]
        F_Old  = 0.
        F_Old2 = 0.

        self.Converged = False

        if (self.Report>=3):
            print("Iter %10s %10s"%("D E", "D eps"))
        
        for IterStep in range(self.Props['MaxIter']):
            E, F = self.GetEnergy(Plan, CE)

            if (IterStep>=2) and not(self.Props['Mix'] is None):
                F = (1. - self.Props['Mix'] - self.Props['Mix2'])*F \
                    + self.Props['Mix']*F_Old + self.Props['Mix2']*F_Old2

            # Recalculate the unoccupied orbitals
            epsE_, CE_ = self.Engine.SolveFock(F, k0 = self.kl)

            epsE[self.kl:] = epsE_
            CE[:,self.kl:] = MixOrbital(CE_, CE__Old)
            CE__Old = CE_*1.

            epsTo = np.sum(epsE[kToArr])

            if (self.Report>=3) and (IterStep>0):
                print("%4d %10.5f %10.5f"%(IterStep, E-E_Old, epsTo-epsTo_Old))

            if (np.abs(E-E_Old)<self.Props['DECut'])\
               and (np.abs(epsTo-epsTo_Old)<self.Props['DepsCut']):
                self.Converged = True
                break

            E_Old = E
            epsTo_Old = epsTo

            F_Old2 = F_Old*1.
            F_Old  = F*1.

        if self.Report>0:
            print("Took %4d iterations to converge"%(IterStep))

        # Do not mix at the last IterStep
        epsE[self.kl:] = epsE_
        CE[:,self.kl:] = CE_

        self.LastIter = IterStep

        f = np.array(Plan['1RDM'])
        self.Lastf = 1.*f
        self.LastPlan = Plan
        self.LastF = 1.*F
        self.LastD = np.einsum('pk,qk,k', CE[:,:len(f)], CE[:,:len(f)], f)
        
        if self.Converged:
            self.CE = CE
            self.epsilonE = epsE
        elif self.Props['Fail']:
            print("Error! Failed to converge - returning None")
            return None
        else:
            print("Warning! Failed to converge - returning p^2EDFT")
            E, F = self.GetEnergy(Plan, self.CE)

        if Dipole:
            mu = np.tensordot(self.Engine.Dip_ao, self.LastD,
                              axes=((1,2),(0,1)))
            return E, mu
        else:
            return E

    # Calculate the transition energies
    def CalcTE(self, C=None, f=None,
               kFrom=None, kTo=None,
               Dipole = False, # Return the dipole as well
               **kwargs):
        if C is None: C = self.CE
        if kFrom is None: kFrom = self.kFrom
        if kTo   is None: kTo   = self.kTo

        kFrom = self.Process_k(kFrom)
        kTo   = self.Process_k(kTo  )

        if not(f is None):
            f = np.round(f)
            CP = C[:,:len(f)][:,f>0.]
            
            F = self.Engine.T_ao + self.Engine.V_ao \
                + 2*self.Engine.GetFJ(CP) - self.Engine.GetFK(CP)
        else:
            F = self.LastF

        C1 = C[:,self.kFrom]
        C2 = C[:,self.kTo]
        
        DEsx = np.sqrt(2)*(C1).dot(F).dot(C2) 
        DEdx = (C1).dot(self.Engine.GetFK(C2)).dot(C1)

        if Dipole:
            mu = np.sqrt(2)*np.einsum('vpq,p,q->v', self.Engine.Dip_ao,
                                      C1, C2, optimize=True)
            return DEsx, DEdx, mu

        return DEsx, DEdx

    # Rediagnolise to update the energies
    def Rediagonalise(self, E00, E01, E02, E11, E12, E22):
        H = np.array([[0,E01,E02],[E01,E11-E00,E12],[E02,E12,E22-E00]])
        w, v = la.eigh(H)
        return E00+w[0], E00+w[1], E00+w[2]


class ExcitationHelper(pExcitationHelper):
    1


if __name__ == "__main__":
    1
