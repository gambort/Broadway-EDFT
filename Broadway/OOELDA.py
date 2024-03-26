from Broadway.EDFT import *
from Broadway.OOEDFT import OOExcitationHelper
from Broadway.LDAFits import *

import numpy as np
import scipy.linalg as la
import numpy.random as ra

import psi4

eV = 27.211

zero_round = 1e-5
rho_Min = 1e-18

def GetPropsLDA(Plan, Report=False):
    # Get the length
    Nf = 0
    for _, _, f in Plan['xcDFA']:
        Nf = max(Nf, len(f))
    
    fLDA = np.zeros((Nf,))
    for WE, WF, f in Plan['xcDFA']:
        if len(f)<Nf:
            fLDA[:len(f)] += WE * f
        else:
            fLDA += WE * f

    
    NCore, Done = 0, False
    NOcc = 0
    for k in range(Nf):
        if fLDA[k]<2.: Done = True

        if not(Done): NCore = k+1
        elif fLDA[k]>0.: NOcc = k+1


    # Calculate the on-top pair-occupation factors
    PProps = {}

    # Exchange
    Px  = 0.*np.outer(fLDA, fLDA)
    for k in range(Nf):
        for kp in range(Nf):
            Px[k,kp] = -fLDA[max(k,kp)]

    # Hartree-exchange
    PHx = 0.
    for WE, _, f in Plan['xcDFA']:
        # Pad if needed
        if (len(f)<len(fLDA)):
            f_ = f*1.
            f = 0.*fLDA
            f[:len(f_)] = f_

        fu = np.minimum(f, 1.)
        fd = f - fu

        PHx += WE*(np.outer(f, f) - np.outer(fu, fu) - np.outer(fd, fd))

    if not(Plan['Extra'] is None) and not(len(Plan['Extra'])==0):
        for WE, _, k1, k2, Kind in Plan['Extra']:
            if k1 == k2:
                PHx[k1, k2] += WE*2.
            else:
                PHx[k1, k2] += WE
                PHx[k2, k1] += WE
            
    if Report:
        print(NCore)
        print(NOcc)

        ii = np.arange(max(0, NCore-1), NOcc)
        print('fLDA = ')
        print(fLDA[ii])

        PH = PHx - Px
        print('Pair densitiess (Hx, H, x)')
        print(PHx[ii,:][:,ii])
        print(PH[ii,:][:,ii])
        print(Px[ii,:][:,ii])
        
    PProps = { 'PHx': PHx, 'Px': Px, 'PH': PHx - Px }
    
        
    return fLDA, NCore, NOcc, PProps


class OOELDAExcitationHelper(OOExcitationHelper):
    def SetExtras(self, **kwargs):
        self.eLDAType = 'RE' # Regular eLDA
        self.eLDAProps = {'a':1/3}
        self.IgnorefDeriv = False

        
        # Initialise the grid when first called
        try:
            self.Grid = self.Engine.GetDFTGrid()
        except:
            print("Engine.GetDFTGrid() is not defined")
            # Note, GetDFTGrid returns a list over points of dictionaries with
            # [ {'w': weights_point, 'lpos': lpos_point, 'lphi': lphi_point}, etc ]
            quit()
            
    def SeteLDAType(self, Type, **kwargs):
        self.eLDAType = Type[:2].upper()
        if 'IgnorefDeriv' in kwargs:
            self.IgnorefDeriv = kwargs['IgnorefDeriv']
        if 'a' in kwargs:
            self.eLDAProps['a'] = kwargs['a']
        else: self.eLDAProps['a'] = 1/3

    def ComputeFxc_eLDA(self, f, C=None, kList=None,
                        NCore = 0, NOcc = 0, PProps = {},
                        **kwargs):
        if C is None: C = self.C

        if NOcc == 0: NOcc = len(f)
            
        # All k we are interested in
        if kList is None: kList = list(self.FockMap)
        kList = np.atleast_1d(kList)


        # Initialise key variables/definitions
        
        # Doubly occupied part of the density
        DCore = np.einsum('pk,qk->pq', C[:,:NCore], C[:,:NCore])

        if self.Report>10: # Debug information
            print("%3d %3d %6.2f %6.2f"%(NCore, NOcc, np.sum(f), np.vdot(DCore, self.Engine.S_ao)))

        # Active part of the density
        kAct = np.arange(NCore, NOcc)

        # Pad f (if needed)
        if len(kAct)>0:
            # Pad f if needed
            if len(f) < np.max(kAct)+1:
                f_ = f*1.
                f = np.zeros(np.max(kAct)+1)
                f[:len(f_)] = f_
                
            ff_k = f[kAct]
            Px_kk = PProps['Px'][(kAct[:,None], kAct)]
            PH_kk = PProps['PH'][(kAct[:,None], kAct)]

        #################################################################################
        # Function for the f-dependent pre-factor on derivative
        def Deriv_Fn_of_f(f): return f, 0
        
        if self.eLDAType in ('U', 'UN'):
            #print("Running in unpolarized mode") # TEMP
            # Unpolarized calculation
            def GetLocalProps(rho_Core, rho_k=None):
                # No rho_k means unpolarized
                if rho_k is None: return 2.*rho_Core, 2.+0.*rho_Core, None, None

                # Compute rho
                rho = 2.*rho_Core + np.dot(rho_k, ff_k)

                return rho, 2.+0.*rho, None, None

            
        elif self.eLDAType in ('OT', 'ON'):
            #print("Running in on-top mode") # TEMP
            def GetLocalProps(rho_Core, rho_k=None):
                # No rho_k means unpolarized
                if rho_k is None: return 2.*rho_Core, 2.+0.*rho_Core, None, None
                
                # Compute rho
                rho = 2.*rho_Core + np.dot(rho_k, ff_k)
                rfrho = 2.**1.5*rho_Core + np.dot(rho_k, ff_k**1.5)

                fb = (rfrho/rho)**2 # fbar for exchange

                r_Core = rho_Core/rho
                r_k = rho_k/rho[:,None]

                Num = -2.*r_Core
                Den =  4.*r_Core
                for k1 in range(len(kAct)):
                    for k2 in range(len(kAct)):
                        Num += Px_kk[k1,k2]*r_k[:,k1]*r_k[:,k2]
                        Den += PH_kk[k1,k2]*r_k[:,k1]*r_k[:,k2]
                Gx = Num/Den

                fb_c = 2*Gx/(1+3*Gx)

                return rho, fb, fb_c, None

            def Deriv_Fn_of_f(f): return np.sqrt(f), 0

        elif self.eLDAType in ('NO', 'OP'):
            #print("Running in optimized and normed mode") # TEMP
            def GetLocalProps(rho_Core, rho_k=None):
                # No rho_k means unpolarized
                if rho_k is None: return 2.*rho_Core, 2.+0.*rho_Core, None, None
                
                # Compute rho
                rho = 2.*rho_Core + np.dot(rho_k, ff_k)

                # Compute rhoa and rhob
                a = self.eLDAProps['a']
                b = 3 - a
                
                rhoa = (2**a)*rho_Core + np.dot(rho_k, ff_k**a)
                rhob = (2**b)*rho_Core + np.dot(rho_k, ff_k**b)

                fb = (rhoa/rho) * (rhob/rho)

                return rho, fb, fb, (rhoa/rho, rhob/rho)

            def Deriv_Fn_of_f(f):
                if f>zero_round: return f**(-2/3), f**(5/3)
                else: return 0., 0.

        else:
            #print("Running in regular mode") # TEMP
            def GetLocalProps(rho_Core, rho_k=None):
                # No rho_k means unpolarized
                if rho_k is None: return 2.*rho_Core, 2.+0.*rho_Core, None, None
                
                # Compute rho
                rho = 2.*rho_Core + np.dot(rho_k, ff_k)

                # Copmute frho
                frho = 4.*rho_Core + np.dot(rho_k, ff_k**2)
                
                return rho, frho/rho, None, None

        #################################################################################
        
        # Initialise to zero
        Fxc = { k: 0.*DCore for k in kList }

        self.ELDAQuadrature = {
            'N':0., 'NCore':0.,
            'N_k': np.zeros((len(kAct),)),
            'Exc': 0.,
            'Ex_unp': 0.,
            'Ex_pol': 0.,
            'fb': 0.,
        }
        
        # Loop over the blocks
        for GridProps in self.Grid:
            # Obtain block information
            w = GridProps['w'] # Grid weights
            lpos = GridProps['lpos'] # Local basis functions
            lphi = GridProps['lphi'] # phi for local basis functions

            # lD = D[(lpos[:, None], lpos)]  -- notation

            # Compute core density
            lD = DCore[(lpos[:, None], lpos)]
            rho_Core = np.einsum('xp,pq,xq->x', lphi, lD, lphi, optimize=True)

            # Ensure density is never zero
            rho_Core = np.maximum(rho_Core, rho_Min)


            if len(kAct)>0:
                # Compute orbital densities for active orbitals
                rho_k = ( lphi.dot( C[(lpos[:, None], kAct)] ) )**2
            else:
                rho_k = None

            rho, fb, fb_c, DensExtra = GetLocalProps(rho_Core, rho_k=rho_k)
                

            rs = 0.62035/rho**(1/3)

            epsxc, depsxc_rs, depsxc_fb = LDA_cofe_deriv(rs, fb, fb_c = fb_c, Sum=True)
                
            self.ELDAQuadrature['N'] += np.dot(w, rho)
            self.ELDAQuadrature['NCore'] += np.dot(w, rho_Core)
            if not(rho_k) is None:
                self.ELDAQuadrature['N_k'] += np.dot(w, rho_k)
            self.ELDAQuadrature['fb'] += np.dot(w, rho*fb)

            self.ELDAQuadrature['Exc'] += np.dot(w, rho*epsxc)

            ex_unp = -0.738559 * rho**(4/3)
            self.ELDAQuadrature['Ex_unp'] += np.dot(w, ex_unp)
            self.ELDAQuadrature['Ex_pol'] += np.dot(w, ex_unp*(2./fb)**(1/3))

            if (fb.max()>(2.+zero_round)) or (fb.max()<(1.-zero_round)):
                print("Warning, fb out of bounds: [ %10.6f %10.6f ]"\
                      %(fb.min(), fb.max()))

            # Derivative with respect to density
            vxc_C = epsxc - rs/3 * depsxc_rs
            vxc_V1 = None
            vxc_V2 = None
            
            if self.IgnorefDeriv:
                vxc_V1 = None
                vxc_V2 = None
            elif self.eLDAType in ('OT', 'ON'):
                # For on-top modes  - based on fb = (Sum f_i^1.5 n_i / Sum f_i n_i)^2
                # v_k = eps - rs/3 * deps_rs + 2(sqrt(f_k fb) - fb) * deps_fb
                #     = (eps - rs/3 * deps_rs - 2 * fb * deps_fb) + 2 * f_k * sqrt(fb) * deps_fb

                # Constant and varying parts
                vxc_C += - 2. * fb * depsxc_fb
                vxc_V1 = 2. * np.sqrt(fb) * depsxc_fb
                vxc_V2 = None
            elif self.eLDAType in ('NO', 'OP'):
                # For optimized and normed mode
                # - based on fb = (na/n)*(nb/n)
                # v_k = (eps - rs/3 * deps_rs - 2 * fb * deps_fb) \
                #       + f_k**(a-1) * rhob/rho + f_k**(b-1) * rhoa/rho

                if DensExtra is None:
                    1
                else:
                    # Constant and varying parts
                    vxc_C += - 2. * fb * depsxc_fb
                    vxc_V1 = DensExtra[1] * depsxc_fb
                    vxc_V2 = DensExtra[0] * depsxc_fb
            else:
                # For other modes - based on fb = Sum f_i^2 n_i / Sum f_i n_i
                # v_k = eps - rs/3 * deps_rs + (f_k - fb) * deps_fb
                #     = (eps - rs/3 * deps_rs - fb * deps_fb) + f_k * deps_fb

                # Constant and varying parts
                vxc_C += - fb * depsxc_fb
                vxc_V1 = depsxc_fb
                vxc_V2 = None

            lV_C = np.einsum('xp,x,x,xq->pq', lphi, vxc_C, w, lphi, optimize=True)
            lV_C = 0.5*(lV_C + lV_C.T) # Ensure symmetry

            if not(vxc_V1 is None):
                lV_V1 = np.einsum('xp,x,x,xq->pq', lphi, vxc_V1, w, lphi, optimize=True)
                lV_V1 = 0.5*(lV_V1 + lV_V1.T) # Ensure symmetry
            else:
                lV_V1 = 0.
                
            if not(vxc_V2 is None):
                lV_V2 = np.einsum('xp,x,x,xq->pq', lphi, vxc_V2, w, lphi, optimize=True)
                lV_V2 = 0.5*(lV_V2 + lV_V2.T) # Ensure symmetry
            else:
                lV_V2 = 0.
                
            

            # Add the temporary back to the larger array by indexing, ensure it is symmetric
            for k in kList:
                if k>=len(f): f_k=0.
                else: f_k = f[k]

                Pre1, Pre2 = Deriv_Fn_of_f(f_k)
                Fxc[k][(lpos[:, None], lpos)] += lV_C + Pre1 * lV_V1 + Pre2 * lV_V2

        self.ELDAQuadrature['fb'] /= self.ELDAQuadrature['N']
        #print("fbar = %10.6f"%(self.ELDAQuadrature['fb']))

        # Symmetrize
        k0 = min(list(Fxc))
        for k in Fxc:
            Fxc[k] = (Fxc[k] + Fxc[k].T)/2
            if k>k0: Fxc[k] = Fxc[k0] #### TEMP

        return self.ELDAQuadrature['Exc'], Fxc

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
        fLDA, NCore, NOcc, PProps = GetPropsLDA(Plan)
        ExcLDA, FxcLDA = self.ComputeFxc_eLDA(fLDA, C=C, kList=[0,],
                                              NCore = NCore, NOcc = NOcc, PProps=PProps)

        E += ExcLDA
        F += FxcLDA[0]
        
        self.LastEns['xcDFA'] = 0.
        self.LastEns['xcDFA Parts'] = []

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

    # Get the energy and the orbital Fock matrices
    # Note, only computes different Fock matrices
    def GetEnergyFocks(self, Plan_, C=None,
                       Raw=False):
        if C is None: C = self.CE

        if Plan_ is None: return None, None, None
        
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
            kUnique = [kLow, kHi]
        else:
            FockMap = { kHi:[] }
            kUnique = [kHi,]
            
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
                
        # Add the DFA terms from the weighted sum of
        # existing DFAs (with appropriate scaling of x)
        fLDA, NCore, NOcc, PProps = GetPropsLDA(Plan)
        CT = C[:,:len(fLDA)]
        ExcLDA, self.LastEns['FxcLDA'] \
            = self.ComputeFxc_eLDA(fLDA, C=C, kList=kUnique,
                                   NCore = NCore, NOcc = NOcc, PProps=PProps)


        # Contains effective Fock operators
        # F_k = 1/fk d/dphi_k^* E[{phi}]
        # Note the normalization using fk
        
        
        FockList = {}
        for k in kUnique:
            # Skip when f[k] is zero (undefined == 0)
            if k>=N_f: continue
            if f[k]==0.: continue
            
            FockList[k] = self.Engine.T_ao + self.Engine.V_ao # Start with TV

            # Get the plan
            FPlan = PlanDeriv(Plan, k, IsEnergy=True)

            # Process the 1RDM sums
            for Key in ('Hx',):
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

            FockList[k] += self.LastEns['FxcLDA'][k]

            # Handle the extra continbutions
            for K, (_, WF, _, _, _) in enumerate(FPlan['Extra']):
                FockList[k] += WF * self.LastEns['Extra Parts'][K][1]

        # Return:
        # the energy E
        # unique Fock operators in FockList
        # map from k to FockList

        if Raw:
            # Return the unprcoessed quantities
            return E, FockList, FockMap

        # Do some additional processing to:
        # * Deal with the unoccupied orbitals
        # * Pad any holes
        # * Convert things to dictionaries for easy use
        
        # Project Fock operators onto orbitals
        MapBack = np.zeros((NOrb,), dtype=int)
        for k in FockList:
            MapBack[FockMap[k]] == k
            FockList[k] = (C.T).dot(FockList[k]).dot(C)

        if len(Plan['kTo'])==0:
            # If kTo is unspecified use the Fock operator for the highest orbital
            # for unoccupied
            FUnocc = FockList[MapBack[kHi-1]]*1.
        else:
            # If kTo is specified use their average Fock operator for unoccupied
            w = 1/len(Plan['kTo'])
            FUnocc = 0.
            for k in Plan['kTo']:
                FUnocc += w * FockList[k]

        FockList[kHi] = FUnocc

        # Tidy up the maps to reduce accidental dupes and sort
        for k in FockMap:
            FockMap[k] = np.array(sorted(list(set(FockMap[k]))))
            
        return E, FockList, FockMap

    def ShowQuadrature(self):
        print("els: N = %10.3f, N_Core = %10.3f, N_k = [ %s ]"\
              %( self.ELDAQuadrature['N'], self.ELDAQuadrature['N'],
                 " ".join(["%10.3f"%(x) for x in self.ELDAQuadrature['N_k']]) )
        )
        print("Ens: xc = %10.2f, x unp = %10.2f, x enh = %10.2f [eV]"\
              %( eV*self.ELDAQuadrature['Exc'],
                 eV*self.ELDAQuadrature['Ex_unp'],
                 eV*(self.ELDAQuadrature['Ex_pol']
                     - self.ELDAQuadrature['Ex_unp']), )                 
        )
