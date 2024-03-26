import psi4
import numpy as np
import scipy.linalg as la

import numpy.random as np_ra

from psi4Engine.LibPairDens import *

eV = 27.211

zero_round = 1e-5

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

# INTERNAL ROUTINE - One-spot rs-hybrid handler
def RSHybridParams(alpha, beta, omega):
    # alpha E_x + beta E_x^{lr} + beta E_x^{sr-DFA} + (1-alpha-beta) E_x^{DFA}
    WDFA = 1. - alpha - beta
    WDFA_SR = beta
    WHF = alpha
    WHF_LR = beta

    # See C:\Users\tgoul\Dropbox\Collabs\Ensemble\EGKS\Implementation\psi4-Notes.pdf
    
    return WDFA, WDFA_SR, WHF, WHF_LR

################################################################################################
# Handle PBE0_XX calculations
# PBE0_[alpha] - hybrid
#
# wPBE0_[alpha]_[omega](_[beta]) - rs hybrid
#   alpha and beta are %s and omega is in Bohr
#   beta : 1-alpha if unspecified
# Yields
# DFA = alpha E_x^{SR} + beta E_x^{lr} + E_x^{PBE,rest} + E_c^{PBE}
# where 'rest' indicates the missing sr and lr contributions
################################################################################################
def TextDFA(DFA):
    if DFA[:5].lower()=="pbe0_":
        X = DFA.split('_')
        alpha = float(X[1])/100.
        if len(X)>2: f_c = max(float(X[2])/100.,zero_round)
        else: f_c = 1.
        return {
            'name':DFA,
            'x_functionals': {"GGA_X_PBE": {"alpha": 1.-alpha, }},
            'c_functionals': {"GGA_C_PBE": {"alpha": f_c, }},
            'x_hf': {"alpha": alpha, },
        }
    elif DFA[:5].lower()=="pbe_h":
        return {
            'name':DFA,
            'x_functionals': {"GGA_X_HJS_PBE": {"alpha": 1., "omega": 10., }},
            'c_functionals': {"GGA_C_PBE": {"alpha": 1., }},
            'x_hf': {"alpha": 0., },
        }
    elif DFA[:5].lower()=="wpbe_":
        # Format
        # wpbe_[alpha%]_[omega]_[beta%]_[corr%]_[lda%]
        #
        # Only alpha needs to be specificiec - others default to:
        #    omega=0.3, beta=1-alpha, corr=100%, lda=0%
        #
        # E.g. wpbe_25_0.5 gives alpha=0.25, omega=0.5, beta=0.75, corr=1.0, lda=0.0
        
        
        X = DFA.split('_')
        alpha = float(X[1])/100.
        if len(X)>2: omega = float(X[2])
        else: omega = 0.3
        if len(X)>3: beta = float(X[3])/100
        else: beta = 1.-alpha
        if len(X)>4: WC = float(X[4])/100
        else: WC = 1.
        if len(X)>5: WLDA_SR = float(X[4])/100
        else: WLDA_SR = 0.

        WDFA, WDFA_SR, WHF, WHF_LR = RSHybridParams(alpha, beta, omega)
        
        DFADef =  {
            'name':DFA,
            'x_hf': {"alpha":WHF, "beta":WHF_LR, "omega":omega, }, 
            'x_functionals': {"GGA_X_HJS_PBE": {"alpha":WDFA_SR - WLDA_SR, "omega":omega, }, },
            'c_functionals': {"GGA_C_PBE": {"alpha":WC, } },
        }
        if np.abs(WDFA)>zero_round:
            DFADef["x_functionals"]["GGA_X_PBE"] = {"alpha":WDFA, }
        if np.abs(WLDA_SR)>zero_round:
            DFADef["x_functionals"]["LDA_X_ERF"] = {"alpha":WLDA_SR, "omega":omega, }
            print("Short-range LDA does not appear to be implemented in psi4")
            quit()
        return DFADef
    else:
        return DFA


################################################################################################
##### Two-body integrals ####
# This call psi4's internal routines and is fairly fast
################################################################################################
class JKHelper:
    def __init__(self, wfn, omega=None, mem=None,
                 Debug=False):
        self.NBas = wfn.nmo()
        self.Has_RS = not(omega is None)

        self.JK = None
        
        if wfn.jk() is None:
            self.NewJK(wfn.basisset(), omega, mem=mem)
        else:
            #self.JK = wfn.jk()
            #self.JK.set_do_wK(True)
            #self.JK.initialize()
            wfn.jk().finalize()
            self.NewJK(wfn.basisset(), omega, mem=mem)

        self.Debug = Debug


    def __del__(self):
        #print("Closing JK helper")
        if not(self.JK is None): self.JK.finalize()
        
    def NewJK(self, basis, omega, mem=None):
        # Finalize the current if required
        if not(self.JK is None): self.JK.finalize()

        self.JK = psi4.core.JK.build(basis, jk_type="DF",
                                     do_wK=self.Has_RS,
                                     memory=128*1024*1024)
        if mem is None:
            mem = self.JK.memory_estimate()
            MaxMem = int(psi4.get_memory()*0.8)
            if mem>MaxMem:
                print("Need approximately 1024^%4.1f bytes out of 1024^%4.1f"\
                      %(np.log(mem)/np.log(1024), np.log(MaxMem)/np.log(1024) ))
                mem = MaxMem
                
        self.JK.set_memory( mem )
        self.JK.set_wcombine(False) # Comment out for older psi4
        if self.Has_RS:
            self.JK.set_omega(omega)
            self.JK.set_omega_alpha(0.0) # Comment out for older psi4
            self.JK.set_omega_beta(1.0) # Comment out for older psi4
            self.JK.set_do_wK(True)

        self.JK.initialize()
        

    def FJ(self, C, CR=None):
        return self.FMaster(C, CR, "J")
    def FK(self, C, CR=None):
        return self.FMaster(C, CR, "K")
    def FK_w(self, C, CR=None):
        return self.FMaster(C, CR, "K_w")
    
    def FMaster(self, C, CR=None, Mode='J'):
        if self.Debug: print("Getting Fock operator %s"%(Mode))
        if not(CR is None):
            if len(CR.shape)==1:
                CRM = psi4.core.Matrix.from_array(CR.reshape((self.NBas,1)))
            else:
                CRM = psi4.core.Matrix.from_array(CR)
            
        if len(C.shape)==1:
            CM = psi4.core.Matrix.from_array(C.reshape((self.NBas,1)))
        else:
            CM = psi4.core.Matrix.from_array(C)
            
        self.JK.C_clear()
        self.JK.C_left_add(CM)
        if CR is None:
            self.JK.C_right_add(CM)
        else:
            self.JK.C_right_add(CRM)
            
        self.JK.compute()
            
        if Mode.upper()=='J':
            return self.JK.J()[0].to_array(dense=True)
        elif Mode.upper()=='K':
            return self.JK.K()[0].to_array(dense=True)
        elif Mode.upper() in ('WK', 'KW', 'K_W'):
            if self.Has_RS:
                return self.JK.wK()[0].to_array(dense=True)
            else: return 0.
        else:
            return self.JK.J()[0].to_array(dense=True), \
                self.JK.K()[0].to_array(dense=True)
        


##### Process density-fitting ####
# THIS ROUTINE IS RETAINED BUT NEVER USED

################################################################################
# Note - all the ERI needs rewriting
# See ~/Molecules/Misc-Code/JK-Tests.py for help
################################################################################

def GetDensityFit(wfn, basis, mints, omega=None,
                  DFName=None, DFMode='RIFIT',
                  return_all = False):
    if DFName is None: DFName = basis.name()
    aux_basis = psi4.core.BasisSet.build\
        (wfn.molecule(), "DF_BASIS_SCF", "",
         DFMode, DFName)
    zero_basis = psi4.core.BasisSet.zero_ao_basis_set()
    SAB = np.squeeze(mints.ao_eri(aux_basis, zero_basis, basis, basis))
    metric = mints.ao_eri(aux_basis, zero_basis, aux_basis, zero_basis)
    metric.power(-0.5, 1e-14)
    metric = np.squeeze(metric)
    ERIA = np.tensordot(metric, SAB, axes=[(1,),(0,)])

    if not(omega is None):
        # Get the density fit business
        # Need to work out how to do density fit on rs part
        IntFac_Apq = psi4.core.IntegralFactory\
            (aux_basis, zero_basis, basis, basis)
        IntFac_AB  = psi4.core.IntegralFactory\
            (aux_basis, zero_basis, aux_basis, zero_basis)
        SAB_w = np.squeeze(
            mints.ao_erf_eri(omega, IntFac_Apq) )
        metric_w = mints.ao_erf_eri(omega, IntFac_AB )
        metric_w.power(-0.5, 1e-14)
        metric_w = np.squeeze(metric_w)
        
        # ERI in auxilliary - for speed up
        ERIA_w = np.tensordot(metric_w, SAB_w, axes=[(1,),(0,)])
    else:
        ERIA_w, SAB_w, metric_w = None, None, None

    if return_all:
        return ERIA, ERIA_w, SAB, SAB_w, metric, metric_w
    else:
        return ERIA, ERIA_w

################################################################################################
##### This is a hack to convert a UKS superfunctional to its RKS equivalent
################################################################################################
# Internal routine
# https://github.com/psi4/psi4/blob/master/psi4/driver/procrouting/dft/dft_builder.py#L251
sf_from_dict =  psi4.driver.dft.build_superfunctional_from_dictionary
# My very hacky mask
def sf_RKS_to_UKS(DFA):
    DFA_Dict = { 'name':DFA.name()+'_u'}
    DFA_Dict['x_functionals']={}
    DFA_Dict['c_functionals']={}
    for x in DFA.x_functionals():
        Name = x.name()[3:]
        alpha = x.alpha()
        omega = x.omega()

        if np.abs(alpha)>zero_round:
            if omega==0.:
                DFA_Dict['x_functionals'][Name] = {"alpha": alpha, }
            else:
                DFA_Dict['x_functionals'][Name] = {"alpha": alpha, "omega": omega, }
    for c in DFA.c_functionals():
        Name = c.name()[3:]
        alpha = c.alpha()
        omega = c.omega()

        if np.abs(alpha)>zero_round:
            if omega==0.:
                DFA_Dict['c_functionals'][Name] = {"alpha": alpha, }
            else:
                DFA_Dict['c_functionals'][Name] = {"alpha": alpha, "omega": omega, }


    npoints = psi4.core.get_option("SCF", "DFT_BLOCK_MAX_POINTS")
    DFAU, _ = sf_from_dict(DFA_Dict,npoints,1,False)
    return DFAU
################################################################################################
##### End hack
################################################################################################

# For nice debug printing
def NiceArr(X):
    return "[ %s ]"%(",".join(["%8.3f"%(x) for x in X]))
def NiceArrInt(X):
    return "[ %s ]"%(",".join(["%5d"%(x) for x in X]))
def NiceMat(X):
    N = X.shape[0]
    if N==0:
        return "[]"
    elif N==1:
        return "["+NiceArr(X[0,:])+"]"
    elif N==2:
        return "["+NiceArr(X[0,:])+",\n "+NiceArr(X[1,:])+"]"
    else:
        R = "["
        for K in range(N-1):
            R+=NiceArr(X[K,:])+",\n "
        R+=NiceArr(X[N-1,:])+"]"
        return R


################################################################################################
# Get the degeneracy of each orbital - not used
################################################################################################
def GetDegen(epsilon, eta=zero_round):
    Degen = np.zeros((len(epsilon),),dtype=int)
    for k in range(len(epsilon)):
        ii =  np.argwhere(np.abs(epsilon-epsilon[k])<eta).reshape((-1,))
        Degen[k] = len(ii)
    return Degen

################################################################################################
# This code handles degeneracies detected and used by psi
# Do not worry about the details
################################################################################################
class SymHelper:
    def __init__(self, wfn):
        self.NSym = wfn.nirrep()
        self.NBasis = wfn.nmo()
        
        self.eps_so = wfn.epsilon_a().to_array()
        self.C_so = wfn.Ca().to_array()
        self.ao_to_so = wfn.aotoso().to_array()
        
        if self.NSym>1:
            self.eps_all = np.hstack(self.eps_so)
            self.k_all = np.hstack([ np.arange(len(self.eps_so[s]), dtype=int)
                                       for s in range(self.NSym)])
            self.s_all = np.hstack([ s * np.ones((len(self.eps_so[s]),), dtype=int)
                                       for s in range(self.NSym)])
        else:
            self.eps_all = self.eps_so * 1.
            self.k_all = np.array(range(len(self.eps_all)))
            self.s_all = np.zeros((len(self.eps_all),), dtype=int)

        self.ii_sorted = np.argsort(self.eps_all)
        self.eps_sorted = self.eps_all[self.ii_sorted]
        self.k_sorted = self.k_all[self.ii_sorted]
        self.s_sorted = self.s_all[self.ii_sorted]

        self.ks_map = {}
        for q in range(len(self.ii_sorted)):
            self.ks_map[(self.s_sorted[q], self.k_sorted[q])] = q

    # Do a symmetry report to help identifying orbitals
    def SymReport(self, kh, eta=zero_round):
        epsh = self.eps_sorted[kh] + eta
        print("Orbital indices by symmetry - | indicates virtual:")
        for s in range(self.NSym):
            Str = "Sym%02d : "%(s)
            eps = self.eps_so[s]
            if not(hasattr(eps, '__len__')) or len(eps)==0: continue

            kk_occ = []
            kk_unocc = []
            for k, e in enumerate(eps):
                if e<epsh: kk_occ += [ self.ks_map[(s,k)] ]
                else: kk_unocc += [ self.ks_map[(s,k)] ]

            Arr = ["%3d"%(k) for k in kk_occ] + [" | "] \
                + ["%3d"%(k) for k in kk_unocc]
            if len(Arr)<=16:
                print("%-8s"%(Str) + " ".join(Arr))
            else:
                for k0 in range(0, len(Arr), 16):
                    kf = min(k0+16, len(Arr))
                    if k0==0:
                        print("%-8s"%(Str) + " ".join(Arr[k0:kf]))
                    else:
                        print(" "*8 + " ".join(Arr[k0:kf]))

    # Report all epsilon
    def epsilon(self):
        return self.eps_sorted

    # Report a given orbital, C_k
    def Ck(self, k):
        if self.NSym==1:
            return self.C_so[:,k]
        else:
            s = self.s_sorted[k]
            j = self.k_sorted[k]

            return self.ao_to_so[s].dot(self.C_so[s][:,j])

    # Report all C
    def C(self, CIn=None):
        if CIn is None: CIn = self.C_so
        
        if self.NSym==1:
            return CIn * 1.
        else:
            C = np.zeros((self.NBasis, self.NBasis))
            k0 = 0
            for k in range(self.NSym):
                C_k = self.ao_to_so[k].dot(CIn[k])
                dk = C_k.shape[1]
                C[:,k0:(k0+dk)] = C_k
                k0 += dk
            return C[:,self.ii_sorted]

    # Convert the so matrix to dense form
    def Dense(self, X):
        if self.NSym==1:
            return X
        else:
            XX = 0.
            for s in range(self.NSym):
                XX += self.ao_to_so[s].dot(X[s]).dot(self.ao_to_so[s].T)
            return XX

    # Solve a Fock-like equation using symmetries
    # if k0>0 use only the subspace spanned by C[:,k0:]
    def SolveFock(self, F, S=None, k0=-1):
        # Note, k0>=0 means to solve only in the basis from C[:,k0:]
        if self.NSym==1:
            if k0<=0:
                return la.eigh(F, b=S)
            else:
                # FV = SVw
                # V=CU
                # FCU = SCUw
                # (C^TFC)U = (C^TSC)Uw
                # XU = Uw
                
                C = self.C()[:,k0:]
                F_C = (C.T).dot(F).dot(C)
                w, U = la.eigh(F_C)
                return w, C.dot(U)
        else:
            k0s = [0]*self.NSym
            ws = [None]*self.NSym
            Cs = [None]*self.NSym

            if k0>0:
                # Use no terms
                for s in range(self.NSym):
                    k0s[s] = self.NBasis
                    
                # Evaluate the smallest k value for each symmetry
                for i in range(k0,self.NBasis):
                    s = self.s_sorted[i]
                    k0s[s] = min(k0s[s],self.k_sorted[i])

            for s in range(self.NSym):
                # Project onto the subset starting at k0s
                C_ao = self.ao_to_so[s].dot(self.C_so[s][:,k0s[s]:])
                F_C = (C_ao.T).dot(F).dot(C_ao)
                if not(S is None):
                    S_C = (C_ao.T).dot(S).dot(C_ao)
                else: S_C = None

                if F_C.shape[0]>0:
                    ws[s], Us = la.eigh(F_C, b=S_C)
                    Cs[s] = C_ao.dot(Us)
                else:
                    ws[s] = []
                    Cs[s] = [[]]

            # Project back onto the main set
            k0 = max(k0,0)
            w = np.zeros((self.NBasis - k0,)) + 200. # If errors, make sure they're high energy
            C = np.zeros((self.NBasis,self.NBasis - k0))
            
            #for i in self.ii_sorted[k0:]: # Old and I am sure not correct
            for i in range(k0, self.NBasis): # Index of orbital
                s = self.s_sorted[i] # Its symmetry
                k = self.k_sorted[i] # Its k value in the symmetry
                w[i-k0] = ws[s][k-k0s[s]] # Copy in - k0s is the smallest value in the subset
                C[:,i-k0] = Cs[s][:,k-k0s[s]] # Like above
                
            return w, C
        

#################################################################################################
# This is the main engine
#################################################################################################

class psi4Engine:
    ###################################################################################
    # Routines for handling instantiation, deletion and miscellaneous
    ###################################################################################

    # Initialise the engine
    #   must specify a wfn
    #   wfn_Ref - may also specify a reference wfn
    #   alpba, beta, omega - can pass hybrid details (not needed)
    #   ComputeERIA - evaluate the DF tensor [pq|A] (not needed)
    #   Report - larger number = more outputs
    def __init__(self, wfn, wfn_Ref = None,
                 alpha = None, beta = None, omega = None,
                 ComputeERIA = False,
                 Report = 0):
        self.Report = Report
        self.wfn = wfn
        if wfn_Ref is None:
            wfn_Ref = wfn

        self.SymHelp = SymHelper(wfn)
        
        self.Da = self.SymHelp.Dense(wfn_Ref.Da().to_array())
        self.Db = self.SymHelp.Dense(wfn_Ref.Db().to_array())
        self.D = self.Da + self.Db
        self.epsilon = self.SymHelp.epsilon()
        self.C = self.SymHelp.C()
        self.F = self.SymHelp.Dense(wfn_Ref.Fa().to_array())

        basis = wfn.basisset()
        self.basis = basis
        self.nbf = self.wfn.nmo() # Number of basis functions
        self.NAtom = self.basis.molecule().natom()

        if not(basis.has_puream()):
            print("Must use a spherical basis set, not cartesian")
            print("Recommend rerunning with def2 or cc-type basis set")
            quit()
       

        self.Enn = self.basis.molecule().nuclear_repulsion_energy()
        
        self.mints = psi4.core.MintsHelper(self.basis)

        self.S_ao = self.mints.ao_overlap().to_array(dense=True)
        self.T_ao = self.mints.ao_kinetic().to_array(dense=True)
        self.V_ao = self.mints.ao_potential().to_array(dense=True)
        self.H_ao = self.T_ao + self.V_ao

        self.Dip_ao = np.array([x.to_array(dense=True)
                                for x in self.mints.ao_dipole()])


        self._SetDFA(wfn)

        self.OverwriteHybrid(alpha=alpha, beta=beta, omega=omega)
        
        # Compute ERIA if asked
        if ComputeERIA:
            self.ERIA, self.ERIA_w \
                = GetDensityFit(wfn, self.basis, self.mints, self.omega)
        else: self.ERIA, self.ERIA_w = None, None


        self.JKHelp = JKHelper(wfn, self.omega)
        
        self.NBas = wfn_Ref.nmo()
        self.NOcc = (wfn_Ref.nalpha() + wfn_Ref.nbeta())//2 # Always round down to double occ
        self.kh = self.NOcc-1 # HOMO
        self.kl = self.kh+1 # LUMO

        # Symmetry group by basis function        
        self.Sym_k = self.SymHelp.s_sorted
        # Atom by basis function        
        self.Atom_k = np.array(
            [ self.basis.function_to_center(i) for i in range(self.nbf) ]
            )

        # Default occupation factors
        self.f_Occ = 2.*np.ones((self.NOcc,))
        
        if self.SymHelp.NSym>1 and self.Report>=0:
            self.SymHelp.SymReport(self.kh)

    # Delete the engine - ensures large tensors are removed
    def __del__(self):
        #print("Closing psi4Engine")
        if not(self.JKHelp is None): self.JKHelp.__del__()
        if not(self.ERIA is None): del self.ERIA

    # Internal routine to make sure the wfn is adapted properly for
    # ensemble calculations
    def _SetDFA(self, wfn):
        # These are used for DFA calculations
        self.VPot = wfn.V_potential() # Note, this is a VBase class
        try:
            self.DFA = self.VPot.functional()
        except:
            self.DFA = None
            self.VPot = None

        # Ensure we have UKS for the excitations
        if not(self.DFA is None):
            # Convert DFA from RKS to UKS
            self.DFAU = sf_RKS_to_UKS(self.DFA)
            # Make a new VPot for the UKS DFA
            self.VPot = psi4.core.VBase.build(self.VPot.basis(), self.DFAU, "UV")
            self.VPot.initialize()

        # Work out the range-separation and density functional stuff
        self.xDFA, self.xDFA_w = 0., 0.
        self.omega = None
        self.alpha = 0.
        self.beta = 0.
        if not(self.DFA is None):
            # Implement DFAs
            if self.DFA.is_x_hybrid():
                # Hybrid functional
                self.alpha = self.DFA.x_alpha()
                self.xDFA = 1. - self.alpha
                if self.DFA.is_x_lrc():
                    # Range-separated hybrid
                    self.omega = self.DFA.x_omega()
                    self.beta = self.DFA.x_beta()

                    if self.Report>=0:
                        print("# RS hybrid alpha = %.2f, beta = %.2f, omega = %.2f"\
                              %(self.alpha, self.beta, self.omega),
                              flush=True)

                    self.xDFA = 1. - self.alpha # 1. - self.alpha
                    self.xDFA_w = - self.beta # - self.beta
            else:
                # Conventional functional
                self.xDFA = 1.
                
            # psi4 matrices for DFA evaluation
            self.DMa = psi4.core.Matrix(self.nbf, self.nbf)
            self.DMb = psi4.core.Matrix(self.nbf, self.nbf)
            self.VMa = psi4.core.Matrix(self.nbf, self.nbf)
            self.VMb = psi4.core.Matrix(self.nbf, self.nbf)
        else:
            self.alpha = 1. # Pure HF theory
            self.xDFA = 0. # Pure HF theory

        self.Has_RS = not(self.omega is None)

    # Force the hybrid parameters (not recommended to be used)
    def OverwriteHybrid(self, alpha=None, beta=None, omega=None):
        if not(alpha is None): self.alpha = alpha
        if not(beta is None): self.beta = beta
        if not(omega is None): self.omega = omega
        self.Has_RS = not(self.omega is None)

    ###################################################################################
    # Routines for handling electron repulsion integrals (ERI)
    ###################################################################################

    # Update the JK helper for fast ERI evaluation
    def UpdateJK(self):
        self.JKHelp.NewJK(self.basis, self.omega)

    # Use this when we handle things internally (slower)
    # ERIA obeys [pq|rs] \approx \sum_A [pq|A][A|rs]
    def GetHalfERI(self, C1=None, C2=None, D=None,
                   K_w=True):
        # Use RS or full
        if K_w: ERIA = self.ERIA_w
        else: ERIA = self.ERIA

        # Must be specified
        if ERIA is None:
            print("Must set ComputeERIA when Engine is initialized (with RS hybrid if reqd)")
            quit()
            
        if not(D is None):
            return np.tensordot(ERIA, D, axes=((1,2),(0,1)))

        if not(C1 is None):
            X = np.tensordot(ERIA, C1, axes=((2,),(0,)))
            if not(C2 is None):
                return np.tensordot(X, X2, axes=((1,),(0,)))
            else:
                return X
        else:
            return ERIA

    # Get Fock operator for J type integral
    def GetFJ(self, CI, Pre=1.):
        if np.abs(Pre)<zero_round: return 0.
        else: return Pre*self.JKHelp.FJ(CI)

    # Get Fock operator for K type integral
    def GetFK(self, CI, Pre=1.):
        if np.abs(Pre)<zero_round: return 0.
        else: return Pre*self.JKHelp.FK(CI)

    # Get Fock operator for range-separated J type integral
    def GetFJ_w(self, CI, Pre=1.):
        if np.abs(Pre)<zero_round: return 0.
        print("****  Calling RS FJ -- weird! ****")
        quit()

    # Get Fock operator for range-separated K type integral
    def GetFK_w(self, CI, Pre=1.):
        if np.abs(Pre)<zero_round: return 0.
        else: return Pre*self.JKHelp.FK_w(CI)

    # Compute energies using occupation factors and orbitals
    #   f : occupation factors
    #   C : orbitals (defaults to internal if unspecified)
    #   Mode : 'J', 'K' or 'wK' (range-separated K)
    # returns an energy
    def GetEMaster_Occ(self, f, C=None, Mode='J'):
        if C is None: C = self.CE

        C = C[:,:len(f)]
        CR = C * f[None,:]

        F = self.JKHelp.FMaster(C, CR, Mode)
        if isinstance(F, np.ndarray):
            return 0.5*np.einsum('pk,pq,qk', C, F, CR)
        else:
            return 0.

    # Specialised versions of the above for J, K and RS K
    def EJ_Occ(self, f, C=None):
        return self.GetEMaster_Occ(f, C, 'J')

    def EK_Occ(self, f, C=None):
        return self.GetEMaster_Occ(f, C, 'K')

    def EK_w_Occ(self, f, C=None):
        return self.GetEMaster_Occ(f, C, 'wK')

    ###################################################################################
    # Routines for handling ensemble plans
    ###################################################################################
    
    # Return the DFA exchange-correlation contribution
    # Returns the energy and Fock operator
    def GetDFA(self, Da=None, Db=None, Pre=1.,
               BothSpin=False):
        if self.DFA is None:
            if BothSpin: return 0., 0., 0.
            else: return 0., 0.
        
        # Use internal Da by default
        if Da is None: Da = self.Da
        if Db is None: Db = Da

        if np.abs(Pre)<zero_round:
            if BothSpin: return 0., 0., 0.
            else: return 0., 0.
        
        self.DMa.np[:,:] = Da
        self.DMb.np[:,:] = Db
        self.VPot.set_D([self.DMa,self.DMb])
        self.VPot.compute_V([self.VMa,self.VMb])
        ExcDFA = Pre * self.VPot.quadrature_values()["FUNCTIONAL"]
        VxcDFA = Pre * self.VMa.to_array(dense=True)

        if BothSpin:
            return ExcDFA, VxcDFA, Pre * self.VMb.to_array(dense=True)

        return ExcDFA, VxcDFA

    # Return the Hartree and exchange contribution (three aliases)
    # Returns the energy and Fock operator
    def GetHF(self, **kwargs):
        return self.GetHx(**kwargs)
    def GetEXX(self, **kwargs):
        return self.GetHx(**kwargs)
    def GetHx(self, Ca=None, Cb=None,
              alphaH = 1., alpha = None, beta = None,
              BothSpin=False):
        if alpha is None: alpha = self.alpha
        if beta is None: beta = self.beta
        
        if Ca is None: Ca = self.C[:,:self.NOcc]
        
        if Cb is None:
            VHx  = self.GetFJ(Ca, Pre=2.*alphaH)
            VHx += self.GetFK(Ca, Pre=-alpha)
            VHx += self.GetFK_w(Ca, Pre=-beta)

            D = np.dot(Ca, Ca.T)*2.

            if np.abs(alphaH)>zero_round:
                EHx = 0.5 * np.tensordot(D, VHx)
            else:
                EHx = 0.
            if BothSpin: return EHx, VHx, VHx
        else:
            FJ   = self.GetFJ(Ca, Pre=alphaH) + self.GetFJ(Cb, Pre=alphaH)
            FKa  = self.GetFK(Ca, Pre=-alpha)
            FKa += self.GetFK_w(Ca, Pre=-beta)
            FKb  = self.GetFK(Cb, Pre=-alpha)
            FKb += self.GetFK_w(Cb, Pre=-beta)

            VHx = FJ + FKa

            Da = np.dot(Ca, Ca.T)
            Db = np.dot(Cb, Cb.T)

            if np.abs(alphaH)>zero_round:
                EHx = 0.5 * np.tensordot(Da+Db, FJ)
            else:
                EHx = 0.
                
            if np.abs(alpha)>zero_round or np.abs(beta)>zero_round:
                EHx += 0.5 * ( np.tensordot(Da, FKa) + np.tensordot(Db, FKb) )
                
            if BothSpin: return EHx, VHx, FJ + FKb

        return EHx, VHx


    # Return the 'extra' contribution if J and K integrals
    # Returns the energy and Fock operator
    def GetExtra(self, C=None, Extra=None):
        # Extra = [(PreE,PreF,k1,k2,Kind), ...]
        # PreE and PreF are prefactors for E and F
        # k1 and k2 are orbitals
        # Kind is J, K or K_w

        if C is None: C = self.C
        if Extra is None: return 0., 0.
        
        EExtra, FExtra = 0., 0.
        for (PreE, PreF, k1, k2, Kind) in Extra:
            C1 = C[:,k1]
            C2 = C[:,k2]
            F11 = self.JKHelp.FMaster(C1, C1, Kind)
                
            EExtra += PreE * (C2).dot(F11).dot(C2)
            FExtra += PreF * F11

        return EExtra, FExtra

    # Return the total contribution from a plan
    # Returns the energy and Fock operator
    def GetEnergy(self, C=None, f = None,
                  Extra = None):
        if C is None: C = self.C
        if f is None: f = self.f_Occ

        if np.mean(np.abs(f - np.round(f)))>zero_round:
            print("Warning! Your occupation factors are non-integer - this is not the right routine")

        fa = np.minimum(f, 1.)
        fb = f - fa

        # Handle the 1-RDM
        nf = len(f)
        D = np.einsum('pk,qk,k->pq', C[:,:nf], C[:,:nf], f)

        FTV = self.T_ao + self.V_ao
        ETV = np.vdot(FTV, D)

        # Handle the Hartree and exchange and correlation
        if np.sum(np.abs(fa-fb))<zero_round: # UKS
            Ca = self.C[:,:nf][:,np.abs(fa-1.)<zero_round]

            EHx, VHx = self.GetHx(Ca=Ca)
            ExcDFA, VxcDFA = self.GetDFA(Da = D/2.)
        else: # RKS
            Ca = self.C[:,:nf][:,np.abs(fa-1.)<zero_round]
            Cb = self.C[:,:nf][:,np.abs(fb-1.)<zero_round]

            EHx, VHx = self.GetHx(Ca=Ca, Cb=Cb)

            Da = np.dot(Ca, Ca.T)
            Db = np.dot(Cb, Cb.T)

            ExcDFA, VxcDFA = self.GetDFA(Da = Da, Db = Db)

        E = ETV + EHx + ExcDFA + self.Enn
        F = FTV + VHx + VxcDFA

        if not(Extra is None):
            EExtra, FExtra = self.GetExtra(C, Extra)

            E += EExtra
            F += FExtra

        if self.Report>=3:
            print("ETV = %10.5f, EHx = %10.5f, ExcDFA = %10.5f"\
                  %(ETV, EHx, ExcDFA) )

        return E, F

    # Solve a Fock operator - returns eps, C
    def Solve(self, F, **kwargs):
        return self.SolveFock(F, **kwargs)
    def SolveFock(self, F, **kwargs):
        return self.SymHelp.SolveFock(F, **kwargs)

    ###################################################################################
    # Routines for custom DFAs
    ###################################################################################

    # Obtain the DFT grid as a list of blocks 'Grid' where
    #   Grid[K]['NP'] is the number of points in the block
    #   Grid[K]['w'] are quadrature weights in the block
    #   Grid[K]['lpos'] is the indices of GTOs that are non-zero in the block
    #   Grid[K]['lphi'] are the values of non-zero GTOS in the block
    def GetDFTGrid(self, GGA=False, MGGA=False, delta=0.):
        Grid = [None]*self.VPot.nblocks()

        basis = self.wfn.basisset()
        
        for b in range(self.VPot.nblocks()):
            block = self.VPot.get_block(b)
            NP = block.npoints()

            Grid[b] = {'NP':NP, 'w': block.w().to_array() }

            blockopoints = psi4.core.BlockOPoints\
                ( block.x(), block.y(), block.z(), block.w(),
                  psi4.core.BasisExtents(basis,delta) )

            lpos = np.array(blockopoints.functions_local_to_global())

            funcs = psi4.core.BasisFunctions(basis, NP, basis.nbf())
            funcs.compute_functions(blockopoints)
            lphi = funcs.basis_values()["PHI"].to_array(dense=True)

            Grid[b]['lpos'] = lpos
            Grid[b]['lphi'] = lphi[:, lpos]

        return Grid
            

    # Given alpha and beta density matrices Da and Db return
    # positions, weights, rs, zeta and s on a grid
    def GetGGAProps(self, Da, Db):
        xyz, w, (rhoa,rhob), _ = GetDensities(None, D1List=(Da,Db), wfn=self.wfn,
                                              return_w = True, return_xyz = True)
        rho = rhoa + rhob
        rho_m = np.maximum(rho, 1e-18)
        zeta = np.abs(rhoa-rhob)/rho_m
        
        rs, s, _ = GetGGAProps(self.wfn, Da+Db)

        return {'xyz':xyz, 'w':w, 'rho':rho, 'rs':rs, 's':s, 'zeta':zeta}

    # Given occupation factors f and orbitals C return
    # positions, weights, rs, zeta and fbar on a grid
    def GetELDAProps(self, f=None, C=None):
        if self.DFA is None: return None
        if f is None: f = self.f_Occ
        if C is None: C = self.C

        C = C[:,:len(f)]

        D = np.einsum('pk,qk,k->pq', C, C, f)
        Da = np.einsum('pk,qk,k->pq', C, C, np.minimum(f,1.))
        fD = np.einsum('pk,qk,k->pq', C, C, f**2)

        xyz, w, (rho,rhoa,frho), _ = GetDensities(None, D1List=(D,Da,fD), wfn=self.wfn,
                                                  return_w = True, return_xyz = True)
        rho_m = np.maximum(rho, 1e-18)
        fbar = frho/np.maximum(rho, rho_m)
        zeta = np.abs(2*rhoa-rho)/rho_m
        rs = 0.62035 * rho_m**(-1/3)

        return {'xyz':xyz, 'w':w, 'rho':rho, 'rs':rs, 'zeta':zeta, 'fbar':fbar}

    # Given occupation factors f and orbitals C return
    # positions, weights, rs, zeta, s and fbar on a grid
    def GetEGGAProps(self, f=None, C=None):
        if self.DFA is None: return None
        if f is None: f = self.f_Occ
        if C is None: C = self.C

        C = C[:,:len(f)]

        D = np.einsum('pk,qk,k->pq', C, C, f)
        Da = np.einsum('pk,qk,k->pq', C, C, np.minimum(f,1.))
        fD = np.einsum('pk,qk,k->pq', C, C, f**2)

        xyz, w, (rho,rhoa,frho), _ = GetDensities(None, D1List=(D,Da,fD), wfn=self.wfn,
                                                  return_w = True, return_xyz = True)
        rho_m = np.maximum(rho, 1e-18)
        fbar = frho/rho_m
        zeta = np.abs(2*rhoa-rho)/rho_m
        rs, s, _ = GetGGAProps(self.wfn, D)

        return {'xyz':xyz, 'w':w, 'rho':rho, 'rs':rs, 's':s, 'zeta':zeta, 'fbar':fbar}

    # Not used and not working properly
    def Gradient_ao(self, atom=0, Kind='P'):
        if Kind.upper()[0] in ('V', 'P'):
            X = self.mints.ao_oei_deriv1("POTENTIAL", atom)
        elif Kind.upper()[0] in ('T', 'K'):
            X = self.mints.ao_oei_deriv1("KINETIC", atom)
        elif Kind.upper()[0] in ('S', 'O'):
            X = self.mints.ao_oei_deriv1("OVERLAP", atom)
        else:
            return None

        return [x.to_array(dense=True) for x in X]

    # Not used and not working properly
    def Gradient_NN(self):
        return self.basis.molecule().nuclear_repulsion_energy_deriv1().to_array(dense=True)
    
            
if __name__ == "__main__":
    psi4.set_output_file("__Engine.out")

    psi4.set_options({
        'basis' : 'def2-tzvp',
        'reference': 'rhf',
    })
    
    MolStr = """
C  0.000  0.000  0.000
H  0.000 -0.000  1.111
H  1.087 -0.000 -0.229
"""
    fgs = [2.,2.,2.,2.]
    fts = [2.,2.,2.,1.,1.]

    for DFA in ('scf', 'pbe', 'pbe0', 'wb97x'):
        print(DFA)
        
        psi4.geometry("0 1\n"+MolStr)
        psi4.set_options({
            'reference': 'rhf',
        })
        E0, wfn = psi4.energy(DFA, return_wfn=True)
    
        Engine = psi4Engine(wfn, Report=0)
        Egs, _ = Engine.GetEnergy(f=fgs)

        Ets_old = 1e10
        for Single in (False, True):
            for step in range(20):
                Ets, Fts = Engine.GetEnergy(f=fts)

                Extra = [(2., 2., Engine.kh, Engine.kl, 'K')]
                Eex, Fex = Engine.GetExtra(Extra=Extra)
                Ess = Ets + Eex

                if Single:
                    Fts += Fex

                epsilon = Engine.epsilon
                C = Engine.C

                epsilon_, C_ = Engine.SolveFock(Fts, k0=Engine.kl)
                Engine.epsilon[Engine.kl:] = epsilon_
                Engine.C[:,Engine.kl:] = C_

                if (np.abs(Ets-Ets_old)<1e-6) and (step>5): break

                Ets_old = Ets
            print("Excitation en = %10.2f %10.2f"\
                  %(eV*(Ets - Egs), eV*(Ess - Egs)))

        psi4.geometry("0 3\n"+MolStr)
        psi4.set_options({
            'reference': 'rohf' if DFA=='scf' else 'uhf',
        })
        E1 = psi4.energy(DFA)

        print("%10.5f %10.5f %10.5f %10.5f"%(E0, Egs, eV*(E1-E0), eV*(Ets-Egs)))

    print(Engine.SymHelp.s_sorted)
    
