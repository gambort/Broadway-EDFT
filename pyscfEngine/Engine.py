from pyscf import gto, scf, dft, df
import pyscf.tools
import numpy as np
import scipy.linalg as la

import numpy.random as np_ra


eV = 27.211

zero_round = 1e-5

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")


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



#################################################################################################
# This is the main engine
#################################################################################################

class pyscfEngine:
    def __init__(self, mol, SCF, dm_Ref = None,
                 # Pass in the molecule and SCF and maybe a reference dm
                 alpha = None, beta = None, omega = None,
                 auxbasis = 'def2-qzvp-jkfit',
                 Report = 0):
        
        if mol.symmetry:
            print("Symmetry is not implemented - quitting!")
            quit()

        self.Report = Report
        self.mol = mol
        self.SCF = SCF


        if dm_Ref is None:
            dm_Ref = SCF.make_rdm1()
            
        if len(dm_Ref.shape)==3:
            self.Da = dm_Ref[0,:,:]
            self.Db = dm_Ref[1,:,:]
        else:
            self.Da = dm_Ref/2.
            self.Db = dm_Ref/2.
            
        self.D = self.Da + self.Db
        self.epsilon = SCF.mo_energy
        self.C = SCF.mo_coeff
        self.f = SCF.mo_occ
                
        self.nbf = len(self.epsilon)
        self.NAtom = len(mol._atom)
       

        self.Enn = SCF.energy_nuc()

        # Compute the one-body integrals
        self.S_ao = mol.intor('int1e_ovlp')
        self.T_ao = mol.intor('int1e_kin')
        self.V_ao = mol.intor('int1e_nuc')

        self._SetDFA(SCF)

        self.OverwriteHybrid(alpha=alpha, beta=beta, omega=omega)
        
        self.NBas = self.nbf
        self.NOcc = int(np.sum(self.f)//2) # Always round down to double occ
        self.kh = self.NOcc-1 # HOMO
        self.kl = self.kh+1 # LUMO

        # Symmetry group by basis function        
        self.Sym_k = np.array([0]*self.NBas)

        # Default occupation factors
        self.f_Occ = 2.*np.ones((self.NOcc,))

    def __del__(self):
        #print("Closing psi4Engine")
        1
        #if not(self.ERIA is None): del self.ERIA
        #if not(self.ERIA_w is None): del self.ERIA_w


    def _SetDFA(self, SCF):
        self.DFA_numint = None
        # These are used for HF and DFA calculations
        self.alpha = 1.
        self.beta = 0.
        self.omega = None
        self.Has_RS = not(self.omega is None)

        try:
            self.xc = SCF.xc
        except:
            return

        self.DFA_numint = SCF._numint
        omega, A, B = self.DFA_numint.rsh_and_hybrid_coeff(self.xc)
        
        self.omega = omega
        self.alpha = B
        self.beta = A - B

        if self.omega<zero_round: self.omega = None
        
        self.Has_RS = not(self.omega is None)
        

    def OverwriteHybrid(self, alpha=None, beta=None, omega=None):
        if not(alpha is None): self.alpha = alpha
        if not(beta is None): self.beta = beta
        if not(omega is None): self.omega = omega
        self.Has_RS = not(self.omega is None)

    def FMaster(self, C1, C2, Kind):
        D = 0.5 * (np.outer(C1, C2) + np.outer(C2, C1))
        if Kind in ('j', 'J'):
            FJ, FK = self.SCF.get_jk(self.mol, D, hermi=1,
                                     with_j = True, with_k = False)
            return FJ
        elif Kind in ('k', 'K'):
            FJ, FK = self.SCF.get_jk(self.mol, D, hermi=1,
                                     with_j = False, with_k = True)
            return FK
        elif Kind.upper() == "K_W":
            FJ, FK = self.SCF.get_jk(self.mol, D, hermi=1,
                                     with_j = False, with_k = True,
                                     omega = self.omega)
            return FK

        return 0.
        
    def GetFJ(self, CI, Pre=1.):
        if np.abs(Pre)<zero_round: return 0.

        D = CI.dot(CI.T)
        FJ, FK = self.SCF.get_jk(self.mol, D, hermi=1,
                                 with_j = True, with_k = False)
        return Pre*FJ

    def GetFK(self, CI, Pre=1.):
        if np.abs(Pre)<zero_round: return 0.

        D = CI.dot(CI.T)
        FJ, FK = self.SCF.get_jk(self.mol, D, hermi=1,
                                 with_j = False, with_k = True)
        return Pre*FK

    def GetFK_w(self, CI, Pre=1.):
        if np.abs(Pre)<zero_round: return 0.
        if self.omega is None or (self.omega==0.): return 0.
        
        D = CI.dot(CI.T)
        FJ, FK = self.SCF.get_jk(self.mol, D, hermi=1,
                                 with_j = False, with_k = True,
                                 omega = self.omega)
        return Pre*FK

    # Compute the DFA terms and return energy and V matrix
    def GetDFA(self, Da=None, Db=None, Pre=1.,
               BothSpin=False):
        if self.DFA_numint is None:
            if BothSpin: return 0., 0., 0.
            else: return 0., 0.
        
        # Use internal Da by default
        if Da is None: Da = self.Da
        if Db is None: Db = Da

        if np.abs(Pre)<zero_round:
            if BothSpin: return 0., 0., 0.
            else: return 0., 0.

        dm = np.zeros((2,self.NBas,self.NBas))
        dm[0,:,:] = Da
        dm[1,:,:] = Db
        n, exc, vxc = self.DFA_numint.nr_uks(self.mol,
                                      self.SCF.grids, self.SCF.xc,
                                      dm)

        ExcDFA = Pre*exc
        VxcDFA = Pre*vxc[0,:,:]

        if BothSpin:
            return ExcDFA, VxcDFA, Pre * vxc[1,:,:]

        return ExcDFA, VxcDFA

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

            if np.abs(alphaH) + np.abs(alpha)>zero_round:
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
            F11 = self.FMaster(C1, C1, Kind)
                
            EExtra += PreE * (C2).dot(F11).dot(C2)
            FExtra += PreF * F11

        return EExtra, FExtra

    def Solve(self, F, **kwargs):
        return la.eigh(F, b=self.S_ao)
    def SolveFock(self, F, k0=0, **kwargs):
        if k0<=0:
            return la.eigh(F, b=self.S_ao)
        else:
            F_C = np.einsum('pq,pk,qj->kj', F, self.C[:,k0:], self.C[:,k0:])
            w, U = la.eigh(F_C)
            return w, self.C[:,k0:].dot(U)

    ###################################################################################
    # Routines for custom DFAs
    ###################################################################################

    def GetDFTGrid(self, GGA=False, MGGA=False, delta=0.):
        try:
            pg = self.SCF.grid
        except:
            pg = dft.gen_grid.Grids(self.mol)
            pg.level = 4 # Not dense because LDA
            pg.build()
            
        xyz = pg.coords
        w = pg.weights
        phi = dft.numint.eval_ao(self.mol, xyz, deriv=0)
        lpos = np.arange(phi.shape[1])

        if GGA or MGGA:
            print("GGA and MGGA not implented yet!")
            quit()

        Grid = [ {'w':w, 'lphi':phi, 'lpos':lpos} ]

        # psi4 implementation
        # Grid = [None]*self.VPot.nblocks()

        # basis = self.wfn.basisset()
        
        # for b in range(self.VPot.nblocks()):
        #     block = self.VPot.get_block(b)
        #     NP = block.npoints()

        #     Grid[b] = {'NP':NP, 'w': block.w().to_array() }

        #     blockopoints = psi4.core.BlockOPoints\
        #         ( block.x(), block.y(), block.z(), block.w(),
        #           psi4.core.BasisExtents(basis,delta) )

        #     lpos = np.array(blockopoints.functions_local_to_global())

        #     funcs = psi4.core.BasisFunctions(basis, NP, basis.nbf())
        #     funcs.compute_functions(blockopoints)
        #     lphi = funcs.basis_values()["PHI"].to_array(dense=True)

        #     Grid[b]['lpos'] = lpos
        #     Grid[b]['lphi'] = lphi[:, lpos]

        return Grid


    def GetGGAProps(self, Da, Db):
        xyz, w, (rhoa,rhob), _ = GetDensities(None, D1List=(Da,Db), wfn=self.wfn,
                                              return_w = True, return_xyz = True)
        rho = rhoa + rhob
        rho_m = np.maximum(rho, 1e-18)
        zeta = np.abs(rhoa-rhob)/rho_m
        
        rs, s, _ = GetGGAProps(self.wfn, Da+Db)

        return {'xyz':xyz, 'w':w, 'rho':rho, 'rs':rs, 's':s, 'zeta':zeta}


    def GetELDAProps(self, f=None, C=None):
        if self.DFA_numint is None: return None
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

    def Gradient_NN(self):
        return self.basis.molecule().nuclear_repulsion_energy_deriv1().to_array(dense=True)
    
    def GetEGGAProps(self, f=None, C=None):
        if self.DFA_numint is None: return None
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
            
if __name__ == "__main__":
    Mol = gto.M(atom = 'O 0 0 0; H 0 1 0; H 0 0 1', basis = 'cc-pvdz')
    #Mol = gto.M(atom = 'He', basis = 'cc-pvdz')

    #SCF = scf.RHF(Mol)
    SCF = scf.RKS(Mol,'wb97x')
    E0 = SCF.kernel()

    Engine = pyscfEngine(Mol, SCF)

    print(E0)
    Ts = np.vdot(Engine.D, Engine.T_ao)
    EV = np.vdot(Engine.D, Engine.V_ao)
    EH, _ = Engine.GetHx(alpha=0.)
    Ex, _ = Engine.GetHx(alphaH=0.)

    print(Ts, EV, EH, Ex)
    
    
    E  = Engine.Enn + Ts + EV + EH + Ex
    print(E)
    print(E-E0)
    
    E, _ = Engine.GetEnergy()
    print(E)
    print(E-E0)

