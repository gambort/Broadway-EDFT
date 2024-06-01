"""
Template for the Engine class
"""

class schemeEngine:
    def __init__(self):
        # Constant values
        self.nbf # Number of basis functions (nbf)
        self.kh # The HOMO index from the original calculation
        self.kl # The LUMO index from the original calculation

        self.NAtom # Number of atoms
        self.Enn # Nuclear-repulsion energy

        self.Has_RS # Whether or not range-separation is allowed

        # Dealing with symmetry
        self.Sym_k[k] # array of nbf values
        # Indicates symmetry index of each state
        # Note - self.Sym_k = np.zeros((self.nbf,))
        #        for implementations without symmetry

        # Orbitals and eigenvalues
        self.f_Occ # GS occupation factors (nbf)
        self.C # GS orbital coefficients in the basis (nbf x nbf)
        self.epsilon # GS orbital energies (nbf)

        # One-body reduced density matrices (1RDMs)
        self.Da # alpha 1RDM (nbf x nbf)
        self.Db # beta  1RDM (nbf x nbf)
        self.D = self.Da + self.Db # Their sum


        # Constant matrices
        self.S_ao # Overlaps in the basis (nbf x nbf)
        self.T_ao # KE in the basis (nbf x nbf)
        self.V_ao # PE in the basis (nbf x nbf)
        self.Dip_ao # Dipole in the basis 3 x (nbf x nbf)

        # In case someone tries to use this engine
        print("The schemeEngine is for development only and does nothing")
        print("quitting!")
        quit()

    # Methods
    # Fock matrices are in the same basis as S_ao, T_ao and V_ao

    # Compute the Hx energy and Fock matrix from Ca, Cb
    # Ca and Cb are (nbf x occ) array of occupied orbitals
    def GetHx(self, Ca, Cb=None, BothSpin=False):
        if Cb is None: Ca = Cb # default to both same

        EHx = 0. # Some function
        if not(Ca==Cb):
            FHx_a_ao = 0. # Some function of Ca and Cb
            FHx_b_ao = 0. # Some function of Ca and Cb
        else:
            FHx_a_ao = 0. # Some function of Ca
            FHx_b_ao = FHx_a_ao # Same as a

        if BothSpin:
            return EHx, FHx_a_ao, FHx_b_ao
        else:
            return EHx, FHx_a_ao

    # Compute the xc energy and Fock matrix from Da, Db
    # Da and Db are (nbf x nbf) matrices in ao basis
    def GetDFA(self, Da, Db=None, Pre=1., BothSpin=False):
        if Db is None: Da = Db # default to both same

        Exc = 0. # Some function
        if not(Da==Db):
            Fxc_a_ao = 0. # Some function of Da and Db
            Fxc_b_ao = 0. # Some function of Da and Db
        else:
            Fxc_a_ao = 0. # Some function of Da
            Fxc_b_ao = FHx_a_ao # Same as a

        if BothSpin:
            return Pre*Exc, Pre*Fxc_a_ao, Pre*Fxc_b_ao
        else:
            return Pre*EHx, Pre*FHx_a_ao
        
    # Handle the extra list
    def GetExtra(self, C=None, Extras):
        # C is coefficients
        # Extras = [ (WE, WF, j, k, Kind), ... ]
        # where:
        #    WE is the prefactor for energies/Fock
        #    j, k are indices for [jj|kk] or [jk|kj] - Fock derivative is on j
        #    Kind is J, K or wK
        if C is None: C = self.C # default to internal coefficients

        def GetERI(Cj, Ck, Kind):
            if Kind=='J':
                FJ = 0. # [jj|pq]
            elif Kind=='K':
                FK = 0. # [jp|qj]
            elif Kind=='wK':
                FK = 0. # [jp|qk]_rs
            return (Ck).dot(FK).dot(Ck), FK
        
        EEx_P = 0.
        FEx_P = 0.
        for WE, WF, j, k, Kind in Extras:
            E, F = GetERI(C[:,j], C[:,k], Kind)
            EEx_P += WE * E
            if np.abs(WF)>0.:
                FEx_P += WF * F
        
        return EEx_P, FEx_P

    
    # Extra stuff needed by specialised routines

    # Solve the Fock equations orthogonal to all orbitals >=k0
    def SolveFock(self, F_ao, k0 = kl):
        return epsE, CE # eigenvalues and coefficients

    # Get a DFT grid for O(N) scaling - for ELDA only
    def GetDFTGrid(self):
        # see psi4Engine for details of contents
        return Grid
