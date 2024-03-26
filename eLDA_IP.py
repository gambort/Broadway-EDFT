import psi4
from psi4Engine.Engine import psi4Engine, TextDFA
from Broadway.OOELDA import *
from Broadway.OOEDFT import *
from Broadway.Helpers import *
        
##############################################################################
# This code demonstrates a number of advanced features of
# Broadway, include the specialized ELDA solver
##############################################################################

psi4.set_output_file("_LDA_IP.out")


Basis = 'aug-cc-pvqz'

psi4.set_options({
    'basis': Basis,
    'reference': 'uhf',
    'fail_on_maxiter': False,
})


Atoms = [
    '', 'H', 'He',
    'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar',
]

# Use a cache to avoid repeating calculations
try:
    X = np.load("./Cache/IP_LDA.npz", allow_pickle=True)
    AtomData = X['AtomData'][()]
except:
    AtomData = { }

# Check if the basis has been cached
if not(Basis in AtomData):
    AtomData[Basis] = {}

# Convert the number of electrons to the spin Q
def NtoQ(N): 
    for R in [(0,4), (4,10), (10,12), (12,18), (18,20), (20,30), (30,36)]:
        if N>R[0] and N<=R[1]:
            if (R[1]-R[0])==6:
                return 1 + min((N-R[0]), 6-(N-R[0]))
            
    return 1 + N%2

# Convert the number of electrons and Q to occupations and HOMO
def Ntof(N, Q):
    f = np.zeros((20,))
    Nu = (N + Q - 1)//2
    Nd = (N - Q + 1)//2

    f[:Nu] += 1.
    f[:Nd] += 1.
    HOMO = Nu-1

    f = f[:Nu]
    
    return f, HOMO

# Iterate over the atoms
for Z, Atom in enumerate(Atoms):
    # Skip H as not interesting (all methods same)
    if Z<2: continue

    # Skip cached values
    if Atom in AtomData[Basis]: continue

    # Show the atom
    print('='*72)
    print("%2s Z = %d, Q = %d"%(Atom, Z, NtoQ(Z)))

    Q  = NtoQ(Z  )
    Qp = NtoQ(Z-1)

    # Get the neutral atom energy and Nalpha
    psi4.geometry("""%d %d\n%s\nsymmetry c1"""%(0, Q , Atom))
    E0 = psi4.energy('svwn')
    Nf = wfn.nalpha()

    # Initialise the cation and its wavefunction
    psi4.geometry("""%d %d\n%s\nsymmetry c1"""%(1, Qp, Atom))
    Ep, wfn = psi4.energy('svwn', return_wfn=True)

    # Calculate the cation using svwn LSDA theory
    IP_svwn = Ep - E0
    print("IP(SVWN)   = %7.2f eV"%(IP_svwn*eV))

    # Pass the cation wavefunction to psi4Engine
    Engine = psi4Engine(wfn, Report=-1)

    # Determine the occupancies and HOMO for netural and cation
    f , HOMO  = Ntof(Z  , Q )
    fp, HOMOp = Ntof(Z-1, Qp)

    # Cut f down to relevant orbitals
    f  = f[:Nf]
    fp = fp[:Nf]

    # Create custom plans for ground state and cation ground state
    #   Note, these are pure states so only have a single Hx and xcDFA
    #   term. Excited states may have multiple 
    Plan  = { '1RDM': f , 'kTo': (HOMO ,), 'Singlet': False, 'Extra': None,
              'Hx':[(1.,1.,f ),],  'xcDFA':[(1.,1.,f ),] }
    Planp = { '1RDM': fp, 'kTo': (HOMOp,), 'Singlet': False, 'Extra': None,
              'Hx':[(1.,1.,fp),],  'xcDFA':[(1.,1.,fp),] }

    # Use the specialised ELDA energy routine which implements a custom DFA
    XHelp = OOELDAExcitationHelper(Engine, Report=-1)
    XHelp.Setxi(0.) # Not necessary but best to be safe

    # This forces the default dwocc fbar model
    XHelp.SeteLDAType('NO', a=1/3)
    Ep = XHelp.Solver(Planp) # Solve neutral
    E0 = XHelp.Solver(Plan ) # Solve cation

    IP_dwocc = Ep - E0
    print("IP(dwocc)  = %7.2f eV"%(IP_dwocc*eV))

    
    # This forces the simple but bad wocc fbar model
    XHelp.SeteLDAType('NO', a=1  )
    Ep = XHelp.Solver(Planp)
    E0 = XHelp.Solver(Plan )

    IP_wocc = Ep - E0
    print("IP(wocc)   = %7.2f eV"%(IP_wocc*eV))

    AtomData[Basis][Atom] = [ IP_svwn*eV, IP_dwocc*eV, IP_wocc*eV ]
    
    np.savez("./Cache/IP_LDA.npz", AtomData=AtomData)

# Print a summary of all results
print("="*72)
print("# N El %7s %7s %7s"%('SVWN', 'dwocc', 'wocc'))
for Z, Atom in enumerate(Atoms):
    if Atom in AtomData[Basis]:
        print("%3d %2s "%(Z, Atom)
              + "%7.2f %7.2f %7.2f"%tuple(AtomData[Basis][Atom])) 
      
