# Standard psi4 run
import psi4

# Use the psi4 engine
from psi4Engine.Engine import *

##############################################################################
# This code demonstrates basic use of Broadway
#
# Note, by using psi4 as an engine we preserve symmetries of the
# system by default
##############################################################################

# We will use eV as units
eV = 27.211 # Ha X eV = En in eV

# Use the full orbital optimized EDFT code 
from Broadway.OOEDFT import *

# Set up and run a standard psi4 job
psi4.set_output_file("_Output_psi4.out") # Save the psi4 outputs
psi4.set_options({
    'basis': 'cc-pvtz',
    'reference': 'rks',
})

if True:
    # Set up BH molecule, which has degenerate LUMO
    psi4.geometry("0 1\nB\nH 1 1.234")
else:
    # Set up H2 molecule, with non-degen LUMO
    psi4.geometry("0 1\nH\nH 1 0.75")


# Run the job and extract the wavefunction from psi4
E0_psi4, wfn = psi4.energy("wb97x", return_wfn=True)

# Turn the wavefunction into an engine
Engine = psi4Engine(wfn)

# Pass the engine to Broadway
# - note, it will report excitations using the ensemblized version of
#   the psi4 DFA
# - note, Report=3 gives moderately verbose outputs
XHelp = OOExcitationHelper(Engine, Report=3)

# Specify the orbitals to promote
XHelp.SetFrom('HOMO')
XHelp.SetTo('LUMO')

# Solve the GS and check difference from psi4
E0 = XHelp.SolveGS()
BadGS = np.abs(E0-E0_psi4)>1e-4 # Identify a bad GS if deviation > 0.1 mHa
# If the GS is bad let us know as indicates problems
if BadGS:
    print("Excessive deviation in GS from psi4 = %.3f eV"%( eV*(E0 - E0_psi4) ))

# Solve the triplet state
ET = XHelp.SolveTS()


# Solve the single-excited singlet
# Note - this will start from the triplet orbitals (last run)
#        and in difficult cases can become trapped in a local minima.
#        Orbitals can be reset in such cases
#XHelp.ResetOrbitals() # Reset to the initial GS orbitals
#XHelp.ResetNaturalOrbitals() # Reset to the natural orbitials
#                              e.g. if you start from a triplet in psi4
ES = XHelp.SolveSX()

# Solve the double-excited singlet
# Note - because BH has a degenerate LUMO it will run in
#        split mode to LUMO & LUMO+1;
#        for H2 it goes to LUMO^2
# Note - in rare cases (e.g. LiH) this will 'collapse' to
#        the ground state, with a small correction from
#        the Hartree energy
#        Always check for ED<ES and ED\approx E0 after a
#        long cycle.
ED = XHelp.SolveDX()

# Output energies and excitation energies
for EID, ID in zip((E0, ET, ES, ED), ('0', 'T', 'S', 'D')):
    print("Gap(%s) = %6.2f eV, En = %10.4f Ha"%( ID, eV*(EID - E0), EID ))

