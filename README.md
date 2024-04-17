# Broadway-EDFT
Broadway code for implementing ensemble DFT and mathematically similar theories

Note, you need a working copy of [psi4](https://psicode.org/) or [pyscf](https://pyscf.org/) to use Broadway.
The psi4 interface is recommended, more fully featured, and used for examples.

## Installation

Broadway has not been designed as a package or library, as it is intended to be used for both demonstration
and development purposes. To install, simply download the code and sub-directories to the directory you
intend to call it from. Alternatively, use links if you wish changes to be shared in multiple projects.

## Basic use

The key steps for using Broadway are:
1) Run an initial psi4 (or pyscf) calculation and return the wavefunction (or scf) object
2) Pass the wavefunction (or scf) object to the correct Engine
3) Create an EDFT class from the Engine, and use it to compute things.

## Worked example

1) Run an initial psi4 calculation and return the wavefunction

```python
import psi4

psi4.set_options({
   "reference": "rhf", # Start from RHF
   "basis": "cc-pvdz", # Set the basis
   })

psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 103") # Set up a water

# Run a PBE0 calculation of water and return the wavefunction as wfn
E_psi4, wfn = psi4.energy("pbe0", return_wfn=True)
```

2) Pass the wavefunction to the Engine

```python
from psi4Engine.Engine import psi4Engine
Engine = psi4Engine(wfn)
```

3) Create an EDFT class from the Engine, and use it to compute things

```python
# We will do full orbital optimized EDFT using the OOEDFT class
from Broadway.OOEDFT import *

# Set up the excitation helper
XHelp = OOExcitationHelper(Engine, Report=3)

# Get the doubly occupied "ground state" and test against psi4
E_GS = XHelp.SolveGS()
if np.abs(E_GS-E_psi4)>1e-4:
   print("Energy is not the same as psi4. If you ran a doubly occupied"
         + " this indicates a problem.")
   quit()

# Calculate the singly-promoted singlet with h->l
E_SX = XHelp.SolveSX()

# Calculate the doubly-promoted singlet with h^2->l^2
E_DX = XHelp.SolveDX()

print('='*72)
print("S0->S1 energy : %6.2f eV"%(27.2*(E_SX-E_GS)))
print("S0->S2 energy : %6.2f eV"%(27.2*(E_DX-E_GS)))
print("S1->S2 energy : %6.2f eV"%(27.2*(E_DX-E_SX)))
```

This should yield an output whose last four lines are:
```
========================================================================
S0->S1 energy :   8.16 eV
S0->S2 energy :  28.35 eV
S1->S2 energy :  20.18 eV
```

## Caution when doing double excitations

Double excitations are solved directly via orbital optimization.
In normal use cases where the excitation is symmetry protected or is otherwise
low energy and stable, the double excitation code will converge smoothly to a
variational minima.

However, in rare cases (usually high energy double excitations)
the orbital optimization will instead settle to the `ground state
orbitals, by swapping l<->h.
There are two signatures of this behaviour:
i) the energy of the double excitation will be lower than that of the
corresponding single exciation with the same orbitals;
ii) the algorithm will iterate for a larger-than-usual (typically over 50)
number of steps.
In such cases the energy should be discarded.
