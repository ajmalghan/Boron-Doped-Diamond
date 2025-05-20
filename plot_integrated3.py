#------------------------------------------------------------------------
#-------------Written by Dr.Ajmalghan, Gleam Innovations, Banglore-------
#------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.integrate import trapz

# ----------------------------
# Configuration & Constants
# ----------------------------

# Create results directory
results_dir = 'pdos_analysis_results'
os.makedirs(results_dir, exist_ok=True)

# Publication-quality plot style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'font.weight': 'bold',
    'axes.linewidth': 1.5,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'legend.framealpha': 0.8,
    'legend.fontsize': 10,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

# Band edges and Fermi levels (in eV)
pure_VBM = 14.001
pure_CBM = 16.741
bdd_VBM = 14.354
bdd_CBM = 16.764

fermi_pure = pure_CBM          # Pure diamond: Fermi at CBM
fermi_bdd = bdd_VBM + 0.05    # BDD: slightly above VBM (degenerate p-type)

# ----------------------------
# Utility Functions
# ----------------------------

def read_dos_file(filename):
    """
    Reads a two-column DOS file: energy (eV), DOS (states/eV)
    Returns energy and DOS arrays.
    """
    try:
        data = np.loadtxt(filename)
        energy = data[:, 0]
        dos = data[:, 1]
        return energy, dos
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return np.array([]), np.array([])

def interpolate_dos(energy_source, dos_source, energy_target):
    """
    Interpolates DOS data to a target energy grid.
    """
    interp_func = interp1d(energy_source, dos_source, kind='cubic', bounds_error=False, fill_value=0)
    return interp_func(energy_target)

# ----------------------------
# Read Data
# ----------------------------

print("Reading Boron-Doped Diamond (BDD) data...")
bdd_c_total_e, bdd_c_total_dos = read_dos_file('C_total_pdos.dat')
bdd_b_total_e, bdd_b_total_dos = read_dos_file('B_total_pdos.dat')
bdd_c_p_e, bdd_c_p_dos = read_dos_file('C_p_pdos.dat')
bdd_b_p_e, bdd_b_p_dos = read_dos_file('B_p_pdos.dat')
bdd_c_s_e, bdd_c_s_dos = read_dos_file('C_s_pdos.dat')
bdd_b_s_e, bdd_b_s_dos = read_dos_file('B_s_pdos.dat')

print("Reading Pure Diamond data...")
pure_dir = 'pure'
pure_total_e, pure_total_dos = read_dos_file(os.path.join(pure_dir, 'C_total_pdos.dat'))
pure_p_e, pure_p_dos = read_dos_file(os.path.join(pure_dir, 'C_p_pdos.dat'))
pure_s_e, pure_s_dos = read_dos_file(os.path.join(pure_dir, 'C_s_pdos.dat'))

# ----------------------------
# Combine BDD total DOS (C + B)
# ----------------------------

# Check if energy grids match for total DOS
if (len(bdd_c_total_e) == len(bdd_b_total_e)) and np.allclose(bdd_c_total_e, bdd_b_total_e):
    bdd_total_e = bdd_c_total_e
    bdd_total_dos = bdd_c_total_dos + bdd_b_total_dos
else:
    print("Energy grids differ for C and B total DOS in BDD; interpolating to common grid...")
    min_e = max(bdd_c_total_e.min(), bdd_b_total_e.min())
    max_e = min(bdd_c_total_e.max(), bdd_b_total_e.max())
    bdd_total_e = np.linspace(min_e, max_e, 1000)
    c_interp = interp1d(bdd_c_total_e, bdd_c_total_dos, kind='cubic', bounds_error=False, fill_value=0)
    b_interp = interp1d(bdd_b_total_e, bdd_b_total_dos, kind='cubic', bounds_error=False, fill_value=0)
    bdd_total_dos = c_interp(bdd_total_e) + b_interp(bdd_total_e)

# ----------------------------
# Plot 1: Total DOS Comparison
# ----------------------------

plt.figure(figsize=(10, 7))
plt.plot(pure_total_e, pure_total_dos, 'b-', lw=2, label='Pure Diamond')
plt.plot(bdd_total_e, bdd_total_dos, 'r-', lw=2, label='Boron-Doped Diamond')
plt.axvline(x=fermi_pure, color='blue', ls='--', lw=1.5, label='Fermi Level (Pure)')
plt.axvline(x=fermi_bdd, color='red', ls='--', lw=1.5, label='Fermi Level (BDD)')
plt.xlabel('Energy (eV)')
plt.ylabel('Density of States (states/eV)')
plt.title('Total DOS Comparison: Pure vs. Boron-Doped Diamond')
plt.legend(frameon=True, fancybox=True)
plt.grid(alpha=0.3, ls='--')
plt.xlim(-9, 32)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '1_total_dos_comparison.png'))
plt.close()

print("Saved: 1_total_dos_comparison.png")

# ----------------------------
# Plot 2: Projected DOS Comparison
# ----------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Pure Diamond PDOS
ax1.plot(pure_total_e, pure_total_dos, 'k-', lw=2, label='Total')
ax1.plot(pure_p_e, pure_p_dos, 'b-', lw=2, label='p-states')
ax1.plot(pure_s_e, pure_s_dos, 'g-', lw=2, label='s-states')
ax1.axvline(x=fermi_pure, color='blue', ls='--', lw=1.5, label='Fermi Level (Pure)')
ax1.set_ylabel('DOS (states/eV)')
ax1.set_title('Pure Diamond: Projected DOS')
ax1.legend(frameon=True, fancybox=True)
ax1.grid(alpha=0.3, ls='--')

# BDD PDOS
ax2.plot(bdd_total_e, bdd_total_dos, 'k-', lw=2, label='Total')
ax2.plot(bdd_c_p_e, bdd_c_p_dos, 'b-', lw=2, label='C p-states')
ax2.plot(bdd_c_s_e, bdd_c_s_dos, 'g-', lw=2, label='C s-states')
ax2.plot(bdd_b_p_e, bdd_b_p_dos, 'r-', lw=2, label='B p-states')
ax2.plot(bdd_b_s_e, bdd_b_s_dos, color='orange', lw=2, label='B s-states')
ax2.axvline(x=fermi_bdd, color='red', ls='--', lw=1.5, label='Fermi Level (BDD)')
ax2.set_xlabel('Energy (eV)')
ax2.set_ylabel('DOS (states/eV)')
ax2.set_title('Boron-Doped Diamond: Projected DOS')
ax2.legend(frameon=True, fancybox=True)
ax2.grid(alpha=0.3, ls='--')

plt.xlim(-9, 32)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '2_projected_dos_comparison.png'))
plt.close()

print("Saved: 2_projected_dos_comparison.png")

# ----------------------------
# Plot 3: DOS Difference (BDD - Pure)
# ----------------------------

# Interpolate pure total DOS to BDD energy grid for difference
pure_total_interp = interpolate_dos(pure_total_e, pure_total_dos, bdd_total_e)
dos_diff = bdd_total_dos - pure_total_interp

plt.figure(figsize=(10, 7))
plt.plot(bdd_total_e, dos_diff, 'm-', lw=2, label='BDD - Pure DOS')
plt.axhline(0, color='black', lw=1)
plt.axvline(x=fermi_pure, color='blue', ls='--', lw=1.5, label='Fermi Level (Pure)')
plt.axvline(x=fermi_bdd, color='red', ls='--', lw=1.5, label='Fermi Level (BDD)')
plt.xlabel('Energy (eV)')
plt.ylabel('DOS Difference (states/eV)')
plt.title('DOS Difference: Boron-Doped Diamond minus Pure Diamond')
plt.legend(frameon=True, fancybox=True)
plt.grid(alpha=0.3, ls='--')
plt.xlim(-9, 32)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '3_dos_difference.png'))
plt.close()

print("Saved: 3_dos_difference.png")

# ----------------------------
# Plot 4: Integrated DOS (Cumulative states)
# ----------------------------

# Integrate DOS from lowest energy up to each energy point
pure_cumulative = np.cumsum(pure_total_dos) * np.gradient(pure_total_e)
bdd_cumulative = np.cumsum(bdd_total_dos) * np.gradient(bdd_total_e)

plt.figure(figsize=(10, 7))
plt.plot(pure_total_e, pure_cumulative, 'b-', lw=2, label='Pure Diamond')
plt.plot(bdd_total_e, bdd_cumulative, 'r-', lw=2, label='BDD')
plt.axvline(x=fermi_pure, color='blue', ls='--', lw=1.5, label='Fermi Level (Pure)')
plt.axvline(x=fermi_bdd, color='red', ls='--', lw=1.5, label='Fermi Level (BDD)')
plt.xlabel('Energy (eV)')
plt.ylabel('Integrated DOS (states)')
plt.title('Integrated DOS (Cumulative States)')
plt.legend(frameon=True, fancybox=True)
plt.grid(alpha=0.3, ls='--')
plt.xlim(-9, 32)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '4_integrated_dos.png'))
plt.close()

print("Saved: 4_integrated_dos.png")

# ----------------------------
# Plot 5: Charge Transfer Estimate (Integrated DOS difference)
# ----------------------------

# Interpolate pure cumulative DOS to BDD grid for difference
pure_cumulative_interp = interpolate_dos(pure_total_e, pure_cumulative, bdd_total_e)
cumulative_diff = bdd_cumulative - pure_cumulative_interp

plt.figure(figsize=(10, 7))
plt.plot(bdd_total_e, cumulative_diff, 'purple', lw=2, label='Cumulative DOS Difference')
plt.axhline(0, color='black', lw=1)
plt.axvline(x=fermi_pure, color='blue', ls='--', lw=1.5, label='Fermi Level (Pure)')
plt.axvline(x=fermi_bdd, color='red', ls='--', lw=1.5, label='Fermi Level (BDD)')
plt.xlabel('Energy (eV)')
plt.ylabel('Integrated DOS Difference (states)')
plt.title('Charge Transfer Estimate: BDD minus Pure')
plt.legend(frameon=True, fancybox=True)
plt.grid(alpha=0.3, ls='--')
plt.xlim(-9, 32)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '5_charge_transfer_estimate.png'))
plt.close()

print("Saved: 5_charge_transfer_estimate.png")

# ----------------------------
# Plot 6: Orbital Contributions (s and p states) for BDD
# ----------------------------

plt.figure(figsize=(10, 7))
plt.plot(bdd_c_p_e, bdd_c_p_dos, 'b-', lw=2, label='C p-states')
plt.plot(bdd_c_s_e, bdd_c_s_dos, 'g-', lw=2, label='C s-states')
plt.plot(bdd_b_p_e, bdd_b_p_dos, 'r-', lw=2, label='B p-states')
plt.plot(bdd_b_s_e, bdd_b_s_dos, color='orange', lw=2, label='B s-states')
plt.axvline(x=fermi_bdd, color='red', ls='--', lw=1.5, label='Fermi Level (BDD)')
plt.xlabel('Energy (eV)')
plt.ylabel('DOS (states/eV)')
plt.title('BDD Orbital Contributions (s and p states)')
plt.legend(frameon=True, fancybox=True)
plt.grid(alpha=0.3, ls='--')
plt.xlim(-9, 32)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '6_bdd_orbital_contributions.png'))
plt.close()

print("Saved: 6_bdd_orbital_contributions.png")

# ----------------------------
# Done
# ----------------------------

print(f"All analysis complete. Plots saved in folder: '{results_dir}'")
