import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Parameters
E_max = 6.83
pos_delta = True
n_protons = 47   # number of protons to fill
n_neutrons = 61  # number of neutrons to fill
min_spacing = 0.02  # minimum vertical spacing between labels
label_size = 13

# Load data
df_neutrons = pd.read_csv("164Tb_neutrons2.dat", delim_whitespace=True, skiprows=1)
df_protons  = pd.read_csv("164Tb_protons2.dat",  delim_whitespace=True, skiprows=1)

# Clean up first column
df_neutrons.rename(columns={df_neutrons.columns[0]: "delta"}, inplace=True)
df_protons.rename(columns={df_protons.columns[0]: "delta"}, inplace=True)

# Filter only positive delta
if pos_delta:
    df_neutrons = df_neutrons[df_neutrons["delta"] >= 0]
    df_protons  = df_protons[df_protons["delta"] >= 0]

# Pick a delta value to evaluate levels
delta_target = 0.44
idx_neu = (np.abs(df_neutrons["delta"] - delta_target)).argmin()
idx_pro = (np.abs(df_protons["delta"] - delta_target)).argmin()

# Extract energies at that delta
levels_neu = [(col, df_neutrons[col].iloc[idx_neu]) for col in df_neutrons.columns[1:]]
levels_pro = [(col, df_protons[col].iloc[idx_pro]) for col in df_protons.columns[1:]]

# Sort by energy
levels_neu.sort(key=lambda x: x[1])
levels_pro.sort(key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(8, 8))

def draw_levels(ax, levels, x_start, x_end, x_dot, color_dot, n_fill, E_max):
    """Draw horizontal levels, labels, and dots with small nudge for overlapping labels."""
    filled = 0
    last_y = -np.inf
    for col, E in levels:
        if E > E_max:
            continue

        # Nudge the level if too close to the previous
        if E - last_y < min_spacing:
            E += min_spacing - (E - last_y)
        last_y = E

        ax.hlines(E, x_start, x_end, color="black", linewidth=1, clip_on=True)
        # Label
        ha = "right" if x_dot < 0.5 else "left"
        lx = x_start - 0.02 if ha=="right" else x_end + 0.02
        ax.text(lx, E, col, va="center", ha=ha, fontsize=10, color="black", clip_on=True)

        # Fill dots
        if filled < n_fill:
            n_draw = min(2, n_fill - filled)
            for i in range(n_draw):
                ax.plot(x_dot + (i-0.5)*0.04, E, 'o', color=color_dot, clip_on=True)
            filled += n_draw

# Draw proton levels (left)
draw_levels(ax, levels_pro, 0.35, 0.45, 0.40, "red", n_protons, E_max)

# Draw neutron levels (right)
draw_levels(ax, levels_neu, 0.55, 0.65, 0.60, "blue", n_neutrons, E_max)

ax.set_xlim(0.2, 0.8)
ax.set_ylim(5.24, E_max)
ax.tick_params(axis='y',labelsize=label_size)
ax.set_ylabel("Single-particle energy ($\hbar\omega$)", fontsize=label_size)
ax.set_xticks([0.40, 0.60])
ax.set_xticklabels(["Protons", "Neutrons"], fontsize=label_size)
# ax.set_title("Single-particle levels in $^{164}$Tb at $\delta =$ ", fontsize=label_size)
ax.set_title(f"Single-particle levels in $^{{164}}$Tb at $\\delta = {delta_target}$", fontsize=label_size)

# Remove box around plot
for spine in ax.spines.values():
    spine.set_visible(False)

# plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig("164Tb_levels.pdf", bbox_inches='tight', dpi=300)
plt.tight_layout()
plt.show()
