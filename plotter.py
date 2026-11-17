import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use serif font (Times-like)
matplotlib.rcParams['font.family'] = 'serif'

# Maximum energy at left edge
E_max = 8.0
E_max_pro = 8.0
pos_delta = True
neut_pro = True

if neut_pro:
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

    delta_neu = df_neutrons["delta"].to_numpy()
    delta_pro = df_protons["delta"].to_numpy()

    # Create side-by-side subplots
    fig, (ax_pro, ax_neu) = plt.subplots(1, 2, figsize=(20, 8))

    # ---- Proton plot ----
    for column in df_protons.columns[1:]:
        y = df_protons[column].to_numpy()
        if y[0] > E_max_pro:
            continue

        if "-" in column:
            style, color = "--", "red"
        else:
            style, color = "-", "black"

        ax_pro.plot(delta_pro, y, linestyle=style, color=color)

        idx = (np.abs(delta_pro - 0.4)).argmin()
        if 0 <= delta_pro[idx] <= delta_pro.max():
            ax_pro.text(delta_pro[idx], y[idx], column, fontsize=16, color=color, clip_on=True)

    ax_pro.set_title("Proton Energy Levels in $^{164}$Tb", fontsize=16)
    ax_pro.set_xlabel(r'$\delta$', fontsize=16)
    ax_pro.set_ylabel(r'Single-particle energy ($\hbar \omega$)', fontsize=16)
    ax_pro.tick_params(axis='x', labelsize=16) 
    ax_pro.tick_params(axis='y', labelsize=16)
    ax_pro.set_xlim(0, 0.6)
    ax_pro.set_ylim(5.0, 7.0)
    ax_pro.grid(axis="x", linestyle="--", alpha=0.7)

    # ---- Neutron plot ----
    for column in df_neutrons.columns[1:]:
        y = df_neutrons[column].to_numpy()
        if y[0] > E_max:
            continue

        if "-" in column:
            style, color = "--", "blue"
        else:
            style, color = "-", "black"

        ax_neu.plot(delta_neu, y, linestyle=style, color=color)

        idx = (np.abs(delta_neu - 0.4)).argmin()
        if 0 <= delta_neu[idx] <= delta_neu.max():
            ax_neu.text(delta_neu[idx], y[idx], column, fontsize=16, color=color, clip_on=True)

    ax_neu.set_title("Neutron Energy Levels in $^{164}$Tb", fontsize=16)
    ax_neu.set_xlabel(r'$\delta$', fontsize=16)
    ax_neu.tick_params(axis='x', labelsize=16)
    ax_neu.tick_params(axis='y', labelsize=16)
    ax_neu.set_xlim(0, 0.6)
    ax_neu.set_ylim(6.0, 8.0)
    ax_neu.grid(axis="x", linestyle="--", alpha=0.7)

    # Plot the labels for the key orbitals on the left edge of each subplot
    # Proton labels
    ax_pro.text(
        0.00, 5.12,r"g$_{7/2}$", fontsize=16, color='black'
    )
    ax_pro.text(
        0.00, 5.25,r"d$_{5/2}$", fontsize=16, color='black'
    )
    ax_pro.text(
        0.00, 5.40,r"h$_{11/2}$", fontsize=16, color='black'
    )
    ax_pro.text(
        0.00, 5.48,r"d$_{3/2}$", fontsize=16, color='black'
    )
    ax_pro.text(
        0.00, 5.60,r"s$_{1/2}$", fontsize=16, color='black'
    )

    # Add shell closure circles with numbers
    ax_pro.text(
        0.05, 5.05, "50",  # (x, y, label text)
        fontsize=16,
        ha="center", va="center",
        bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", lw=1.5)
    )

    # Neutron labels
    ax_neu.text(
        0.00, 6.1,r'h$_{11/2}$', fontsize=16, color='blue'
    )
    ax_neu.text(
        0.00, 6.47,r'f$_{7/2}$', fontsize=16, color='blue'
    )
    ax_neu.text(
        0.00, 6.64,r'h$_{9/2}$', fontsize=16, color='blue'
    )
    ax_neu.text(
        0.00, 6.78,r'p$_{3/2}$', fontsize=16, color='blue'
    )
    ax_neu.text(
        0.00, 6.83,r'f$_{5/2}$', fontsize=16, color='blue'
    )
    ax_neu.text(
        0.00, 6.93,r'i$_{13/2}$', fontsize=16, color='black'
    )

    ax_neu.text(
        0.05, 6.25, "82",  # (x, y, label text)
        fontsize=16,
        ha="center", va="center",
        bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", lw=1.5)
    )

    # ---- Shared layout ----
    plt.tight_layout()
    plt.savefig("164Tb_protons_neutrons.pdf", dpi=300, bbox_inches="tight")
    plt.show()

else:
    # Load data
    df_neutrons = pd.read_csv(
        "deformed.dat",
        delim_whitespace=True,
        skiprows=1,
        comment=None
    )

    # Clean up first column
    df_neutrons.rename(columns={df_neutrons.columns[0]: "delta"}, inplace=True)

    # Filter only positive delta
    if pos_delta:
        df_neutrons = df_neutrons[df_neutrons["delta"] >= 0]
        # df_neutron = df_neutrons[df_neutrons["delta"] < 0.4]
    delta = df_neutrons["delta"].to_numpy()

    plt.figure(figsize=(12, 8))

    for column in df_neutrons.columns[1:]:
        y = df_neutrons[column].to_numpy()

        # Skip orbitals whose energy at delta_min exceeds E_max
        if y[0] > E_max:
            continue

        # Style: negative parity = blue & dashed, positive parity = black & solid
        if "-" in column:
            style = "--"
            color = "blue"
        else:
            style = "-"
            color = "black"

        # Plot the orbital
        plt.plot(delta, y, linestyle=style, color=color)

        # Add a flexible label along the curve around delta = 0.3
        idx = (np.abs(delta - 0.4)).argmin()
        plt.text(
            delta[idx], y[idx], column, fontsize=13, color=color, clip_on=True,
            # bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1)
        )

    # Add specific labels at left edge for key orbitals

    # plt.text(
    #     0.00, 6.09,r'g$_{9/2}$', fontsize=18, color='blue'
    # )
    plt.text(
        0.00, 6.1,r'h$_{11/2}$', fontsize=16, color='blue'
    )

    plt.text(
        0.00, 6.47,r'f$_{7/2}$', fontsize=18, color='blue'
    )
    plt.text(
        0.00, 6.64,r'h$_{9/2}$', fontsize=18, color='blue'
    )
    plt.text(
        0.00, 6.78,r'p$_{3/2}$', fontsize=18, color='blue'
    )
    plt.text(
        0.00, 6.83,r'f$_{5/2}$', fontsize=18, color='blue'
    )
    plt.text(
        0.00, 6.93,r'i$_{13/2}$', fontsize=18, color='black'
    )

    plt.text(
    0.05, 6.25, "82",  # (x, y, label text)
    fontsize=16,
    ha="center", va="center",
    bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", lw=1.5)
    )


    plt.xlabel(r'$\delta$', fontsize=16)
    plt.ylabel(r'Neutron single-particle levels ($\hbar \omega$)', fontsize=16)
    plt.title("Neutron Energy Levels in $^{157}$Sm")
    # increase font size of ticks
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Force x-axis to start at zero
    plt.xlim(0, delta.max())
    plt.ylim(6.0, 8.0)

    # Add grid for x-axis
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save as pdf
    plt.savefig('157Sm_neutrons.pdf', dpi=300, bbox_inches='tight')

    plt.show()
