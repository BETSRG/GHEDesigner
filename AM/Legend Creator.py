import matplotlib.pyplot as plt

# ---- labels (edit if needed) ----
labels = [
    "Tilted Boreholes",  # gray line
    "Borehole Top",      # blue dot
    "Borhole Bottom",        # red x
    "Tilt"               # yellow line
]

outfile = "tilted_legend.png"
# ---------------------------------

n_items = len(labels)
y_vals = list(reversed(range(n_items)))   # positions: top to bottom
x_symbol_start = 0.08
x_symbol_end   = 0.18
x_text         = 0.22

fig, ax = plt.subplots(figsize=(2.6, 1.4), dpi=200)

for i, (y, label) in enumerate(zip(y_vals, labels)):
    if i == 0:
        # Tilted Boreholes: gray horizontal line
        ax.plot([x_symbol_start, x_symbol_end], [y, y],
                linewidth=2, color='0.6')
    elif i == 1:
        # Tilted Entry: blue dot
        ax.scatter(x_symbol_start + 0.05, y,
                   s=25, color='blue', marker='o')
    elif i == 2:
        # Tilted End: red x
        ax.scatter(x_symbol_start + 0.05, y,
                   s=25, color='red', marker='x', linewidths=2)
    elif i == 3:
        # Tilt: yellow horizontal line
        ax.plot([x_symbol_start, x_symbol_end], [y, y],
                linewidth=2, color='yellow')

    ax.text(x_text, y, label,
            va='center', ha='left',
            fontsize=8, color='#4c5a6a')  # gray-blue text

# Clean up axes so it just looks like a legend
ax.set_xlim(0, 1)
ax.set_ylim(-0.5, n_items - 0.5)
ax.axis('off')

plt.tight_layout()
fig.savefig(outfile, bbox_inches='tight', transparent=True)
plt.close(fig)

print(f"Saved legend image as {outfile}")
