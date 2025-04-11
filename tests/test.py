import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_lstm_cell():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw the main cell
    cell = patches.FancyBboxPatch((3.5, 2.5), 3, 3, boxstyle="round,pad=0.1", edgecolor='black', facecolor='#F0F8FF')
    ax.add_patch(cell)
    ax.text(5, 4, "Cell State", fontsize=12, ha='center')

    # Add forget gate
    ax.add_patch(patches.Rectangle((1, 4.5), 2, 0.8, edgecolor='black', facecolor='#FFC0CB'))
    ax.text(2, 4.9, "Forget Gate", ha='center', va='center', fontsize=10)
    ax.arrow(3, 5, 0.5, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

    # Add input gate
    ax.add_patch(patches.Rectangle((1, 3.2), 2, 0.8, edgecolor='black', facecolor='#90EE90'))
    ax.text(2, 3.6, "Input Gate", ha='center', va='center', fontsize=10)
    ax.arrow(3, 3.6, 0.5, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

    # Add output gate
    ax.add_patch(patches.Rectangle((1, 2), 2, 0.8, edgecolor='black', facecolor='#ADD8E6'))
    ax.text(2, 2.4, "Output Gate", ha='center', va='center', fontsize=10)
    ax.arrow(3, 2.4, 0.5, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

    # Arrows for cell state flow
    ax.arrow(4, 6, 0, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(6.5, 4.5, 0.5, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(7.1, 4.5, "Cₜ", va='center', fontsize=12)

    # Arrows and labels for input and hidden state
    ax.arrow(0.5, 3.6, 0.4, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(0, 3.6, "xₜ", fontsize=12)

    ax.arrow(0.5, 5.5, 0.4, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(0, 5.5, "hₜ₋₁", fontsize=12)

    # Hidden state output
    ax.arrow(6.5, 3, 0.5, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(7.1, 3, "hₜ", va='center', fontsize=12)

    ax.set_xlim(0, 8)
    ax.set_ylim(1.5, 6.5)
    ax.axis('off')
    plt.title("LSTM Cell - Visual Explanation", fontsize=14)
    plt.show()

draw_lstm_cell()