import tkinter as tk  # GUI library used to create the user interface
from PIL import Image, ImageTk  # Used to resize images cleanly for uniform thumbnails
import os  # Used to find all chart image files in the outputs folder


# ========================================
# INSURANCE RISK & CLAIM ANALYSIS PROJECT
# 06 - RESULTS DASHBOARD (GUI)
# ========================================

BACKGROUND_COLOUR = "#f4f6f8"
HEADER_COLOUR = "#1f3864"
CARD_COLOUR = "#ffffff"

THUMBNAIL_WIDTH = 320   # every thumbnail is resized to this width, so they're all consistent
THUMBNAIL_HEIGHT = 200
COLUMNS = 3             # how many thumbnails per row

# This list must exist for the whole program, or tkinter will
# "forget" the images once they go out of scope and they'll
# disappear from the screen even though the code looks correct.
loaded_images = []


# ----------------------------------------
# KEY INSIGHT FOR EACH CHART
# ----------------------------------------

# A short, plain-English summary of what each chart actually shows,
# with the key number highlighted. Keyed by filename so it's easy
# to look up whichever chart is being displayed.
CHART_DESCRIPTIONS = {
    "chart1_claim_amount_distribution.png":
        "Shows how claim amounts are spread across all 1,000 claims. Most claims fall into two "
        "groups: a low-value spike around £5,000-8,000, and a broader peak around £55,000-65,000 — "
        "suggesting two distinct types of incident.",

    "chart2_fraud_by_severity.png":
        "Compares fraud rate across incident severity levels. Major Damage claims are fraudulent "
        "about 60% of the time — roughly 5x higher than every other severity level.",

    "chart3_fraud_by_insured_hobbies.png":
        "Compares fraud rate across 20 hobbies. Chess (83%) and Cross-fit (74%) stand out sharply "
        "above every other hobby, which mostly sit between 9-30%.",

    "chart4_incidents_by_hour.png":
        "Shows how many incidents occurred at each hour of the day. Incidents spike around midnight, "
        "3am, and 5pm, with quieter periods around 1-2am and 11am.",

    "chart5_average_claim_by_education.png":
        "Compares average claim amount across education levels. Claim amounts rise mildly with "
        "education, from around £49,000 (Associate) to £55,000 (MD/PhD) — about a 13% difference.",

    "chart6_correlation_heatmap.png":
        "Shows how every pair of numeric variables relates to each other, from -1 (opposite "
        "directions) to +1 (move together) — the diagonal is always 1.00, since anything is "
        "perfectly correlated with itself. The strongest real relationship (0.98) is between "
        "total_claim_amount and vehicle_claim, which makes sense since total_claim_amount is "
        "calculated directly from vehicle_claim, injury_claim, and property_claim added together. "
        "Age and months_as_customer are also strongly linked (0.92) — older customers have "
        "typically held their policy longer. Most other pairs sit close to 0, meaning the vast "
        "majority of variables have no meaningful relationship with each other, which is actually "
        "a good sign: it means the dataset isn't full of hidden duplicate information. Only "
        "numeric columns are shown here, since correlation can't be calculated on text categories "
        "like incident_severity or insured_hobbies — those are explored separately using chi-square "
        "tests instead.",

    "chart7_feature_importance.png":
        "Ranks which clues the Random Forest model relied on most to predict fraud. Incident severity "
        "(Major Damage) is by far the top predictor, followed by the chess hobby outlier, claim "
        "amount, and policy premium.",
}

DEFAULT_DESCRIPTION = "No description available yet for this chart."


# ----------------------------------------
# TURN A FILENAME INTO A READABLE TITLE
# ----------------------------------------

def make_readable_title(filename):
    # "chart2_fraud_by_severity.png" -> "Fraud By Severity"
    name = filename.replace('.png', '')
    parts = name.split('_')[1:]  # drop the "chartN" part at the start
    return ' '.join(parts).title()


# ----------------------------------------
# RESIZE AN IMAGE TO A FIXED THUMBNAIL SIZE
# ----------------------------------------

def make_thumbnail(path):
    # Open with PIL, which resizes far more cleanly than tkinter's own
    # subsample() (which only shrinks by whole-number factors and gives
    # inconsistent sizes when the original charts have different shapes)
    image = Image.open(path)
    image.thumbnail((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.LANCZOS)
    return ImageTk.PhotoImage(image)


# ----------------------------------------
# OPEN A FULL-SIZE VERSION OF A CHART
# ----------------------------------------

def show_full_size(path, title):
    popup = tk.Toplevel(window)
    popup.title(title)
    popup.configure(bg=BACKGROUND_COLOUR)

    # Scale the chart to fit comfortably within the screen — capping
    # ONLY width (like before) let tall charts open TALLER than the
    # screen, pushing the top (title) off-screen. Capping both width
    # and height fixes that.
    image = Image.open(path)
    screen_width = popup.winfo_screenwidth()
    screen_height = popup.winfo_screenheight()
    max_width = min(1100, int(screen_width * 0.85))
    max_height = int(screen_height * 0.75)  # leave room for the description + close button

    width_ratio = max_width / image.width
    height_ratio = max_height / image.height
    scale = min(width_ratio, height_ratio, 1.0)  # never scale UP, only down

    if scale < 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)

    full_image = ImageTk.PhotoImage(image)
    loaded_images.append(full_image)  # keep it alive

    label = tk.Label(popup, image=full_image, bg=BACKGROUND_COLOUR)
    label.pack(padx=10, pady=(10, 5))

    # Key insight panel, highlighted so it stands out from the chart itself
    filename = os.path.basename(path)
    description = CHART_DESCRIPTIONS.get(filename, DEFAULT_DESCRIPTION)

    insight_frame = tk.Frame(popup, bg="#fff4d6", highlightbackground="#e8c874",
                              highlightthickness=1)
    insight_frame.pack(fill="x", padx=20, pady=(5, 10))

    insight_label = tk.Label(
        insight_frame,
        text="Key Insight",
        font=("Segoe UI", 10, "bold"),
        bg="#fff4d6",
        fg="#7a5b00",
        anchor="w"
    )
    insight_label.pack(fill="x", padx=12, pady=(8, 0))

    description_label = tk.Label(
        insight_frame,
        text=description,
        font=("Segoe UI", 10),
        bg="#fff4d6",
        fg="#333333",
        wraplength=min(max_width, 900),
        justify="left",
        anchor="w"
    )
    description_label.pack(fill="x", padx=12, pady=(2, 10))

    # Center the popup on screen so nothing opens off the visible area
    popup.update_idletasks()
    popup_width = popup.winfo_width()
    popup_height = popup.winfo_height()
    x = (screen_width - popup_width) // 2
    y = (screen_height - popup_height) // 2
    popup.geometry(f"+{x}+{y}")

    close_button = tk.Button(popup, text="Close", command=popup.destroy,
                              bg=HEADER_COLOUR, fg="white", relief="flat",
                              padx=10, pady=4)
    close_button.pack(pady=(0, 10))


# ----------------------------------------
# MAIN WINDOW
# ----------------------------------------

window = tk.Tk()
window.title("Insurance Risk Analysis Dashboard")
window.configure(bg=BACKGROUND_COLOUR)
window.geometry("1100x750")  # a sensible starting size, still resizable

# Header
header = tk.Label(
    window,
    text="Insurance Risk & Claims Analysis — Results Dashboard",
    font=("Segoe UI", 16, "bold"),
    bg=HEADER_COLOUR,
    fg="white",
    padx=20,
    pady=15
)
header.pack(fill="x")

subheader = tk.Label(
    window,
    text="Click any chart below to view it full size — scroll down for more",
    font=("Segoe UI", 10),
    bg=BACKGROUND_COLOUR,
    fg="#555555",
    pady=10
)
subheader.pack()


# ----------------------------------------
# SCROLLABLE AREA
# ----------------------------------------

# A plain Frame can't scroll on its own, so it's placed inside a
# Canvas (which can scroll), with a Scrollbar attached to that canvas.
canvas_container = tk.Frame(window, bg=BACKGROUND_COLOUR)
canvas_container.pack(fill="both", expand=True, padx=10, pady=10)

scroll_canvas = tk.Canvas(canvas_container, bg=BACKGROUND_COLOUR, highlightthickness=0)
scrollbar = tk.Scrollbar(canvas_container, orient="vertical", command=scroll_canvas.yview)
scroll_canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
scroll_canvas.pack(side="left", fill="both", expand=True)

# The grid of chart cards lives inside this frame, which sits inside the canvas
grid_frame = tk.Frame(scroll_canvas, bg=BACKGROUND_COLOUR)
canvas_window = scroll_canvas.create_window((0, 0), window=grid_frame, anchor="nw")

def update_scroll_region(event=None):
    scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

grid_frame.bind("<Configure>", update_scroll_region)

# Let the mouse wheel scroll the canvas too, for convenience
def on_mousewheel(event):
    scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

scroll_canvas.bind_all("<MouseWheel>", on_mousewheel)


# ----------------------------------------
# LOAD EVERY CHART FROM THE OUTPUTS FOLDER
# ----------------------------------------

chart_files = sorted([f for f in os.listdir('outputs') if f.endswith('.png')])

for index, filename in enumerate(chart_files):
    path = os.path.join('outputs', filename)
    title = make_readable_title(filename)

    # A "card" frame around each thumbnail, so it looks like a tile
    card = tk.Frame(grid_frame, bg=CARD_COLOUR, padx=10, pady=10,
                     highlightbackground="#dddddd", highlightthickness=1)
    row = index // COLUMNS
    col = index % COLUMNS
    card.grid(row=row, column=col, padx=12, pady=12)

    thumbnail = make_thumbnail(path)
    loaded_images.append(thumbnail)

    thumb_label = tk.Label(card, image=thumbnail, bg=CARD_COLOUR, cursor="hand2")
    thumb_label.pack()

    # Clicking the thumbnail opens the full-size version.
    # p=path, t=title "locks in" the correct chart for each thumbnail —
    # without this, every thumbnail would open the LAST chart in the loop.
    thumb_label.bind("<Button-1>", lambda event, p=path, t=title: show_full_size(p, t))

    caption = tk.Label(card, text=title, font=("Segoe UI", 9, "bold"),
                        bg=CARD_COLOUR, fg="#333333", pady=6)
    caption.pack()

    # Short preview of the key insight, so there's context even
    # before clicking to see the full description
    short_description = CHART_DESCRIPTIONS.get(filename, DEFAULT_DESCRIPTION)
    preview_text = short_description.split('.')[0] + "."
    preview_label = tk.Label(card, text=preview_text, font=("Segoe UI", 8),
                              bg=CARD_COLOUR, fg="#777777",
                              wraplength=THUMBNAIL_WIDTH, justify="left")
    preview_label.pack()

window.mainloop()
