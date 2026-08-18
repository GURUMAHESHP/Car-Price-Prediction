import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import random
from datetime import datetime

import pandas as pd
import joblib

# Optional fallback for models saved with standard pickle.
import pickle

# Matplotlib is only used for the prediction-history graph.
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "car_price_model.pkl"
DATASET_FILE = "cleaned_car_data.csv"
RAW_DATASET_FILE = "quikr_car.csv"
HISTORY_FILE = "prediction_history.csv"

CURRENT_YEAR = datetime.now().year


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()
root.title("Car Price Prediction System")
root.geometry("1500x900")
root.minsize(1200, 750)
root.configure(bg="#091018")


# ============================================================
# COLORS
# ============================================================

BG = "#091018"
CARD = "#14202C"
INPUT = "#213444"
CYAN = "#00D9FF"
WHITE = "#E8EEF2"
GRAY = "#AAB7C4"
GREEN = "#00E676"
RED = "#FF4D6D"
BUTTON = "#304453"
DARK_RESULT = "#0E1923"


# ============================================================
# VARIABLES
# ============================================================

company_var = tk.StringVar(value="Maruti")
year_var = tk.StringVar(value="2018")
kms_var = tk.StringVar(value="22000")
fuel_var = tk.StringVar(value="Petrol")

price_var = tk.StringVar(value="₹ --")
status_var = tk.StringVar(
    value="Enter car details or click AUTO-FILL DATA"
)


# ============================================================
# DATASET / MODEL STATE
# ============================================================

model = None
auto_data = None


# ============================================================
# HISTORY
# ============================================================

history_columns = [
    "Date",
    "Company",
    "Year",
    "Kilometres Driven",
    "Fuel Type",
    "Car Age",
    "Predicted Price",
]


def load_history():
    """Load prediction history from CSV."""
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=history_columns)

    try:
        df = pd.read_csv(HISTORY_FILE)

        for column in history_columns:
            if column not in df.columns:
                df[column] = ""

        return df[history_columns]

    except Exception:
        return pd.DataFrame(columns=history_columns)


def save_history(company, year, kms, fuel, age, price):
    """Append one prediction to the history CSV."""
    new_record = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Company": company,
        "Year": year,
        "Kilometres Driven": kms,
        "Fuel Type": fuel,
        "Car Age": age,
        "Predicted Price": price,
    }

    df = load_history()

    df = pd.concat(
        [df, pd.DataFrame([new_record])],
        ignore_index=True,
    )

    df.to_csv(HISTORY_FILE, index=False)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_trained_model():
    """
    Load the trained Random Forest model.

    The training program may have saved the model with joblib,
    while older versions may have used standard pickle.
    Therefore joblib is tried first and pickle is used as fallback.
    """
    global model

    if not os.path.exists(MODEL_FILE):
        messagebox.showerror(
            "Model Error",
            f"{MODEL_FILE} was not found.\n\n"
            "Make sure the trained model file is in the same "
            "folder as this program.",
            parent=root,
        )
        return False

    joblib_error = None
    pickle_error = None

    # First try joblib.
    try:
        model = joblib.load(MODEL_FILE)
        return True
    except Exception as exc:
        joblib_error = exc

    # Fallback to standard pickle.
    try:
        with open(MODEL_FILE, "rb") as file:
            model = pickle.load(file)
        return True
    except Exception as exc:
        pickle_error = exc

    messagebox.showerror(
        "Model Error",
        "Unable to load the trained model.\n\n"
        "The model file may have been saved with a different "
        "Python/scikit-learn environment.\n\n"
        f"Joblib error:\n{joblib_error}\n\n"
        f"Pickle error:\n{pickle_error}",
        parent=root,
    )

    return False


# ============================================================
# LOAD DATASET FOR AUTO-FILL
# ============================================================

def clean_dataset_for_autofill(df):
    """
    Clean the dataset using the EXACT semantic column names expected
    by this project.

    IMPORTANT:
    We do not infer Company from column position. This prevents values
    such as 2012 from accidentally appearing in the Company dropdown.
    """
    # Normalize column names only for matching.
    normalized_columns = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    aliases = {
        "company": ["company", "make", "manufacturer"],
        "year": ["year", "manufacturing_year", "manufacturing year"],
        "kms_driven": [
            "kms_driven",
            "kms driven",
            "kilometres_driven",
            "kilometers_driven",
            "kilometres driven",
            "kilometers driven",
        ],
        "fuel_type": ["fuel_type", "fuel type", "fuel"],
    }

    selected = {}

    for target, possible_names in aliases.items():
        for name in possible_names:
            if name in normalized_columns:
                selected[target] = normalized_columns[name]
                break

    required = ["company", "year", "kms_driven", "fuel_type"]

    if any(col not in selected for col in required):
        return None

    # Select ONLY the intended columns and rename them.
    data = df[
        [
            selected["company"],
            selected["year"],
            selected["kms_driven"],
            selected["fuel_type"],
        ]
    ].copy()

    data.columns = [
        "company",
        "year",
        "kms_driven",
        "fuel_type",
    ]

    # --------------------------------------------------------
    # COMPANY
    # --------------------------------------------------------
    data["company"] = (
        data["company"]
        .astype(str)
        .str.strip()
    )

    # Reject values that are clearly years/numbers.
    company_numeric = pd.to_numeric(
        data["company"],
        errors="coerce",
    )

    data.loc[
        company_numeric.notna(),
        "company"
    ] = pd.NA

    # --------------------------------------------------------
    # FUEL
    # --------------------------------------------------------
    data["fuel_type"] = (
        data["fuel_type"]
        .astype(str)
        .str.strip()
    )

    # Reject numeric fuel values too.
    fuel_numeric = pd.to_numeric(
        data["fuel_type"],
        errors="coerce",
    )

    data.loc[
        fuel_numeric.notna(),
        "fuel_type"
    ] = pd.NA

    # --------------------------------------------------------
    # YEAR
    # --------------------------------------------------------
    data["year"] = pd.to_numeric(
        data["year"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # KILOMETRES
    # --------------------------------------------------------
    data["kms_driven"] = (
        data["kms_driven"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("kms", "", case=False, regex=False)
        .str.replace("km", "", case=False, regex=False)
        .str.strip()
    )

    data["kms_driven"] = pd.to_numeric(
        data["kms_driven"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # REMOVE INVALID ROWS
    # --------------------------------------------------------
    data = data.dropna(
        subset=[
            "company",
            "year",
            "kms_driven",
            "fuel_type",
        ]
    )

    data = data[
        (data["company"].str.len() > 0)
        & (data["fuel_type"].str.len() > 0)
        & (data["year"] >= 1990)
        & (data["year"] <= CURRENT_YEAR)
        & (data["kms_driven"] >= 0)
        & (data["kms_driven"] <= 2_000_000)
    ]

    # Extra safety: a Company must contain at least one letter.
    data = data[
        data["company"].str.contains(
            r"[A-Za-z]",
            regex=True,
            na=False,
        )
    ]

    return data.reset_index(drop=True)


def load_autofill_dataset():
    """Load cleaned dataset, falling back to the original dataset."""
    global auto_data

    candidate_files = [
        DATASET_FILE,
        RAW_DATASET_FILE,
    ]

    for file_name in candidate_files:
        if not os.path.exists(file_name):
            continue

        try:
            df = pd.read_csv(file_name)
            cleaned = clean_dataset_for_autofill(df)

            if cleaned is not None and not cleaned.empty:
                auto_data = cleaned
                return True

        except Exception:
            continue

    auto_data = None
    return False


# ============================================================
# DYNAMIC COMBOBOX VALUES
# ============================================================

def update_combo_values():
    """Populate dropdowns from the validated semantic dataset columns."""
    companies = []
    fuels = []

    if auto_data is not None and not auto_data.empty:
        # Final defensive filtering: company names must contain letters
        # and must not be pure numeric values.
        company_series = (
            auto_data["company"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        company_series = company_series[
            company_series.str.contains(
                r"[A-Za-z]",
                regex=True,
                na=False,
            )
        ]

        companies = sorted(
            company_series.unique().tolist()
        )

        fuel_series = (
            auto_data["fuel_type"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        fuel_series = fuel_series[
            fuel_series.str.contains(
                r"[A-Za-z]",
                regex=True,
                na=False,
            )
        ]

        fuels = sorted(
            fuel_series.unique().tolist()
        )

    # Fallback values if dataset is unavailable.
    if not companies:
        companies = [
            "Maruti",
            "Hyundai",
            "Honda",
            "Toyota",
            "Ford",
            "Mahindra",
            "Tata",
            "Volkswagen",
            "Renault",
            "Nissan",
            "Skoda",
            "Chevrolet",
            "Fiat",
        ]

    if not fuels:
        fuels = [
            "Petrol",
            "Diesel",
            "CNG",
        ]

    company_combo["values"] = tuple(companies)
    fuel_combo["values"] = tuple(fuels)

    # Always start with a valid company/fuel value.
    if company_var.get() not in companies:
        company_var.set(companies[0])

    if fuel_var.get() not in fuels:
        fuel_var.set(fuels[0])


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(root, bg=BG)

header.pack(
    fill="x",
    pady=(25, 5),
)

title = tk.Label(
    header,
    text="🚙  CAR PRICE PREDICTION SYSTEM",
    font=("Segoe UI", 28, "bold"),
    fg=CYAN,
    bg=BG,
)

title.pack()

subtitle = tk.Label(
    header,
    text="Machine Learning  •  Random Forest Regression",
    font=("Segoe UI", 13),
    fg=GRAY,
    bg=BG,
)

subtitle.pack(pady=(5, 15))


# ============================================================
# MAIN CARD
# ============================================================

main_card = tk.Frame(
    root,
    bg=CARD,
)

main_card.pack(
    fill="both",
    expand=True,
    padx=45,
    pady=15,
)


# ============================================================
# CAR INFORMATION TITLE
# ============================================================

info_title = tk.Label(
    main_card,
    text="CAR INFORMATION",
    font=("Segoe UI", 18, "bold"),
    fg=CYAN,
    bg=CARD,
)

info_title.pack(pady=(30, 20))


# ============================================================
# INPUT FRAME
# ============================================================

form = tk.Frame(
    main_card,
    bg=CARD,
)

form.pack()


def create_label(text, row):
    label = tk.Label(
        form,
        text=text,
        font=("Segoe UI", 12),
        fg=GRAY,
        bg=CARD,
        anchor="w",
        width=22,
    )

    label.grid(
        row=row,
        column=0,
        padx=15,
        pady=12,
        sticky="w",
    )

    return label


def create_entry(variable, row):
    entry = tk.Entry(
        form,
        textvariable=variable,
        font=("Segoe UI", 12),
        bg=INPUT,
        fg=WHITE,
        insertbackground=WHITE,
        relief="flat",
        width=30,
    )

    entry.grid(
        row=row,
        column=1,
        padx=15,
        pady=12,
        ipady=8,
    )

    return entry


# Company
create_label("Company", 0)

company_combo = ttk.Combobox(
    form,
    textvariable=company_var,
    font=("Segoe UI", 12),
    width=28,
    state="readonly",
)

company_combo.grid(
    row=0,
    column=1,
    padx=15,
    pady=12,
    ipady=5,
)


# Year
create_label("Manufacturing Year", 1)
year_entry = create_entry(year_var, 1)


# Kilometres
create_label("Kilometres Driven", 2)
kms_entry = create_entry(kms_var, 2)


# Fuel
create_label("Fuel Type", 3)

fuel_combo = ttk.Combobox(
    form,
    textvariable=fuel_var,
    font=("Segoe UI", 12),
    width=28,
    state="readonly",
)

fuel_combo.grid(
    row=3,
    column=1,
    padx=15,
    pady=12,
    ipady=5,
)


# ============================================================
# PRICE RESULT
# ============================================================

result_frame = tk.Frame(
    main_card,
    bg=DARK_RESULT,
    highlightbackground="#23394A",
    highlightthickness=1,
)

result_frame.pack(
    fill="x",
    padx=75,
    pady=(25, 10),
)

result_title = tk.Label(
    result_frame,
    text="PREDICTED CAR PRICE",
    font=("Segoe UI", 13, "bold"),
    fg=GRAY,
    bg=DARK_RESULT,
)

result_title.pack(pady=(20, 5))

price_label = tk.Label(
    result_frame,
    textvariable=price_var,
    font=("Segoe UI", 30, "bold"),
    fg=GREEN,
    bg=DARK_RESULT,
)

price_label.pack(pady=(0, 15))

status_label = tk.Label(
    result_frame,
    textvariable=status_var,
    font=("Segoe UI", 11),
    fg=GREEN,
    bg=DARK_RESULT,
)

status_label.pack(pady=(0, 20))


# ============================================================
# MODEL INPUT PREPARATION
# ============================================================

def prepare_model_input(company, year, kms, fuel, car_age):
    """
    Create the input DataFrame used by the trained model.

    The project model is expected to use:
        company
        year
        kms_driven
        fuel_type
        car_age

    If the loaded model exposes feature_names_in_, the columns are
    reordered to match the model exactly.
    """
    input_data = pd.DataFrame(
        {
            "company": [company],
            "year": [year],
            "kms_driven": [kms],
            "fuel_type": [fuel],
            "car_age": [car_age],
        }
    )

    # Match the exact training-column order when available.
    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is not None:
        expected = list(feature_names)

        if set(expected) == set(input_data.columns):
            input_data = input_data[expected]

    return input_data


# ============================================================
# PREDICTION
# ============================================================

def predict_price():
    """Validate user input, predict price, and save history."""
    try:
        if model is None:
            raise RuntimeError(
                "The trained model is not loaded."
            )

        company = company_var.get().strip()
        fuel = fuel_var.get().strip()

        if not company:
            raise ValueError("Please select a company.")

        if not fuel:
            raise ValueError("Please select a fuel type.")

        year_text = year_var.get().strip()

        if not year_text:
            raise ValueError(
                "Please enter the manufacturing year."
            )

        kms_text = (
            kms_var.get()
            .replace(",", "")
            .replace(" ", "")
            .strip()
        )

        if not kms_text:
            raise ValueError(
                "Please enter kilometres driven."
            )

        year = int(float(year_text))
        kms = float(kms_text)

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if year < 1990 or year > CURRENT_YEAR:
            raise ValueError(
                f"Manufacturing year must be between "
                f"1990 and {CURRENT_YEAR}."
            )

        if kms < 0:
            raise ValueError(
                "Kilometres driven cannot be negative."
            )

        if kms > 2_000_000:
            raise ValueError(
                "Please enter a realistic kilometres value."
            )

        # ----------------------------------------------------
        # CAR AGE
        # ----------------------------------------------------

        car_age = CURRENT_YEAR - year

        # ----------------------------------------------------
        # CREATE INPUT DATA
        # ----------------------------------------------------

        input_data = prepare_model_input(
            company,
            year,
            kms,
            fuel,
            car_age,
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)

        predicted_price = float(prediction[0])

        if predicted_price < 0:
            predicted_price = 0.0

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        price_var.set(
            f"₹ {predicted_price:,.0f}"
        )

        status_var.set(
            f"Prediction completed  •  "
            f"Car age: {car_age} years"
        )

        status_label.config(
            fg=GREEN
        )

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        save_history(
            company,
            year,
            kms,
            fuel,
            car_age,
            round(predicted_price, 2),
        )

    except ValueError as exc:
        status_label.config(fg=RED)

        messagebox.showwarning(
            "Invalid Input",
            str(exc),
            parent=root,
        )

    except Exception as exc:
        status_label.config(fg=RED)

        messagebox.showerror(
            "Prediction Error",
            "An error occurred while predicting the price.\n\n"
            f"{exc}",
            parent=root,
        )


# ============================================================
# AUTO-FILL FROM DATASET
# ============================================================

def auto_fill_data():
    """
    Automatically fill the GUI with a valid car record from
    cleaned_car_data.csv.

    The price is intentionally NOT filled because price is the
    target that the ML model is supposed to predict.
    """
    global auto_data

    if auto_data is None or auto_data.empty:
        if not load_autofill_dataset():
            messagebox.showwarning(
                "Dataset Not Found",
                f"Could not find a usable dataset.\n\n"
                f"Expected:\n{DATASET_FILE}\n"
                f"or:\n{RAW_DATASET_FILE}",
                parent=root,
            )
            return

    try:
        # Rebuild dropdowns from the validated dataset before selecting
        # a row, so the GUI and AUTO-FILL always use the same schema.
        update_combo_values()

        row = auto_data.sample(
            n=1,
            random_state=random.randint(0, 999999),
        ).iloc[0]

        company_var.set(str(row["company"]).strip())
        year_var.set(str(int(float(row["year"]))))
        kms_var.set(
            f"{float(row['kms_driven']):,.0f}"
        )
        fuel_var.set(str(row["fuel_type"]).strip())

        price_var.set("₹ --")

        status_var.set(
            "Valid car data automatically loaded from dataset"
        )

        status_label.config(
            fg=CYAN
        )

    except Exception as exc:
        messagebox.showerror(
            "Auto-Fill Error",
            f"Could not load sample data.\n\n{exc}",
            parent=root,
        )


# ============================================================
# FIXED SAMPLE DATA
# ============================================================

def load_sample():
    """Load the same sample used during project testing."""
    company_var.set("Maruti")
    year_var.set("2018")
    kms_var.set("22,000")
    fuel_var.set("Petrol")

    price_var.set("₹ --")

    status_var.set(
        "Sample data loaded successfully"
    )

    status_label.config(
        fg=CYAN
    )


# ============================================================
# CLEAR
# ============================================================

def clear_fields():
    """Clear input fields."""
    company_values = company_combo["values"]
    fuel_values = fuel_combo["values"]

    company_var.set(
        company_values[0] if company_values else ""
    )

    year_var.set("")
    kms_var.set("")

    fuel_var.set(
        fuel_values[0] if fuel_values else ""
    )

    price_var.set("₹ --")

    status_var.set(
        "Fields cleared"
    )

    status_label.config(
        fg=GRAY
    )

    year_entry.focus_set()


# ============================================================
# MODEL INFO
# ============================================================

def show_model_info():
    """Display project and model information."""
    messagebox.showinfo(
        "Model Information",
        """
CAR PRICE PREDICTION SYSTEM

Machine Learning Model:
Random Forest Regression

Features Used:
• Company
• Manufacturing Year
• Kilometres Driven
• Fuel Type
• Car Age

Dataset:
Cleaned Quikr Car Dataset

Preprocessing:
One-Hot Encoding

Output:
Estimated resale price of a used car.

Application Features:
• Automatic dataset-based input
• Sample data
• Price prediction
• Prediction history
• CSV export
• Price visualization
""",
        parent=root,
    )


# ============================================================
# HISTORY WINDOW
# ============================================================

def show_history():
    """Display, export, graph, and clear prediction history."""
    history_window = tk.Toplevel(root)

    history_window.title(
        "Prediction History"
    )

    history_window.geometry(
        "1100x650"
    )

    history_window.configure(
        bg=BG
    )

    tk.Label(
        history_window,
        text="📋  PREDICTION HISTORY",
        font=("Segoe UI", 22, "bold"),
        fg=CYAN,
        bg=BG,
    ).pack(
        pady=(20, 15)
    )

    table_frame = tk.Frame(
        history_window,
        bg=BG,
    )

    table_frame.pack(
        fill="both",
        expand=True,
        padx=20,
        pady=10,
    )

    columns = (
        "Date",
        "Company",
        "Year",
        "KM",
        "Fuel",
        "Age",
        "Price",
    )

    tree = ttk.Treeview(
        table_frame,
        columns=columns,
        show="headings",
    )

    headings = {
        "Date": "Date",
        "Company": "Company",
        "Year": "Year",
        "KM": "KM Driven",
        "Fuel": "Fuel Type",
        "Age": "Car Age",
        "Price": "Predicted Price",
    }

    widths = {
        "Date": 160,
        "Company": 120,
        "Year": 80,
        "KM": 120,
        "Fuel": 100,
        "Age": 80,
        "Price": 150,
    }

    for col in columns:
        tree.heading(
            col,
            text=headings[col],
        )

        tree.column(
            col,
            width=widths[col],
            anchor="center",
        )

    scrollbar = ttk.Scrollbar(
        table_frame,
        orient="vertical",
        command=tree.yview,
    )

    tree.configure(
        yscrollcommand=scrollbar.set
    )

    tree.pack(
        side="left",
        fill="both",
        expand=True,
    )

    scrollbar.pack(
        side="right",
        fill="y",
    )

    def refresh_table():
        for item in tree.get_children():
            tree.delete(item)

        df = load_history()

        for _, row in df.iterrows():
            try:
                kms_display = (
                    f"{float(row['Kilometres Driven']):,.0f}"
                )
            except Exception:
                kms_display = str(
                    row["Kilometres Driven"]
                )

            try:
                price_display = (
                    f"₹ {float(row['Predicted Price']):,.0f}"
                )
            except Exception:
                price_display = str(
                    row["Predicted Price"]
                )

            tree.insert(
                "",
                "end",
                values=(
                    row["Date"],
                    row["Company"],
                    row["Year"],
                    kms_display,
                    row["Fuel Type"],
                    row["Car Age"],
                    price_display,
                ),
            )

    refresh_table()

    button_frame = tk.Frame(
        history_window,
        bg=BG,
    )

    button_frame.pack(
        pady=15,
    )

    def delete_history():
        if not os.path.exists(HISTORY_FILE):
            messagebox.showinfo(
                "History",
                "No prediction history exists.",
                parent=history_window,
            )
            return

        confirm = messagebox.askyesno(
            "Clear History",
            "Are you sure you want to delete all prediction history?",
            parent=history_window,
        )

        if not confirm:
            return

        try:
            os.remove(HISTORY_FILE)
            refresh_table()

            messagebox.showinfo(
                "History",
                "Prediction history cleared.",
                parent=history_window,
            )

        except Exception as exc:
            messagebox.showerror(
                "Error",
                str(exc),
                parent=history_window,
            )

    def export_history():
        df = load_history()

        if df.empty:
            messagebox.showinfo(
                "Export",
                "There is no prediction history to export.",
                parent=history_window,
            )
            return

        file_path = filedialog.asksaveasfilename(
            parent=history_window,
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
            ],
            initialfile="prediction_history.csv",
        )

        if not file_path:
            return

        try:
            df.to_csv(
                file_path,
                index=False,
            )

            messagebox.showinfo(
                "Export Successful",
                "Prediction history exported successfully.",
                parent=history_window,
            )

        except Exception as exc:
            messagebox.showerror(
                "Export Error",
                str(exc),
                parent=history_window,
            )

    def show_graph():
        df = load_history()

        if df.empty:
            messagebox.showinfo(
                "Graph",
                "Make at least one prediction first.",
                parent=history_window,
            )
            return

        prices = pd.to_numeric(
            df["Predicted Price"],
            errors="coerce",
        ).dropna()

        if prices.empty:
            messagebox.showinfo(
                "Graph",
                "No valid prediction values were found.",
                parent=history_window,
            )
            return

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            range(1, len(prices) + 1),
            prices,
            marker="o",
        )

        plt.title(
            "Car Price Prediction History"
        )

        plt.xlabel(
            "Prediction Number"
        )

        plt.ylabel(
            "Predicted Price (₹)"
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.tight_layout()
        plt.show()

    def make_history_button(
        text,
        command,
        bg_color=BUTTON,
        fg_color=WHITE,
    ):
        return tk.Button(
            button_frame,
            text=text,
            command=command,
            font=("Segoe UI", 11, "bold"),
            bg=bg_color,
            fg=fg_color,
            activebackground=CYAN,
            activeforeground="black",
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2",
        )

    make_history_button(
        "📊 VIEW GRAPH",
        show_graph,
        CYAN,
        "black",
    ).pack(
        side="left",
        padx=8,
    )

    make_history_button(
        "💾 EXPORT CSV",
        export_history,
    ).pack(
        side="left",
        padx=8,
    )

    make_history_button(
        "🗑 CLEAR HISTORY",
        delete_history,
        RED,
        "white",
    ).pack(
        side="left",
        padx=8,
    )

    make_history_button(
        "CLOSE",
        history_window.destroy,
    ).pack(
        side="left",
        padx=8,
    )


# ============================================================
# MAIN BUTTONS
# ============================================================

button_frame = tk.Frame(
    main_card,
    bg=CARD,
)

button_frame.pack(
    pady=25,
)


def make_button(
    parent,
    text,
    command,
    bg_color=BUTTON,
    fg_color=WHITE,
):
    return tk.Button(
        parent,
        text=text,
        command=command,
        font=("Segoe UI", 11, "bold"),
        bg=bg_color,
        fg=fg_color,
        activebackground=CYAN,
        activeforeground="black",
        relief="flat",
        padx=18,
        pady=12,
        cursor="hand2",
    )


predict_button = make_button(
    button_frame,
    "🔮  PREDICT PRICE",
    predict_price,
    CYAN,
    "black",
)

predict_button.pack(
    side="left",
    padx=6,
)


auto_button = make_button(
    button_frame,
    "⚡  AUTO-FILL DATA",
    auto_fill_data,
)

auto_button.pack(
    side="left",
    padx=6,
)


sample_button = make_button(
    button_frame,
    "📌  SAMPLE",
    load_sample,
)

sample_button.pack(
    side="left",
    padx=6,
)


clear_button = make_button(
    button_frame,
    "↻  CLEAR",
    clear_fields,
)

clear_button.pack(
    side="left",
    padx=6,
)


info_button = make_button(
    button_frame,
    "ⓘ  MODEL INFO",
    show_model_info,
)

info_button.pack(
    side="left",
    padx=6,
)


history_button = make_button(
    button_frame,
    "📋  HISTORY",
    show_history,
)

history_button.pack(
    side="left",
    padx=6,
)


# ============================================================
# FOOTER
# ============================================================

footer = tk.Label(
    root,
    text="AI/ML Internship Project  •  Car Price Prediction",
    font=("Segoe UI", 10),
    fg="#71808E",
    bg=BG,
)

footer.pack(
    pady=(0, 15),
)


# ============================================================
# KEYBOARD SHORTCUT
# ============================================================

root.bind(
    "<Return>",
    lambda event: predict_price(),
)


# ============================================================
# STARTUP
# ============================================================

# Load dataset first so the dropdowns can use the actual dataset.
load_autofill_dataset()

# Load company/fuel values from the dataset.
update_combo_values()

# Load the trained model.
if not load_trained_model():
    root.destroy()
else:
    # Focus on the main input.
    year_entry.focus_set()

    # Start GUI.
    root.mainloop()
