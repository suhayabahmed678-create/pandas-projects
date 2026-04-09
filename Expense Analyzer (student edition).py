"""
Smart Expense Tracker - Student Edition
========================================
A pandas-focused CLI expense tracker built for students.
Track daily spending, analyze habits, and stay within budget.

Author  : suhayab
GitHub  : https://github.com/suhayab
License : MIT
"""
import pandas as pd
import os
import sys
from datetime  import datetime,date


# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE = 'expense.csv'
BUDGET_FILE = 'budget.csv'

CATEGORIES = [
    "Food", "Transport", "Books", "Entertainment",
    "Clothing", "Health", "Utilities", "Other"
]

COLUMNS = ["id", "date", "category", "description", "amount"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load expenses from CSV, or create empty DataFrame."""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE , parse_dates=['date'])
        return df
    return pd.DataFrame(columns=COLUMNS)

def save_data(df: pd.DataFrame) -> None:
    df.to_csv(DATA_FILE, index=False)

def load_budget() -> pd.DataFrame:
    """Load monthly budgets per category."""
    if os.path.exists(BUDGET_FILE):
        return pd.read_csv(BUDGET_FILE)
    return pd.DataFrame(columns=["category" , "monthly limit"])

def save_budget (bdf: pd.DataFrame) -> None:
    bdf.to_csv(BUDGET_FILE, index=False)

def next_id(df: pd.DataFrame) -> int:
    return int(df["id"].max()) +1 if not df.empty else 1

def separator(char :str = "─" ,width :int = 52) -> None:
    print(char * width)

def header(title: str) -> None:
    separator("=")
    print(F"  {title}")
    separator("=")

# ── Core Features ─────────────────────────────────────────────────────────────

def add_expense (df: pd.DataFrame ) -> pd.DataFrame:
    header("➕  Add New Expense")

    print("categories:")
    for i,cat  in enumerate(CATEGORIES,1):
        print(f"{i}). {cat}")

    try:
        cat_idx = int(input("\nChoose category (number): ")) - 1
        if cat_idx not in range (len(CATEGORIES)):
            print("invalid category number")
            return df
        category = CATEGORIES[cat_idx]

        description = input("Description : ").strip() or "N/A"
        amount = float(input("Amount ($)  : "))
        if amount <= 0:
            print("Amount must be positive.")
            return df

        date_str = input("Date (YYYY-MM-DD [today] ): ").strip()
        expense_date = datetime.strptime(date_str, "%Y-%m-%d").date()\
                       if date_str else date.today()


    except ValueError:
        print("invalid input")
        return df

    new_row = pd.DataFrame([{
        "id"         : next_id(df),
        "date"       : pd.Timestamp(expense_date),
        "category"   : category,
        "description": description,
        "amount"     : amount,
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    return df 
    save_data(df)
    print(f"\n added ${amount:.2f} under '{category}'." )
    return df

def view_expense(df: pd.DataFrame) -> None:
    """Display all expenses in a clean table."""
    header("all expenses")
    if df.empty:
        print ("no expenses found")
        return

    display = df.copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    display["amount"] = display["amount"] .map("${:.2f}".format)
    print(display.to_string(index=False))
    separator()
    total = display["amount"].sum()
    print(f"  TOTAL SPEND : ${total:.2f}
          
def monthly_summary(df: pd.DataFrame) -> None:
        """Pandas groupby summary by month & category."""
        header("monthly summary")
        if df.empty:
            print ("no data found")
            return

        df ["month"] = df["date"].dt.to_period("M")

        summary = (
            df.groupby(["month", "category"]) ["amount"]
            .sum()
            .reset_index()
            .rename(columns={"amount": "total"})
            .sort_values(["month","total"] , ascending= [True,False])
        )

        for month, grp in summary.groupby("month"):
            print(f"\n  📅 {month}")
            separator("-", 40)
            for _, row in grp.iterrows():
                print (f"  {row['category']:<15} ${row['total']:>8.2f}")
            print(f" {"TOTAL":<15} ${grp['total'].sum():>8.2f}")

def budget_check(df: pd.DataFrame) -> None:
    header("budget header (This Month)")
    bdf = load_budget()
    if bdf.empty:
        print ("No budget set. Use option 5 to set budgets.")
        return

    now = pd.Timestamp.now().to_period("M")
    mask = df["date"].dt.to_peroid("M") == now
    this_month = df[mask].groupby("category") ["amount"].sum().reset_index()

    merged = bdf.merge(this_month, on="category", how="left").fillna(0)
    merged["spent"] = merged["amount"]
    merged["remaining"] = merged["monthly_limit"] - merged["spent"]
    merged["status"] = merged["remaining"].apply(
        lambda x: "ok" if x >= 0 else "over"
    )

    separator("-",52)
    print(F" {"Category":<14} {"Limit:>8"} {"Spend":>8} {"Left"} status")
    separator("-",52)

    for _, row in  merged.iterrows():
        print(
              f"{row['category']:<14} "
              f"${row['monthly Limit']:>6.0f} "
              f"${row['spent']:>6.2f} "
              f"${row['remaining']:>8.2f}"

        )

def set_budget() -> None:
    """Set or update monthly budget for a category."""
    header("set monthly budget")
    bdf = load_budget()

    for i, category in enumerate("CATEGORIES",1):
        print(f"  {i} {category} ")

    try:
        cat_idx = int(input("\nchoose category : ")) - 1
        category = CATEGORIES[cat_idx]
        limit = float(input(f"Monthly limit for {category} ($): "))
    except (ValueError, IndexError):
        print ("invalid input")
        return

    if category in bdf ["category"].values
        bdf.loc[bdf["category"] == category, "monthly_limit"] = limit
    else:
        bdf = pd.concat(
            [bdf, pd.DataFrame([{"category": category, "monthly_limit": limit}])],
            ignore_index=True
        )

    save_budget(bdf)
    print (f" budget for '{category}' is now set to {limit:.2f}/month")

def delete_expense(df:pd.DataFrame) -> pd.DataFrame:
    """Delete an expense by ID."""
    header("delete expense")
    try:
        exp_id = int(input("Enter expense ID to delete: "))
    except ValueError:
        print("invalid input")
        return df

    if exp_id not in df ["id"].values:
        print ("id not found")
        return df

    df = df[df["id"] != exp_id].reset_index(drop=True)
    save_data(df)
    print(f"✅ Expense #{exp_id} deleted.")
    return df



def top_spending(df: pd.DataFrame) -> None:
    """Show top 5 highest expenses using pandas nlargest."""
    header("🏆  Top 5 Expenses")
    if df.empty:
        print("  No data.")
        return
    top = df.nlargest(5, "amount")[["date", "category", "description", "amount"]]
    top["date"]   = top["date"].dt.strftime("%Y-%m-%d")
    top["amount"] = top["amount"].map("${:.2f}".format)
    print(top.to_string(index=False))

# ── Main Menu ─────────────────────────────────────────────────────────────────

def main():
    df = load_data()

    menu = """
  1. ➕  Add Expense
  2. 📋  View All Expenses
  3. 📊  Monthly Summary
  4. 🎯  Budget Check
  5. 💰  Set Budget
  6. 🗑️  Delete Expense
  7. 🏆  Top 5 Expenses
  0. 🚪  Exit
"""
    while True:
        header("💸  Smart Expense Tracker — Student Edition")
        print(menu)
        choice = input("  Choose option: ").strip()

        if   choice == "1": df = add_expense(df)
        elif choice == "2": view_expenses(df)
        elif choice == "3": monthly_summary(df)
        elif choice == "4": budget_check(df)
        elif choice == "5": set_budget()
        elif choice == "6": df = delete_expense(df)
        elif choice == "7": top_spending(df)
        elif choice == "0":
            print("\n  👋 Bye! Track smart, spend wise.\n")
            sys.exit(0)
        else:
            print("❌ Invalid option.")

        input("\n  Press Enter to continue...")


if __name__ == "__main__":
    main()

