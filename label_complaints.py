"""
Maps CFPB Issue column to PaymentCategory labels.
Outputs cfpb_labeled.parquet — used as labeled corpus + RAGAS test set.
"""

import pandas as pd

ISSUE_TO_CATEGORY = {
    # CHARGEBACK_DISPUTE
    "Problem with a purchase shown on your statement": "CHARGEBACK_DISPUTE",
    "Billing disputes": "CHARGEBACK_DISPUTE",

    # UNAUTHORIZED_TRANSACTION
    "Fraud or scam": "UNAUTHORIZED_TRANSACTION",
    "Unauthorized transactions or other transaction problem": "UNAUTHORIZED_TRANSACTION",
    "Identity theft / Fraud / Embezzlement": "UNAUTHORIZED_TRANSACTION",
    "Problem with a lender or other company charging your account": "UNAUTHORIZED_TRANSACTION",

    # ACCOUNT_BLOCKED
    "Closing an account": "ACCOUNT_BLOCKED",
    "Closing your account": "ACCOUNT_BLOCKED",
    "Closing/Cancelling account": "ACCOUNT_BLOCKED",
    "Problem getting a card or closing an account": "ACCOUNT_BLOCKED",

    # PAYMENT_FAILED
    "Problem when making payments": "PAYMENT_FAILED",
    "Money was not available when promised": "PAYMENT_FAILED",
    "Trouble using your card": "PAYMENT_FAILED",
    "Trouble using the card": "PAYMENT_FAILED",
    "Problem with a purchase or transfer": "PAYMENT_FAILED",

    # WRONG_DEDUCTION
    "Fees or interest": "WRONG_DEDUCTION",
    "Unexpected or other fees": "WRONG_DEDUCTION",
    "Wrong amount charged or received": "WRONG_DEDUCTION",
    "Problem caused by your funds being low": "WRONG_DEDUCTION",

    # SETTLEMENT_DELAY
    "Money transfers": "SETTLEMENT_DELAY",
    "Other transaction problem": "SETTLEMENT_DELAY",

    # REFUND_DELAYED
    "Problem with a company's investigation into an existing problem": "REFUND_DELAYED",
    "Problem with a credit reporting company's investigation into an existing problem": "REFUND_DELAYED",
}

# Sub-issue overrides — more specific, win over Issue mapping
SUBISSUE_TO_CATEGORY = {
    "Transaction was not authorized": "UNAUTHORIZED_TRANSACTION",
    "Card was charged for something you did not purchase with the card": "CHARGEBACK_DISPUTE",
    "Credit card company isn't resolving a dispute about a purchase on your statement": "CHARGEBACK_DISPUTE",
    "Card opened without my consent or knowledge": "UNAUTHORIZED_TRANSACTION",
    "Card opened as result of identity theft or fraud": "UNAUTHORIZED_TRANSACTION",
    "Funds not handled or disbursed as instructed": "SETTLEMENT_DELAY",
    "Funds not received from closed account": "SETTLEMENT_DELAY",
    "Problem during payment process": "PAYMENT_FAILED",
    "Problem with fees": "WRONG_DEDUCTION",
    "Overdrafts and overdraft fees": "WRONG_DEDUCTION",
    "Company closed your account": "ACCOUNT_BLOCKED",
    "Can't close your account": "ACCOUNT_BLOCKED",
}


def assign_label(row):
    sub = row.get("Sub-issue", "")
    issue = row.get("Issue", "")
    if isinstance(sub, str) and sub in SUBISSUE_TO_CATEGORY:
        return SUBISSUE_TO_CATEGORY[sub]
    if isinstance(issue, str) and issue in ISSUE_TO_CATEGORY:
        return ISSUE_TO_CATEGORY[issue]
    return None


if __name__ == "__main__":
    print("Loading parquet...")
    df = pd.read_parquet("cfpb_telecom.parquet")

    print("Assigning labels...")
    df["label"] = df.apply(assign_label, axis=1)

    labeled = df[df["label"].notna()].copy()
    unlabeled_count = df["label"].isna().sum()

    print(f"\nTotal rows       : {len(df):,}")
    print(f"Labeled rows     : {len(labeled):,}")
    print(f"Unlabeled (drop) : {unlabeled_count:,}")
    print(f"\nLabel distribution (pre-cap):")
    print(labeled["label"].value_counts())

    # Cap at 30K per category to prevent retrieval bias
    CAP = 30_000
    capped = pd.concat([
        group.sample(min(len(group), CAP), random_state=42)
        for _, group in labeled.groupby("label")
    ]).reset_index(drop=True)

    print(f"\nLabel distribution (post-cap at {CAP:,}):")
    print(capped["label"].value_counts())
    print(f"\nFinal corpus size: {len(capped):,}")

    capped.to_parquet("cfpb_labeled.parquet", index=False)
    print("\nSaved: cfpb_labeled.parquet")
