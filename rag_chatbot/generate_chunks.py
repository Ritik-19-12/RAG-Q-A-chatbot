import pandas as pd

df = pd.read_csv('../data/Training Dataset.csv')
df.fillna("Unknown", inplace=True)

chunks = []

for _, row in df.iterrows():
    text = f"""
    Applicant {row['Loan_ID']} is a {row['Gender']} {row['Married']} with education: {row['Education']}, self-employed: {row['Self_Employed']}.
    Income: {row['ApplicantIncome']}, Coapplicant: {row['CoapplicantIncome']}, Loan Amount: {row['LoanAmount']}, Loan Term: {row['Loan_Amount_Term']},
    Credit History: {row['Credit_History']}, Property Area: {row['Property_Area']}, Loan Status: {row['Loan_Status']}.
    """
    chunks.append(text.strip())

# Save to file
with open("../models/chunks.txt", "w", encoding='utf-8') as f:
    for chunk in chunks:
        f.write(chunk + "\n")
