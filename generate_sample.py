"""
Generates a realistic messy CSV with mixed dates, currency (Rs, ₹), and gender inconsistencies.
"""
import pandas as pd
import numpy as np

def create_messy_dataset(filename="sample_messy_data.csv"):
    data = {
        "Emp ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "Name": ["Alice S.", "Bob M.", "Charlie B.", "Diana P.", "Eve K.", "Frank T.", "Grace H.", "Harry P.", "Isha R.", "Jack D."],
        "Gender": ["f", "female", "MALE", "F", "m", "male", "Female", "M", "f", "man"],
        "Join Date": [
            "15/02/2023",    # DD/MM/YYYY
            "2023-01-15",    # YYYY-MM-DD
            "2023.03.10",    # YYYY.MM.DD
            "04-01-2023",    # MM-DD-YYYY
            "2023/05/20",    # YYYY/MM/DD
            "18-Jun-2023",   # DD-Mon-YYYY
            None,            # Missing
            "30/08/2023",    # DD/MM/YYYY
            "2023.09.12",    # YYYY.MM.DD
            "25-Dec-2023"    # DD-Mon-YYYY
        ],
        "Salary": ["Rs. 55,000", "₹ 62,000", "Rs 71000", "₹ 58,500", "Rs. 1,20,000", "85000", "₹ 64,000", "Rs 72000", "Rs. 95,000", "₹ 51,000"],
        "Bonus": ["₹ 5,000", "Rs. 2,000", "3500", "Rs 4,000", "₹ 15,000", "8000", "Rs. 1,500", "₹ 6,000", "Rs. 7,500", "₹ 3,000"],
        "City": [" Mumbai ", "Delhi", "bangalore", " Mumbai", "delhi", "Bangalore", "Pune", "pune", "Mumbai ", "DELHI"]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df

if __name__ == "__main__":
    create_messy_dataset()
    print("Created demo dataset with mixed dates, Rs/₹, and f/female!")