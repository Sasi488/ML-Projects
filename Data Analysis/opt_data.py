import pandas as pd
import os
from fpdf import FPDF

def analyze_and_generate_report(folder_path):
    # Collect all CSV and Excel files in the specified directory
    files = [f for f in os.listdir(folder_path) if f.endswith(('.csv', '.xlsx'))]
    
    if not files:
        message = "No CSV or Excel files found in the specified directory."
        print(message)
        save_to_pdf([message], "data_analysis_report.pdf")
        return

    report_content = []  # Store content for PDF

    for file in files:
        file_path = os.path.join(folder_path, file)

        # Read the data
        data = pd.read_csv(file_path) if file.endswith('.csv') else pd.read_excel(file_path)

        # Collecting content for both console and PDF
        report_content.append(f"\nAnalyzing {file}...")
        report_content.append(f"Shape: {data.shape}")

        report_content.append("\nColumn Data Types:")
        report_content.append(data.dtypes.to_string())

        report_content.append("\nStatistical Analysis (Numerical Columns):")
        report_content.append(data.describe().to_string())

        # Detect missing values
        missing_values = data.isnull().sum()
        total_missing = missing_values.sum()  # Total number of missing values

        report_content.append("\nMissing Values in Dataset:")
        if total_missing > 0:
            report_content.append(missing_values[missing_values > 0].to_string())
        else:
            report_content.append("No missing values found.")

        # Separate numerical, categorical, and mixed columns
        num_cols = data.select_dtypes(include='number').columns.tolist()
        cat_cols = data.select_dtypes(include='object').columns.tolist()

        report_content.append("\nDataset Division:")
        report_content.append(f"Numerical Columns: {num_cols}")
        report_content.append(f"Categorical Columns: {cat_cols}")

        # Suggested optimization
        if num_cols and cat_cols:
            suggestion = "Mixed dataset detected. Suggested optimization: Mixed methods like regression analysis and classification."
        elif num_cols:
            suggestion = "Numerical dataset detected. Suggested optimization: Regression methods like Linear or Logistic Regression."
        elif cat_cols:
            suggestion = "Categorical dataset detected. Suggested optimization: Classification methods like Decision Trees or Random Forests."
        else:
            suggestion = "The dataset does not contain valid data types for analysis."

        report_content.append(f"\n{suggestion}")

        # Generate equations for numerical columns
        report_content.append("\nGenerated Equations for Numerical Columns:")
        for col in num_cols:
            equation = f"y = m * {col} + b (Linear Regression)"
            reason = f"Reason: This equation suggests a linear relationship where 'y' is the predicted value and '{col}' is the independent variable."
            report_content.append(equation)
            report_content.append(reason)

        

    # Print report to console
    for line in report_content:
        print(line)

    # Save report to PDF
    save_to_pdf(report_content, "data_analysis_report.pdf")


def save_to_pdf(content, pdf_filename):
    """Saves the provided content to a PDF file."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in content:
        pdf.multi_cell(0, 10, line)

    pdf.output(pdf_filename)
    print(f"\nReport saved to {pdf_filename}")


if __name__ == "__main__":
    # Prompt the user to input the folder path containing the files
    folder_path = input("Enter the path to the folder containing CSV and Excel files: ")
    analyze_and_generate_report(folder_path)
