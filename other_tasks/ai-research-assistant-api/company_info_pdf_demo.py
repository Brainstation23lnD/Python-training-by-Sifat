from fpdf import FPDF

# Create instance of FPDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Title
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, 'Company XYZ - 12 Month Report', ln=True, align='C')

# Dummy data for 12 months
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
revenues = [12000, 15000, 17000, 16000, 18000, 19000, 20000, 21000, 19500, 22000, 23000, 25000]
expenses = [8000, 9000, 9500, 9200, 9700, 10000, 10200, 10500, 10100, 11000, 11500, 12000]
profit = [rev - exp for rev, exp in zip(revenues, expenses)]

# Table header
pdf.set_font("Arial", 'B', 12)
pdf.cell(40, 10, "Month", border=1)
pdf.cell(50, 10, "Revenue ($)", border=1)
pdf.cell(50, 10, "Expenses ($)", border=1)
pdf.cell(50, 10, "Profit ($)", border=1)
pdf.ln()

# Table content
pdf.set_font("Arial", '', 12)
for i in range(12):
    pdf.cell(40, 10, months[i], border=1)
    pdf.cell(50, 10, str(revenues[i]), border=1)
    pdf.cell(50, 10, str(expenses[i]), border=1)
    pdf.cell(50, 10, str(profit[i]), border=1)
    pdf.ln()

# Footer note
pdf.ln(10)
pdf.set_font("Arial", 'I', 10)
pdf.multi_cell(0, 10, "This is a demo PDF containing 12 months company financial information. You can import this PDF into your RAG system for querying data.")

# Save PDF
file_path = './company_12month_demo.pdf'
pdf.output(file_path)
file_path
