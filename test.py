from fpdf import FPDF
def generate_pdf():
    # Sample array data
    data = [['Name', 'Age', 'City'],
            ['John', 25, 'New York'],
            ['Mary', 30, 'London'],
            ['Bob', 20, 'Paris'],['kadavule haish eh', 20, 'covai na gethu']]
    # Create a PDF object
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Set column widths
    col_widths = [40, 60, 40]
    # Create table header
    for i, col in enumerate(data[0]):
        pdf.cell(col_widths[i], 10, col, 1)
    pdf.ln()
    # Create table rows
    for row in data[1:]:
        for i, col in enumerate(row):
            pdf.cell(col_widths[i], 10, str(col), 1)
        pdf.ln()
    # Save the PDF to a file
    pdf.output('table.pdf')
    
generate_pdf()