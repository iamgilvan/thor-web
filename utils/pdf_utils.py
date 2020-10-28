from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        # Logo
        self.image('ime-logo.png', 16, 6, 20)
        # Arial bold 15
        self.set_font('Arial', '', 15)
        # Move to the right
        self.cell(80)
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', '', 6)
        self.multi_cell(0, 2, 'All rights reserved. The non-commercial (academic) use of this software is free of charge. The only thing that is asked in return is to cite this software when results are used in publications.', 0, 'C')
        self.multi_cell(0, 5, 'ALMEIDA, Gilvan Praxedes; TENÓRIO, Fabricio Maione; ARAUJO, Jean de Carvalho; GOMES, Carlos Francisco Simões; SANTOS, Marcos dos. THOR Web software (v.1), 2020.', 0, 'C')
        # Page number
        self.cell(0, 7, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')