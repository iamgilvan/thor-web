class Thor:
    def __init__(self):
        self.selected_method = None
        self.assignment_method_selected = None
        self.alternatives = []
        self.criterias = []
        self.decisors = []
        self.result = []
        self.pesofim = []
        self.peso =[]
        self.result_tca_s1 = []
        self.result_tca_s2 = []
        self.result_tca_s3 = []
        self.tca_s1_citerio_removed = ""
        self.tca_s2_citerio_removed = ""
        self.tca_s3_citerio_removed = ""

class Result:
    def __init__(self):
        self.S_result = []
        self.somatorio = []
        self.original = ""

class ResultTca:
    def __init__(self):
        self.title = ""
        self.subtitle = ""
        self.S_result = []
        self.somatorio = []
        self.original = ""
        self.original2 = ""