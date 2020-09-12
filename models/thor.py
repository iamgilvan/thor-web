class Thor:
    def __init__(self):
        self.selected_method = None
        self.assignment_method_selected = None
        self.alternatives = []
        self.criterias = []
        self.decisors = []
        self.result = []
        self.pesofim = []
        self.peso = []
        self.pesom = []
        self.pesomList = []
        self.result_tca_s1 = []
        self.result_tca_s2 = []
        self.result_tca_s3 = []
        self.tca_s1_citerio_removed = ""
        self.tca_s2_citerio_removed = ""
        self.tca_s3_citerio_removed = ""
        self.removedTcaN = ""
        self.result_tca_n = []
        self.usartca = False
        self.questionObj = []
        self.indexDecisor = 0
        self.indexCri1 = 0
        self.indexCriMarc = 0
        self.marc = 2

class Result:
    def __init__(self):
        self.S_result = []
        self.somatorio = []
        self.original = ""

class ResultTca:
    def __init__(self):
        self.title = ""
        self.sub_title = ""
        self.S_result = []
        self.somatorio = []
        self.original = ""
        self.original2 = ""

class TcaNebulosa:
    def __init__(self):
        self.title =""
        self.head = []
        self.input_rows = []
        self.medias = []
        self.mediaalt = []
        self.mediamedias =[]
        self.removed = ""

class Question:
    def __init__(self):
        self.question = ""
        self.questionB = ""
        self.min = 0
        self.max = 0
        self.position = 0
        self.decisor = 1