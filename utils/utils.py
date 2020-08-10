
class Utils:

    @staticmethod
    def create_initial_matrix(alternatives_number: int):
        matrix = []
        for alt in range(alternatives_number):
            line = [ 0 for column in range(alternatives_number)]
            matrix.append(line)
        return matrix


    @staticmethod
    def create_initial_weight(criteria_number: int):
        return [ 0 for i in range(criteria_number)]


    @staticmethod
    def avg(pertinency: [int]):
        x = 0
        for i in range(len(pertinency)):
            if(pertinency[i]) != 0:
                x+=pertinency[i]
            tcam = (x/(len(pertinency)))
        return tcam


    @staticmethod
    def comma_to_dot(a):
        return a.strip().replace(",",".")


    @staticmethod
    def first_disagreement(self,a,b,c, thor):
        control, partial_sum, total_sum = 0
        for i in range(thor.criteria):
            if not thor.weights[0][i]==0:
                if b[i]=="aPb":
                    if thor.selected_method == 1:
                        partial_sum += thor.weights[0][i]
                        total_sum += thor.weights[0][i]
                    else:
                        partial_sum += thor.weights[0][i]*c[i]
                elif b[i]=="aQb":
                    if thor.selected_method == 1:
                        total_sum += thor.weights[0][i]*c[i]
                    else:
                        total_sum +=abs(thor.weights[i]*c[i]*(((((abs(a[i]))-thor.q[i])/(thor.p[i]-thor.q[i]))*(0.5)+0.5)))
                elif b[i]=="aIb":
                    if thor.selected_method == 1:
                        total_sum += thor.weights[0][i]*0.5
                    else:
                        total_sum += thor.weights[0][i]*0.5*c[i]
                elif b[i]=="bIa":
                    if thor.selected_method == 1:
                        total_sum += thor.weights[0][i]*0.5
                    else:
                        total_sum +=thor.weights[0][i]*0.5*c[i]
                    if abs(a[i]) >= thor.disagreement[i]:
                        total_sum +=1
                elif b[i]=="bQa":
                    if thor.selected_method == 1:
                        total_sum += thor.weights[0][i]*c[i]
                    else:
                        total_sum+=abs(thor.weights[0][i]*c[i]*(((((abs(a[i]))-thor.q[i])/(thor.p[i]-thor.q[i]))*(0.5)+0.5)))
                    if abs(a[i])>= thor.disagreement[i]:
                        control +=1
                elif b[i]=="bPa":
                    if thor.selected_method == 1:
                        total_sum += thor.weights[0][i]
                    else:
                        total_sum += thor.weights[0][i]*c[i]
                    if abs(a[i])>=thor.disagreement[i]:
                        control+=1

        self.get_ms(control, partial_sum, total_sum, thor)


    @staticmethod
    def second_disagreement(self, a,b,c, thor):
        control, partial_sum, total_sum = 0
        for i in range(thor.criteria):
            if not thor.weights[1][i]==0:
                if b[i]=="aPb":
                    if thor.selected_method == 1:
                        partial_sum+=thor.weights[1][i]
                        total_sum+=thor.weights[1][i]
                    else:
                        partial_sum+=thor.weights[1][i]*c[i]
                elif b[i]=="aQb":
                    if thor.selected_method == 1:
                        partial_sum+=thor.weights[1][i]*c[i]
                        total_sum+=thor.weights[1][i]*c[i]
                    else:
                        partial_sum+=abs(thor.weights[1][i]*c[i]*(((((abs(a[i]))-thor.q[i])/(thor.p[i]-thor.q[i]))*(0.5)+0.5)))
                elif b[i]=="aIb":
                    if thor.selected_method == 1:
                        total_sum+=thor.weights[1][i]*0.5
                    else:
                        total_sum+=thor.weights[1][i]*0.5*c[i]
                elif b[i]=="bIa":
                    if thor.selected_method == 1:
                        total_sum+=thor.weights[1][i]*0.5
                    else:
                         total_sum+=thor.weights[1][i]*0.5*c[i]
                    if abs(a[i])>=thor.disagreement[i]:
                        control+=1
                elif b[i]=="bQa":
                    if thor.selected_method == 1:
                        total_sum+=thor.weights[1][i]*c[i]
                    else:
                        total_sum+=abs(thor.weights[1][i]*c[i]*(((((abs(a[i]))-thor.q[i])/(thor.p[i]-thor.q[i]))*(0.5)+0.5)))
                    if abs(a[i])>=thor.disagreement[i]:
                        control+=1
                elif b[i]=="bPa":
                    if thor.selected_method == 1:
                        total_sum+=thor.weights[1][i]
                    else:
                        total_sum+=thor.weights[1][i]*c[i]
                    if abs(a[i]) >= thor.disagreement[i]:
                        control+=1

        self.get_ms(control, partial_sum, total_sum, thor)


    @staticmethod
    def third_disagreement(self,a,b,c,thor):
        control, partial_sum, total_sum = 0
        for i in range(thor.criteria):
          if not thor.weights[2][i]==0:
            if b[i]=="aPb":
                if thor.selected_method == 1:
                    partial_sum+=thor.weights[2][i]
                    total_sum+=thor.weights[2][i]
                else:
                    partial_sum+=thor.weights[2][i]*c[i]
            elif b[i]=="aQb":
                if thor.selected_method == 1:
                    partial_sum+=thor.weights[2][i]*c[i]
                    total_sum+=thor.weights[2][i]*c[i]
                else:
                    partial_sum+=abs(thor.weights[2][i]*c[i]*(((((abs(a[i]))-thor.q[i])/(thor.p[i]-thor.q[i]))*(0.5)+0.5)))
            elif b[i]=="aIb":
                if thor.selected_method == 1:
                    partial_sum+=thor.weights[2][i]*0.5
                    total_sum+=thor.weights[2][i]*0.5
                else:
                    partial_sum+=thor.weights[2][i]*0.5*c[i]
            elif b[i]=="bIa":
                if thor.selected_method == 1:
                    partial_sum+=thor.weights[2][i]*0.5
                    total_sum+=thor.weights[2][i]*0.5
                else:
                    partial_sum+=thor.weights[2][i]*0.5*c[i]
                if abs(a[i])>=thor.disagreement[i]:
                    control+=1
            elif b[i]=="bQa":
                if thor.selected_method == 1:
                    total_sum+=thor.weights[2][i]*c[i]
                else:
                    total_sum+=abs(thor.weights[2][i]*c[i]*((((abs(a[i]))-thor.q[i])/(thor.p[i]-thor.q[i]))*(0.5)+0.5))
                if abs(a[i])>=thor.disagreement[i]:
                    control+=1
            elif b[i]=="bPa":
                if thor.selected_method == 1:
                    total_sum+=thor.weights[2][i]
                else:
                    total_sum+=thor.weights[2][i]*c[i]
                if abs(a[i])>=thor.disagreement[i]:
                    control+=1

        self.get_ms(control, partial_sum, total_sum, thor)


    def get_ms(control, partial_sum, total_sum, thor):
        if control > 0:
            thor.ms.append(round(0.50,3))
        else:
            thor.ms.append(partial_sum/total_sum if thor.selected_method == 1 else partial_sum/(total_sum+partial_sum))


    @staticmethod
    def check_dominance_thor_one(a,c, thor, index_weight):
        aPb_value, not_aPb_value = 0
        for i in range(thor.criteria):
            if a[i]=="aPb":
                aPb_value+=thor.weights[index_weight][i]
            elif a[i]=="aQb":
                not_aPb_value+=thor.weights[index_weight][i]*c[i]
            elif a[i]=="aIb":
                not_aPb_value+=thor.weights[index_weight][i]*0.5
            elif a[i]=="bIa":
                not_aPb_value+=thor.weights[index_weight][i]*0.5
            elif a[i]=="bQa":
                not_aPb_value+=thor.weights[index_weight][i]*c[i]
            elif a[i]=="bPa":
                not_aPb_value+=thor.weights[index_weight][i]

        return True if aPb_value > not_aPb_value else False

    @staticmethod
    def check_dominance_thor_two(a,b,c,thor, index_weight):
        aPb_value, not_aPb_value = 0
        for i in range(thor.criteria):
            if a[i]=="aPb":
                aPb_value+=thor.weights[index_weight][i]*c[i]
            elif a[i]=="aQb":
                not_aPb_value+=abs((thor.weights[index_weight][i])*(c[i])*(((((abs(b[i]))-thor.q[i])/(thor.p[i]-thor.q[i]))*(0.5)+0.5)))
            elif a[i]=="aIb":
                not_aPb_value+=thor.weights[index_weight][i]*0.5*c[i]
            elif a[i]=="bIa":
                not_aPb_value+=thor.weights[index_weight][i]*0.5*c[i]
            elif a[i]=="bQa":
                not_aPb_value+=abs((thor.weights[index_weight][i])*(c[i])*(((((abs(b[i]))-thor.q[i])/(thor.p[i]-thor.q[i]))*(0.5)+0.5)))
            elif a[i]=="bPa":
                not_aPb_value+=thor.weights[index_weight][i]*c[i]

        return True if aPb_value > not_aPb_value else False


    @staticmethod
    def indifference(a,b,c):
        return float((a+b+c)/3)


    @staticmethod
    def difference(a,b):
        return a-b