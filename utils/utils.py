
class Utils:


    @staticmethod
    def compara(a,b,p,q,k):
        if (a-b)>(p[k]):
            x="aPb"
        elif (a-b)>(q[k]):
            x="aQb"
        elif (a-b)>=0:
            x="aIb"
        elif (a-b)>=(-q[k]):
            x="bIa"
        elif (a-b)>=(-p[k]):
            x="bQa"
        elif (a-b)<(-p[k]):
            x="bPa"
        return x

    @staticmethod
    def media(pertinency: [int]):
        x = 0
        for i in range(len(pertinency)):
            if(pertinency[i]) != 0:
                x+=pertinency[i]
            tcam = (x/(len(pertinency)))
        return tcam

    @staticmethod
    def discordancias1(a,b,c,d, peso, cri):
        cont1=0
        soma1=0
        somat=0
        for i in range(cri):
            if not peso[i]==0:
                if b[i]=="aPb":
                    soma1+=peso[i]
                    somat+=peso[i]
                elif b[i]=="aQb":
                    somat+=peso[i]*c[i]
                elif b[i]=="aIb":
                    somat+=peso[i]*0.5
                elif b[i]=="bIa":
                    somat+=peso[i]*0.5
                    if abs(a[i])>=d[i]:
                        cont1+=1
                elif b[i]=="bQa":
                    somat+=peso[i]*c[i]
                    if abs(a[i])>=d[i]:
                        cont1+=1
                elif b[i]=="bPa":
                    somat+=peso[i]
                    if abs(a[i])>=d[i]:
                        cont1+=1
        if cont1>0:
            ms1=round(0.50,3)
        else:
            ms1=(soma1/somat)
        return ms1

    @staticmethod
    def discordancias1T2(a,b,c, d,p, q, peso, cri):
        cont1=0
        soma1=0
        somat=0
        for i in range(cri):
            if not peso[i]==0:
                if b[i]=="aPb":
                    soma1+=peso[i]*c[i]
                elif b[i]=="aQb":
                    somat+=abs(peso[i]*c[i]*(((((abs(a[i]))-q[i])/(p[i]-q[i]))*(0.5)+0.5)))
                elif b[i]=="aIb":
                    somat+=peso[i]*0.5*c[i]
                elif b[i]=="bIa":
                    somat+=peso[i]*0.5*c[i]
                    if abs(a[i])>=d[i]:
                        cont1+=1
                elif b[i]=="bQa":
                    somat+=abs(peso[i]*c[i]*(((((abs(a[i]))-q[i])/(p[i]-q[i]))*(0.5)+0.5)))
                    if abs(a[i])>=d[i]:
                        cont1+=1
                elif b[i]=="bPa":
                    somat+=peso[i]*c[i]
                    if abs(a[i])>=d[i]:
                        cont1+=1
        if cont1>0:
            ms1=round(0.50,3)
            return ms1
        else:
            ms1=(soma1/(somat+soma1))
            return ms1


    @staticmethod
    def discordancias2(a,b,c, d, peso, cri):
        cont2=0
        soma1=0
        somat=0
        for i in range(cri):
            if not peso[i]==0:
                if b[i]=="aPb":
                    soma1+=peso[i]
                    somat+=peso[i]
                elif b[i]=="aQb":
                    soma1+=peso[i]*c[i]
                    somat+=peso[i]*c[i]
                elif b[i]=="aIb":
                    somat+=peso[i]*0.5
                elif b[i]=="bIa":
                    somat+=peso[i]*0.5
                    if abs(a[i])>=d[i]:
                        cont2+=1
                elif b[i]=="bQa":
                    somat+=peso[i]*c[i]
                    if abs(a[i])>=d[i]:
                        cont2+=1
                elif b[i]=="bPa":
                    somat+=peso[i]
                    if abs(a[i]) >= d[i]:
                        cont2+=1
        if cont2>0:
            ms2=round(0.50,3)
        else:
            ms2=(soma1/somat)
        return ms2

    @staticmethod
    def discordancias2T2(a,b,c,d,p, q, peso, cri):
        cont2=0
        soma1=0
        somat=0
        for i in range(cri):
            if not peso[i]==0:
                if b[i]=="aPb":
                    soma1+=peso[i]*c[i]
                elif b[i]=="aQb":
                    soma1+=abs(peso[i]*c[i]*(((((abs(a[i]))-q[i])/(p[i]-q[i]))*(0.5)+0.5)))
                elif b[i]=="aIb":
                    somat+=peso[i]*0.5*c[i]
                elif b[i]=="bIa":
                    somat+=peso[i]*0.5*c[i]
                    if abs(a[i])>=d[i]:
                        cont2+=1
                elif b[i]=="bQa":
                    somat+=abs(peso[i]*c[i]*(((((abs(a[i]))-q[i])/(p[i]-q[i]))*(0.5)+0.5)))
                    if abs(a[i])>=d[i]:
                        cont2+=1
                elif b[i]=="bPa":
                    somat+=peso[i]*c[i]
                    if abs(a[i]) >= d[i]:
                        cont2+=1
        if cont2>0:
            ms2=round(0.50,3)
            return ms2
        else:
            ms2=(soma1/(somat+soma1))
            return ms2

    @staticmethod
    def discordancias3(a,b,c,d, peso, cri):
        cont3=0
        soma1=0
        somat=0
        for i in range(cri):
            if not peso[i]==0:
                if b[i]=="aPb":
                    soma1+=peso[i]
                    somat+=peso[i]
                elif b[i]=="aQb":
                    soma1+=peso[i]*c[i]
                    somat+=peso[i]*c[i]
                elif b[i]=="aIb":
                    soma1+=peso[i]*0.5
                    somat+=peso[i]*0.5
                elif b[i]=="bIa":
                    soma1+=peso[i]*0.5
                    somat+=peso[i]*0.5
                    if abs(a[i])>=d[i]:
                        cont3+=1
                elif b[i]=="bQa":
                    somat+=peso[i]*c[i]
                    if abs(a[i])>=d[i]:
                        cont3+=1
                elif b[i]=="bPa":
                    somat+=peso[i]
                    if abs(a[i])>=d[i]:
                        cont3+=1

        if cont3>0:
            ms3=round(0.50,3)
            return ms3
        else:
            ms3=(soma1/somat)
            return ms3

    @staticmethod
    def discordancias3T2(a,b,c,d,p, q, peso, cri):
        cont3=0
        soma1=0
        somat=0
        for i in range(cri):
            if not peso[i]==0:
                if b[i]=="aPb":
                    soma1+=peso[i]*c[i]
                elif b[i]=="aQb":
                    soma1+=abs(peso[i]*c[i]*(((((abs(a[i]))-q[i])/(p[i]-q[i]))*(0.5)+0.5)))
                elif b[i]=="aIb":
                    soma1+=peso[i]*0.5*c[i]
                elif b[i]=="bIa":
                    soma1+=peso[i]*0.5*c[i]
                    if abs(a[i])>=d[i]:
                        cont3+=1
                elif b[i]=="bQa":
                    somat+=abs(peso[i]*c[i]*((((abs(a[i]))-q[i])/(p[i]-q[i]))*(0.5)+0.5))
                    if abs(a[i])>=d[i]:
                        cont3+=1
                elif b[i]=="bPa":
                    somat+=peso[i]*c[i]
                    if abs(a[i])>=d[i]:
                        cont3+=1
        if cont3>0:
            ms3=round(0.50,3)
            return ms3
        else:
            ms3=(soma1/(somat+soma1))
            return ms3

    @staticmethod
    def s1(a,b,c, peso, cri):
        soma11=0
        soma21=0
        for i in range(cri):
            if a[i]=="aPb":
                soma11+=peso[i]
            elif a[i]=="aQb":
                soma21+=peso[i]*c[i]
            elif a[i]=="aIb":
                soma21+=peso[i]*0.5
            elif a[i]=="bIa":
                soma21+=peso[i]*0.5
            elif a[i]=="bQa":
                soma21+=peso[i]*c[i]
            elif a[i]=="bPa":
                soma21+=peso[i]
        if (soma11>soma21):
            return "domina"
        else:
            return "não domina"


    @staticmethod
    def s1T2(a,b,c, p,q,peso,cri):
        soma11=0
        soma21=0
        for i in range(cri):
            if a[i]=="aPb":
                soma11+=peso[i]*c[i]
            elif a[i]=="aQb":
                soma21+=abs((peso[i])*(c[i])*(((((abs(b[i]))-q[i])/(p[i]-q[i]))*(0.5)+0.5)))
            elif a[i]=="aIb":
                soma21+=peso[i]*0.5*c[i]
            elif a[i]=="bIa":
                soma21+=peso[i]*0.5*c[i]
            elif a[i]=="bQa":
                soma21+=abs((peso[i])*(c[i])*(((((abs(b[i]))-q[i])/(p[i]-q[i]))*(0.5)+0.5)))
            elif a[i]=="bPa":
                soma21+=peso[i]*c[i]
        if (soma11>soma21):
            return "domina"
        else:
            return "não domina"


    @staticmethod
    def s2T2(a,b,c,p,q,peso,cri):
        soma12=0
        soma22=0
        for i in range(cri):
            if a[i]=="aPb":
                soma12 += peso[i]*c[i]
            elif a[i]=="aQb":
                soma12+=abs(peso[i]*c[i]*(((((abs(b[i]))-q[i])/(p[i]-q[i]))*(0.5)+0.5)))
            elif a[i]=="aIb":
                soma22+=peso[i]*0.5*c[i]
            elif a[i]=="bIa":
                soma22+=peso[i]*0.5*c[i]
            elif a[i]=="bQa":
                soma22+=abs(peso[i]*c[i]*(((((abs(b[i]))-q[i])/(p[i]-q[i]))*(0.5)+0.5)))
            elif a[i]=="bPa":
                soma22+=peso[i]*c[i]
        if (soma12>soma22):
            return "domina"
        else:
            return "não domina"
    @staticmethod
    def s2(a,b,c, peso, cri):
        soma12=0
        soma22=0
        for i in range(cri):
            if a[i]=="aPb":
                soma12 += peso[i]
            elif a[i]=="aQb":
                soma12+=peso[i]*c[i]
            elif a[i]=="aIb":
                soma22+=peso[i]*0.5
            elif a[i]=="bIa":
                soma22+=peso[i]*0.5
            elif a[i]=="bQa":
                soma22+=peso[i]*c[i]
            elif a[i]=="bPa":
                soma22+=peso[i]
        if (soma12>soma22):
            return "domina"
        else:
          return "não domina"
  
    @staticmethod
    def s3(a,b,c, peso, cri):
        soma13=0
        soma23=0
        for i in range(cri):
            if a[i]=="aPb":
                soma13+=peso[i]
            elif a[i]=="aQb":
                soma13+=peso[i]*c[i]
            elif a[i]=="aIb":
                soma13+=peso[i]*0.5
            elif a[i]=="bIa":
                soma13+=peso[i]*0.5
            elif a[i]=="bQa":
                soma23+=peso[i]*c[i]
            elif a[i]=="bPa":
                soma23+=peso[i]
        if (soma13>soma23):
            return "domina"
        else:
            return "não domina"

    @staticmethod
    def s3T2(a,b,c,p,q,peso,cri):
        soma13=0
        soma23=0
        for i in range(cri):
            if a[i]=="aPb":
                soma13+=peso[i]*c[i]
            elif a[i]=="aQb":
                soma13+=abs(peso[i]*c[i]*(((((abs(b[i]))-q[i])/(p[i]-q[i]))*(0.5)+0.5)))
            elif a[i]=="aIb":
                soma13+=peso[i]*0.5*c[i]
            elif a[i]=="bIa":
                soma13+=peso[i]*0.5*c[i]
            elif a[i]=="bQa":
                soma23+=abs(peso[i]*c[i]*(((((abs(b[i]))-q[i])/(p[i]-q[i]))*(0.5)+0.5)))
            elif a[i]=="bPa":
                soma23+=peso[i]*c[i]

        if (soma13>soma23):
            return "domina"
        else:
            return "não domina"

    @staticmethod
    def ind(a,b,c):
        return float((a+b+c)/3)


    @staticmethod
    def dif(a,b):
        return a-b