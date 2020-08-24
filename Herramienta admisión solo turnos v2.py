# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 08:55:06 2019

@author: jgonzalezh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:22:36 2019

@author: jgonzalezh
"""

import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from pulp import *
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import random 
import datetime
import math

start=time.process_time()

def data(tabla):
    #Función para leer la data, pregunta a nuestro servidor de capacity por el nombre de la tabla que se solicite
    conn = pyodbc.connect("Driver={SQL Server};"
                      "Server=ACHS-ANALYTICAZ.achs.cl;"
                      "Database=az-capacity;"
                      "Trusted_Connection=yes;")
    sql= "Select * from " + tabla +" where Centro!='Policlínico Especialidades Concepción'"
    #la consulta se hace a través de un select
    data=pd.read_sql(sql, conn)
    #guarda el archivo como una variable de python
    return data #devuelve la variable
def data2(tabla):
    #Función para leer la data, pregunta a nuestro servidor de capacity por el nombre de la tabla que se solicite
    conn = pyodbc.connect("Driver={SQL Server};"
                      "Server=ACHS-ANALYTICAZ.achs.cl;"
                      "Database=az-capacity;"
                      "Trusted_Connection=yes;")
    sql= "Select * from " + tabla
    #la consulta se hace a través de un select
    data=pd.read_sql(sql, conn)
    #guarda el archivo como una variable de python
    return data #devuelve la variable



#if data_dmd.iat[][]
class Herramienta:
    def __init__(self,data_dmd, data_turnos):
        #lecturas tablas servidor
        plazo='6 meses'
        self.data=data_dmd 
        #Variables auxiliares
        self.Centros=data_dmd.Centro.unique()
        fil=len(self.data.index) 
        self.N_CE=len(data_dmd.Centro.unique())
        n=7
        g=2
        m=48
        dias=['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo'] 
        Tipo=['Espontaneo', 'Controles']
        horas=['00:00:00.0000000','00:30:00.0000000','01:00:00.0000000','01:30:00.0000000','02:00:00.0000000','02:30:00.0000000','03:00:00.0000000','03:30:00.0000000',
            '04:00:00.0000000','04:30:00.0000000','05:00:00.0000000','05:30:00.0000000','06:00:00.0000000','06:30:00.0000000','07:00:00.0000000','07:30:00.0000000',
            '08:00:00.0000000','08:30:00.0000000','09:00:00.0000000','09:30:00.0000000','10:00:00.0000000','10:30:00.0000000','11:00:00.0000000','11:30:00.0000000',
            '12:00:00.0000000','12:30:00.0000000','13:00:00.0000000','13:30:00.0000000','14:00:00.0000000','14:30:00.0000000','15:00:00.0000000','15:30:00.0000000',
            '16:00:00.0000000','16:30:00.0000000','17:00:00.0000000','17:30:00.0000000','18:00:00.0000000','18:30:00.0000000','19:00:00.0000000','19:30:00.0000000',
            '20:00:00.0000000','20:30:00.0000000','21:00:00.0000000','21:30:00.0000000','22:00:00.0000000','22:30:00.0000000','23:00:00.0000000','23:30:00.0000000']
        #Variables de demanda variables y programada, D[centro][0=espontaneo;1=controles][dia][bloque]
        self.D=[[[[0 for t in range(m)] for i in range(n)]for e in range(g) ]for c in range(self.N_CE)]
        #self.Q=[[[[0 for t in range(m)] for i in range(n)]for e in range(g) ]for c in range(self.N_CE)]
        #llenar variables con la demanda
        for i in range(len(data_dmd.index)):
            if data_dmd.iat[i,15]==plazo:
                for c in range(self.N_CE):
                    if data_dmd.iat[i,0]==self.Centros[c]:
                        for d in range(7):
                            if data_dmd.iat[i,1]==dias[d]:
                                for h in range(48):
                                    if data_dmd.iat[i,2]==horas[h]:
                                        #self.Q[c][0][d][h]+=data_dmd.iat[i,5]
                                        #self.Q[c][1][d][h]+=data_dmd.iat[i,6]
                                        self.D[c][0][d][h]+=data_dmd.iat[i,14]/30
                                        #self.D[c][1][d][h]+=data_dmd.iat[i,4]/30  
        #llenar las variables con los turnos
        FTE=1
        self.turno=data_turnos 
        turnos=self.turno
        fill=len(turnos.index)
        turnos['Hora Inicio']=turnos['Hora inicio Turno'].dt.hour
        turnos['Hora Inicio']=turnos['Hora inicio Turno'].dt.hour
        turnos['Minutos Inicio']=turnos['Hora inicio Turno'].dt.minute
        turnos['Hora Fin']=turnos['Hora Fin Turno'].dt.hour
        turnos['Minutos Fin']=turnos['Hora Fin Turno'].dt.minute
        turnos['Hora Colacion In']=turnos['Hora Inicio Colación'].dt.hour
        turnos['Minutos colacion In']=turnos['Hora Inicio Colación'].dt.minute
        turnos['Hora Colacion fin']=turnos['Hora Fin Colación'].dt.hour
        turnos['Minutos colacion fin']=turnos['Hora Fin Colación'].dt.minute
        self.T=[[[0 for i in range(48)]for j in range(7)]for c in range(self.N_CE)]
        
        for g in range(fill):
            for c in  range(self.N_CE):
                if turnos.iat[g,0]==self.Centros[c]:
                    for h in range(7):
                        if (turnos.iat[g,11]+turnos.iat[g,12]/60)!=(turnos.iat[g,13]+ turnos.iat[g,14]/60):                        
                            if turnos.iat[g,2]==dias[h]:
                               for j in range(int(round(2*turnos.iat[g,9]+turnos.iat[g,10]/30))):
                                #for j in range(int(2*turnos.iat[g,11]+turnos.iat[g,12]/30)):
                                    if j>=int(round(2*turnos.iat[g,7]+turnos.iat[g,8]/30)) and j<int(round(2*turnos.iat[g,11]+turnos.iat[g,13]/30)): #revisa si tiene que ponerle -1
                                        self.T[c][h][j]+=FTE
                                    elif j>=int(round(2*turnos.iat[g,13]+turnos.iat[g,14]/30)) and j<=int(round(2*turnos.iat[g,9]+(turnos.iat[g,10]/30))):
                                        self.T[c][h][j]+=FTE
                        else:
                            if turnos.iat[g,2]==dias[h]:
                                for j in range(int(round(2*turnos.iat[g,9]+turnos.iat[g,10]/30))):
                                    if j>=int(round(2*turnos.iat[g,7]+turnos.iat[g,8]/30)) and j<=int(round(2*turnos.iat[g,9]+(turnos.iat[g,10]/30))): #revisa si tiene que ponerle -1
                                        self.T[c][h][j]+=FTE
    def plot(self,c):
        # Graficar
        witdh=0.7
        #largo de las barras
        ## Posible utilidad posterior: label2=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','45','46','47','48']
        label=['00:00',' ','01:00','','02:00',' ','03:00',' ',
        '04:00',' ','05:00',' ','06:00',' ','07:00',' ',
        '08:00',' ','09:00',' ','10:00',' ','11:00',' ',
        '12:00',' ','13:00',' ','14:00',' ','15:00',' ',
        '16:00',' ','17:00',' ','18:00',' ','19:00',' ',
        '20:00',' ','21:00',' ','22:00',' ','23:00',' ']
        plt.style.use('default')
        #se agregan el label del eje x
        index = np.arange(len(label)) # array([0,1,2,...,47])
        #lista del largo del label
        #plt.close()  
        Z=plt.figure() 
        Z.set_facecolor('lightgoldenrodyellow')
        #se abre el plot antes (al pnerlo ah{i me arreglo un problema de ploteo})
        dias=['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
        #Días de la semana para títulos de los diferentes graficos
        for i in range(7):
            #Iteración para gráficar lso 7 días de la semana
            #x=2*i
            #=2*i+1#,n in enumerate(7):
            Esp=self.D[c][0][i][:] #Demanda espontanea del d{ia de la semana i}
            Cit1=self.D[c][1][i][:] #Demanda programada de día i 
            plt.tight_layout()
            plt.subplot(421+i) #se grafica en un marco de 2 (horizontal) x 4 vertical, posición 1+i (tiene 8 posiciones)
            p1=plt.bar(index, np.array(Esp), witdh, color='darkkhaki')#'mediumaquamarine') #'seagreen')#Gráfico de barras de espontaneos
            p2=plt.bar(index, np.array(Cit1), witdh, bottom=np.array(Esp), color='seagreen')
            #p3=plt.bar(index, np.array(Cit2), witdh, bottom=np.array(Cit1)+np.array(Esp),color='lightskyblue')## yerr=DEsp,color='lightskyblue',capsize=2 , ecolor='darkslateblue')
            #'lawngreen')#'mediumseagreen')
            #p2=plt.bar(index, Cit, witdh, bottom=Esp, yerr=DEsp,color='mediumseagreen' ,capsize=2 , ecolor='darkslateblue') #grafico de barra, agregar citados sobre espontaneos
            turnos=self.T[c][i][:]
            T2=[[0 for t in range(48)]for d in range(7)]
            for j in range(48):
               T2[i][j]=self.T[c][i][j]*0.83
            #Turnos por trabajar
            p5=plt.plot(index, turnos,'mediumaquamarine')#'salmon') #se gráfican 
            p6=plt.plot(index, T2[i][:] ,linestyle=':',color='mediumaquamarine')#'salmon') #se gráfican 
            #print('Grafico') #mostrar el minuto en que se graficaban todos
            plt.title(dias[i])
            #Se agrega el título por gráfico
            plt.subplots_adjust(hspace=0.4)
            #Se define el espacio entre gráficos
            plt.ylabel('Demanda', fontsize=12)
            #Label eje y
            plt.xticks(index, label, fontsize=6, rotation=90)
            #se agregan las horas del día para el eje x, fuente tamaño 5 y rotación 90 grados
            plt.suptitle(self.Centros[c], fontsize=14)
            plt.legend((p1[0],p2[0],p5[0],p6[0]), ('Espontaneo','Citados','Turnos','Utilización meta') ,loc='upper left',prop={'size':8})# 
                            #Se agrega loa leyenda para cada una de las variables
            #posición de la leyenda
            #plt.grid()
        plt.show()
    def plot_opt(self,Y,dictio,c):
        # Graficar
        witdh=0.7
        #largo de las barras
        ## Posible utilidad posterior: label2=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','45','46','47','48']
        label=['00:00',' ','01:00',' ','02:00',' ','03:00',' ',
        '04:00',' ','05:00',' ','06:00',' ','07:00',' ',
        '08:00',' ','09:00',' ','10:00',' ','11:00',' ',
        '12:00',' ','13:00',' ','14:00',' ','15:00',' ',
        '16:00',' ','17:00',' ','18:00',' ','19:00',' ',
        '20:00',' ','21:00',' ','22:00',' ','23:00',' ']
        dias2=[0,1,2,3,4,5]
        plt.style.use('default')
        #se agregan el label del eje x
        index = np.arange(len(label)) # array([0,1,2,...,47])
        #lista del largo del label
        #plt.close()  
        Z=plt.figure() 
        Z.set_facecolor('lightgoldenrodyellow')
        #se abre el plot antes (al pnerlo ah{i me arreglo un problema de ploteo})
        dias=['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
        #Días de la semana para títulos de los diferentes graficos
        almuerzos=[26,27,28,29,30,26,27,28,29,30,25]
        AP_AL=[[0 for t in range(48)] for i in range(len(almuerzos))]
    
        for i in range(len(almuerzos)):
            AP_AL[i][almuerzos[i]]=1 #marcar con uno el bloque del almuerzo de la persona en una matriz comparable 
        
        Tur=[[0 for t in range(48)]for i in dias]
        
        for i in dias2:
            Esp=self.D[c][0][i][:] #Demanda espontanea del d{ia de la semana i}
            Cit1=self.D[c][1][i][:] #Demanda programada de día i 
            """
            #Iteración para gráficar lso 7 días de la semana
            #x=2*i
            #=2*i+1#,n in enumerate(7):
            Esp=self.D[c][0][i][:] #Demanda espontanea del dia de la semana i
            Cit1=self.D[c][1][i][:] #Demanda programada de día i 
            for j in range(48):
                for m in range(9):
                    #if type(TO[(i,j,m)].varValue)!=type(None):
                    Tur[i][j]+=TO[(i,j,m)].varValue
                #else:
                   # Tur[i][j]=+0"""
            for m in range(10): 
                for k in range(len(dictio)):
                    Tur[i][:]+=Y[(m,k,i)].varValue*(np.array(dictio[k][:]-np.array(AP_AL[m])))
                    #print(Y[(m,k,i)].varValue)
                    #print(np.array(dictio[k][:]))
                    #print(np.array(AP_AL[m]))
                    #print(np.array(dictio[k][:]-np.array(AP_AL[m])))
            #print(Tur[i][:], 'dia', i)        
            #print(type(Tur[i][:]))
            plt.tight_layout()
            
            plt.subplot(421+i) #se grafica en un marco de 2 (horizontal) x 4 vertical, posición 1+i (tiene 8 posiciones)
            p1=plt.bar(index, np.array(Esp), witdh, color='darkkhaki')#'mediumaquamarine') #'seagreen')#Gráfico de barras de espontaneos
            p2=plt.bar(index, np.array(Cit1), witdh, bottom=np.array(Esp), color='seagreen')
            p5=plt.plot(index, Tur[i][:],'mediumaquamarine')
            #p3=plt.bar(index, np.array(Cit2), witdh, bottom=np.array(Cit1)+np.array(Esp),color='lightskyblue')## yerr=DEsp,color='lightskyblue',capsize=2 , ecolor='darkslateblue')
            #'lawngreen')#'mediumseagreen')
            #p2=plt.bar(index, Cit, witdh, bottom=Esp, yerr=DEsp,color='mediumseagreen' ,capsize=2 , ecolor='darkslateblue') #grafico de barra, agregar citados sobre espontaneos
            #turnos=self.T[c][i][:]
            T2=[[0 for t in range(48)]for d in dias]
            for j in range(48):
               T2[i][j]=Tur[i][j]*0.83
            #Turnos por trabajar
            p5=plt.plot(index, Tur[i][:],'mediumaquamarine')#'salmon') #se gráfican 
            p6=plt.plot(index, T2[i][:] ,linestyle=':',color='mediumaquamarine')#'salmon') #se gráfican 
            #print('Grafico') #mostrar el minuto en que se graficaban todos
            plt.title(dias[i])
            #Se agrega el título por gráfico
            plt.subplots_adjust(hspace=0.4)
            #Se define el espacio entre gráficos
            plt.ylabel('Demanda', fontsize=12)
            #Label eje y
            plt.xticks(index, label, fontsize=6, rotation=90)
            #se agregan las horas del día para el eje x, fuente tamaño 5 y rotación 90 grados
            plt.suptitle(self.Centros[c]+" Optimización Turnos", fontsize=14)
            plt.legend((p1[0],p2[0],p5[0],p6[0]), ('Espontaneo','Citados','Turnos','Utilización meta') ,loc='upper left',prop={'size':8}) #,p6[0] ,'Utilización meta'
                            #Se agrega loa leyenda para cada una de las variables
            #posición de la leyenda
            #plt.grid()
        plt.show()
"""        
diccionario_T=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0], #turno de 8 a 17
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0], #turno de 8:30 a 17:30
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0], #turno de 9 a 18:00
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0], #turno de 11:00 a 20:00
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0], #turno de 11:30 a 20:30
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#turno de 9:30 a 13:30
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0], #14:00 a 18:00
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #8:00-12:00
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #8:30-12_30
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #9:00-13:00
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  #9:00-15:00
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0], #12-18
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #12-16
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]] #16:20
almuerzos=[26,25,27,24,28,26,25,27,24]
AP_AL=[[0 for t in range(48)] for i in range(len(almuerzos))]

for i in range(len(almuerzos)):
    AP_AL[i][almuerzos[i]]=1 #marcar con uno el bloque del almuerzo de la persona en una matriz comparable 
k=2
m=1
np.array(diccionario_T[k][:])-np.array(AP_AL[m])
"""


def suma(M):
    x=[0,0,0,0,0,0,0]
    for j in range(7):
        for i in range(48): 
            x[j]+=M[1][j][i]
    return x

def sumaobjetivo(D,Y,c,F,P,turn,AP_AL):
    K=len(turn)
    medicos=[0,1,2,3,4,5,6,7,8,9,10]
    dias=[0,1,2,3,4,5]    
    
    x=pulp.lpSum([(100*m*m)*Y[(m,k,d)]for k in range(K) for m in medicos for d in dias])+pulp.lpSum([10*Y[(m,k,5)]for k in range(K) for m in medicos])#+pulp.lpSum([((turn[k][i]-AP_AL[m][i]*turn[k][i])*P*Y[(m,k,d)]-D[c][0][d][i]-D[c][1][d][i])for i in range(48) for d in dias for k in range(K) for m in medicos])#+pulp.lpSum([1*((turn[k][i]-AP_AL[m][i]*turn[k][i])*P*Y[(m,k,d)]-1)*-D[c][0][d][i] for i in range(48) for d in range(5)for k in range(K) for m in medicos])
    #Se castiga el número de médicos, pero para que no se alarguen turnos innecesariamente) en la mañana, se castiga la demanda insatisfecha (no agregaste la hora de almuerzo aca)
    """ (agregar por separado a fn X*tur en lpSum, deberías castigar el horario extendido"""        #se prioriza holgura (puede servir para agendar controles,                                                                        #AP_AL[m][j]*diccionario_T[k][j] esto sirve para restar los almuerzos
    
    
    #pulp.lpSum([pulp.lpSum([TO[(d,i,m)]for m in range(9)])-D[c][0][d][i]-D[c][1][d][i] for d in range(5) for i in range(48)])+pulp.lpSum([(100+m+m*m*m)*TO[(d,i,m)]for m in range(9) for d in range(5) for i in range(48)])# + Aux[(d,m)]
    #x=pulp.lpSum([-D[c][0][d][i]-D[c][1][d][i] for i in range(48) for d in range(5)])*10+pulp.lpSum([TO[(d,i,m)]for m in range(9) for d in range(5) for i in range(48)])*10
    #+pulp.lpSum([(100+m+m*m*m)*TO[(d,i,m)]for m in range(9)])
    #x=pulp.lpSum([(C[(d,i)]-P*F*T[c][d][i]+F*D[c][0][d][i])*100+i*C[(d,i)]for i in range(48) for d in range(5)])
    """
    for i in range(48):
        for d in range(5):
            x+=(C[(d,i)]-P*F*T[c][d][i]+F*D[c][1][d][i])*100+i*C[(d,i)]
    """

    return x
    
def optimizacion(clase_dmd,c):
    data_horarios=data2("DI_HorariosCentros_AP_JGH")
    medicos=[0,1,2,3,4,5,6,7,8,9,10]
    largo=len(data_horarios.Comuna)
    bloque_inicio=17
    bloque_fin=35
    for g in range(largo):
        if clase_dmd.Centros[c]==data_horarios.iat[g,1] and data_horarios.iat[g,2] is not None and data_horarios.iat[g,3] is not None :
            inicio_str = data_horarios.iat[g,2][:-1]
            inicio_obj = datetime.datetime.strptime(inicio_str, "%H:%M:%S.%f")
            fin_str = data_horarios.iat[g,3][:-1]
            fin_obj = datetime.datetime.strptime(fin_str, "%H:%M:%S.%f")
            inicio=inicio_obj.hour*2
            fin= (fin_obj.hour*2)-1 # El último bloque termina en la media hora previa (se agrupa todo lo de 17_18:00 en el bloque 17:30)
            if inicio_obj.minute==30:
                inicio+=1
            if fin_obj.minute==30:
                fin+=1
            bloque_inicio=inicio
            bloque_fin=fin 
    dias2=['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
    dias=[0,1,2,3,4,5]
    #tipo=[0,1]
    index = np.arange(48)
    modelo=LpProblem('Controles y turnos', LpMinimize)
    D=clase_dmd.D
    #T=clase_dmd.T
    #Q=clase_dmd.Q 
    F=2 #Controles que caben en un bloque 
    P=0.83 #Productividad
    x=list(range(48))
    diccionario_T=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #turno de 8 a 17
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0], #turno de 8:30 a 17:30
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0], #turno de 9 a 18:00
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0], #turno de 9:30 a 18:30
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0], #turno de 10:00 a 19:00
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0], #turno de 10:30 a 19:30 
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0], #turno de 11:00 a 20:00
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],#, #turno de 11:30 a 20:30
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # 8: 16 para sabado
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] #
            #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #turno de 9:30 a 13:30
            #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0], #14:00 a 18:00
            #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #8:00-12:00
            #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #8:30-12_30
            #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #9:00-13:00
            #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #9:00-15:00
            #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]] #12-18
            #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #12-16
            #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #11-15
            #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #10-14
            #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]] #16-20
    #len(diccionario_T[0][:])
    #len(np.array(diccionario_T[0][:])+np.array(diccionario_T[1][:]))
    K=len(diccionario_T)
    
    #C=LpVariable.dicts('Controles propuestos', [(j,i)for i in index for j in dias],lowBound=0, upBound=None,cat='Integer')
    #TO=LpVariable.dicts('Turnos propuestos', [(j,i,m)for i in index for j in dias for m in medicos],lowBound=0, upBound=1,cat='Binary')
    #Aux=LpVariable.dicts('Auxiliar jornada', [(j,m) for j in dias for m in medicos],cat='Binary')
    Y=LpVariable.dicts('médicos y turnos', [(m,k,d) for k in range(K) for m in medicos for d in dias],lowBound=0, upBound=None,cat='Binary')
    
    almuerzos=[26,27,28,29,30,26,27,28,29,30,25]
    AP_AL=[[0 for t in range(48)] for i in range(len(almuerzos))]

    for i in range(len(almuerzos)):
        AP_AL[i][almuerzos[i]]=1 #marcar con uno el bloque del almuerzo de la persona en una matriz comparable 
        
    modelo+=sumaobjetivo(D,Y,c,F,P,diccionario_T,AP_AL)    
    for m in medicos:
        modelo+=pulp.lpSum([Y[(m,k,d)]*(diccionario_T[k][j]) for k in range(K) for d in dias for j in range(48)])<=90#pulp.lpSum([Y[(m,k,d)]for k in range(K) for d in dias]) #no trabaja más de 90 horas
        #modelo+=pulp.lpSum([Y[(m,k,d)]*(diccionario_T[k][j]) for k in range(K) for d in dias for j in range(48)])>=round(pulp.lpSum([Y[(m,k,d)]for k in range(K) for d in dias])/6,0)*90 #se redondea a uno cuando trabaja, es decir que cuando trabaja tiene que tener 90 medias horas a la semana
        #modelo+=pulp.lpSum([Y[(m,k,d)]for k in range(K) for d in dias])>=pulp.lpSum([Y[(m,k,0)]for k in range(K)])*5 #si trabaja el lunes trabaja al menos 5 días, la idea es que la restricción de arriba no se acerque a cero
        #modelo+=pulp.lpSum([Y[(m,k,d)]for k in range(K) for d in dias])>=pulp.lpSum([Y[(m,k,1)]for k in range(K)])*5 #idem resto de los días de la semana
        #modelo+=pulp.lpSum([Y[(m,k,d)]for k in range(K) for d in dias])>=pulp.lpSum([Y[(m,k,2)]for k in range(K)])*5
        #modelo+=pulp.lpSum([Y[(m,k,d)]for k in range(K) for d in dias])>=pulp.lpSum([Y[(m,k,3)]for k in range(K)])*5
        #modelo+=pulp.lpSum([Y[(m,k,d)]for k in range(K) for d in dias])>=pulp.lpSum([Y[(m,k,4)]for k in range(K)])*5
        modelo+=pulp.lpSum([Y[(m,k,0)]for k in range(K)])*90<=pulp.lpSum([Y[(m,k,d)]*(diccionario_T[k][j]) for k in range(K) for d in dias for j in range(48)]) #si trabaja lunes suma al menos 90h en la semana
        modelo+=pulp.lpSum([Y[(m,k,1)]for k in range(K)])*90<=pulp.lpSum([Y[(m,k,d)]*(diccionario_T[k][j]) for k in range(K) for d in dias for j in range(48)]) #idem otros dias
        modelo+=pulp.lpSum([Y[(m,k,2)]for k in range(K)])*90<=pulp.lpSum([Y[(m,k,d)]*(diccionario_T[k][j]) for k in range(K) for d in dias for j in range(48)])
        modelo+=pulp.lpSum([Y[(m,k,3)]for k in range(K)])*90<=pulp.lpSum([Y[(m,k,d)]*(diccionario_T[k][j]) for k in range(K) for d in dias for j in range(48)])
        modelo+=pulp.lpSum([Y[(m,k,4)]for k in range(K)])*90<=pulp.lpSum([Y[(m,k,d)]*(diccionario_T[k][j]) for k in range(K) for d in dias for j in range(48)])
        modelo+=pulp.lpSum([Y[(m,k,5)]for k in range(K)])*90<=pulp.lpSum([Y[(m,k,d)]*(diccionario_T[k][j]) for k in range(K) for d in dias for j in range(48)])
        modelo+=pulp.lpSum([Y[(m,k,5)]for k in range(K)])*80<=pulp.lpSum([Y[(m,k,d)]*(diccionario_T[k][j]) for k in range(K) for d in [0,1,2,3,4] for j in range(48)]) # si trabaja sabado en la semana suma al menos 80 horas # no creo que sea lo mejor ponerla 
        
        #falta una si trabaja, trabaja 45, para que no tome el turno corto en la semana
    for d in dias:
        modelo+=pulp.lpSum([D[c][0][d][j]+D[c][1][d][j] for j in index] )<=pulp.lpSum([(np.array(diccionario_T[k][j])-np.array(AP_AL[m][j]*diccionario_T[k][j]))*Y[(m,k,d)] for j in index for m in medicos for k in range(K)])
        #tener capacidad en el día para cumplir con toda la demanda
        for m in medicos:
            modelo+=pulp.lpSum([Y[(m,k,d)] for k in range(K)])<=1 #cada doctor tiene un pool de turno al dia maximo
            for i in range(0,bloque_inicio):    
                modelo+=pulp.lpSum([Y[m,k,d]*diccionario_T[k][i] for  k in range(K)])<=0 #no se tiene turno hasta después del cierre del centro
            for i in range(bloque_fin+1,48):    
                modelo+=pulp.lpSum([Y[m,k,d]*diccionario_T[k][i] for  k in range(K)])<=0 #no se tiene turno hasta después del cierre del centro
                            
            for k in range(K):
                modelo+=pulp.lpSum([diccionario_T[k][i]*Y[(m,k,d)] for i in range(48)])<=18 #turno maximo de 18 periodos de 30 min #creo q es innecesaria
        for i in range(bloque_inicio,bloque_fin):
             #modelo+=pulp.lpSum([D[c][0][d][j]+D[c][1][d][j] for j in [i-2,i-1,i,i+1,i+2]] )<=pulp.lpSum([(np.array(diccionario_T[k][j])-np.array(AP_AL[m][j]*diccionario_T[k][j]))*Y[(m,k,d)] for j in [i-2,i-1,i,i+1,i+2] for m in medicos for k in range(K)])
             modelo+=pulp.lpSum([D[c][0][d][j] for j in [i-2,i-1,i,i+1,i+2]] )*0.75<=pulp.lpSum([(np.array(diccionario_T[k][j])-np.array(AP_AL[m][j]*diccionario_T[k][j]))*Y[(m,k,d)] for j in [i-2,i-1,i,i+1,i+2] for m in medicos for k in range(K)])
             #el turno cumple con al menos dla demanda espontanea (dado que los controles se pueden reagendar esta restriccioon cumple con la funcion de tener cobertura por vencidad de la demanda estocastica)
    for i in range(0,16): 
        for m in medicos:
            modelo+=pulp.lpSum([Y[m,k,5]*diccionario_T[k][i] for  k in range(K)])<=0 #no se tiene turno hasta después del cierre del centro
    for i in range(30,48):
        for m in medicos:
            modelo+=pulp.lpSum([Y[m,k,5]*diccionario_T[k][i] for  k in range(K)])<=0 #no se tiene turno hasta después del cierre del centro
                            
    
    modelo.solve() #SIEMPRE ANTES PARA QUE NO SEAN NULL LAS VARIABLES
    print('\n'+clase_dmd.Centros[c]," ",c," ",LpStatus[modelo.status]) #escribe centro y estatus de la optimizacion 
    T3=[[0 for i in range(48)]for d in dias] #variable para ver los turnos por bloque, por dia
    M=[0 for d in dias] #variable para ver la cantidad de medicos por dia
    TY=0
    DY=0
    db=[]
    for m in medicos: 
        
        for j in dias:
            
            for k in range(K):
                for i in index:
                    T3[j][i]+=(Y[(m,k,j)].varValue)*(diccionario_T[k][i]-AP_AL[m][i]*diccionario_T[k][i])
                M[j]+=Y[(m,k,j)].varValue
                if Y[(m,k,j)].varValue>0:
                    T=[0 for i in range(48)]
                    T2=[0 for i in range(48)]
                    medias_horas=0
                    for i in index:
                        T[i]=diccionario_T[k][i]-AP_AL[m][i]*diccionario_T[k][i]
                        T2[i]=diccionario_T[k][i]
                        medias_horas+=T2[i]
                    print ('Dia', dias2[j],'Admisionista',m+1,'turno',k+1,T, medias_horas)
                    db.append((clase_dmd.Centros[c],dias2[j],'Admisionista '+str(m+1),turno(T)[0],turno(T)[1],medias_horas/2,medias_horas/90 ))

        TX=0
        DX=0
        for i in index:
            TX+=T3[j][i]
            DX+=D[c][0][j][i]+D[c][1][j][i]
        TY+=TX
        DY+=DX   
        #print('Utilización',dias2[j],DX/TX, '\tHoras médicas\t', TX/2)
    #print('Médicos por día',M[:])
    #print('Utilización total',DY/TY, '\tFTE\t', TY/90,'\n')
    LP=LpStatus[modelo.status]
    print(LP)#,Y)
    clase_dmd.plot_opt(Y,diccionario_T,c)    
    return db


def buscar_centro(Centro,centros):
    indice=0
    while indice<len(centros):
        if Centro==centros[indice]:
            break
        else:
            indice+=1 
    return indice
def turno(lista):
    #lista=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    largo=len(lista)
    i=0
    while lista[i]==0:  
        i+=1
    inicio=i
    while lista[i]==1 or lista[i+1]==1:
        fin=i
        i+=1
    if math.modf((inicio)/2)[0] ==0:
        inicio_str=str(int(math.modf((inicio)/2)[1]))+':00:00'
    else:
        inicio_str=str(int(math.modf((inicio)/2)[1]))+':30:00'
    if math.modf((fin)/2)[0] ==0:
        fin_str=str(int(math.modf((fin)/2)[1]))+':30:00'
    else:
        fin_str=str(int(math.modf(((fin)/2)+1)[1]))+':00:00'
    return [inicio_str,fin_str]



data_dmd=data("CP_DemandaAdmisionistasTurnos")

data_turnos=data2("DI_turnos_admision_jgh")

data_turnos.columns
H=Herramienta(data_dmd,data_turnos) 

H.D
data_turnos['Hora Inicio Colación']
#print(*H.T)
#print(*H.D)
#print(*H.Q)

#H.plot(7)
for i in range(len(H.Centros)):
    plt.close()


Centro_c_t=[7,52]#range(len(H.Centros))#[11,81] #
#[0,4,5,7,9,11,12,13,15,16,17,18,19,21,22,23]#[12]
#[13,15,16,31,32,33,34,35,36,37]#[13]
#H.plot(13)
#[0,4,5,7,9,11,12,13,15,16,17,18,19,21,22,23]
Export=pd.DataFrame(columns=['Centro','Día','Médico','Inicio Turno', 'Fin Turno','Horas','FTE'])
for i in Centro_c_t:#range(len(H.Centros)):

    db=optimizacion(H,i)
    H.plot(i)
    df= pd.DataFrame(db, columns=['Centro','Día','Médico','Inicio Turno', 'Fin Turno','Horas', 'FTE'])    
    Export=pd.concat([Export,df])#, ignore_index=True)     
end=time.process_time()
print('T° process:',end-start)
Export.to_excel("Propuesta_turnos_admision.xlsx", index=False) 
"""
TEST1=optimizacion(H,11)
TEST2=optimizacion(H,18)
print(TEST1[1])
type(TEST1[1])
TEST1[1]=='Optimal' #funciona para ver si es optimo. 
print(TEST2)
"""
#start2=time.process_time()
#H.Centros
#optimizacion2(H,21)
#H.plot(21)
"""
optimizacion(H,21)
H.plot(21)

optimizacion(H,4)
H.plot(4)
"""
"""
optimizacion2(H,4)
H.plot(4)
optimizacion2(H,13)
H.plot(13)
"""

buscar_centro('Rancagua',H.Centros)
#print(H.Centros)
buscar_centro('Parral',H.Centros)
buscar_centro('Melipilla',H.Centros)
buscar_centro('Concepción',H.Centros)
"""
plt.close()
plt.close()
optimizacion(H,21)
H.plot(21)
"""
#for i in range(12):
#    H.plot(i)
                                        
#end2=time.process_time()
#print(end2-start2)
      
"""
Encontrar la posición de un item dentro de un array
import numpy as np

np_array = np.array((1, 5, 9, 3, 7, 2, 0))
np.where(np_array == 5)
(array([1]),)
"""
        
        
        
