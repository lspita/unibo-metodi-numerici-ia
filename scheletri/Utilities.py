import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols, Matrix, lambdify

#Funzione simbolica per calcolare la derivata prima di una funzione simbolica in una variabile
#Esempio
x=sym.symbols('x')
fs =  sym.exp(-x)-(x+1)
dfs=sym.diff(fs,x,1) #Il secondo argomento rappresenta la variabile rispetto a cui derivare e l'ultima l'ordine della derivata
fp=lambdify(x,dfs,np)  #l'ultimo argomento np (nickaname di numpy) serve per specificare che la lambda function 
#può prendere come argomento un numpy array ed eseguire l'operazione su tutte le sue componenti.
f=lambdify(x,fs,np)



#Utili per il metodo di Newton_raphson per il calcolare il punto di minimo di un'equazione nonlineare in 2 variabili:
#Calcolo simbolico di matrice hessiana e vettore gradiente

x_sym, y_sym = symbols('x_sym y_sym')
#Esempio di funzione non lineare simbolica
F_sym=0.5*(0.001*(x_sym-1)**2+(x_sym**2-y_sym)**2)

#Calcolo vettore gradiente
grad_f = sym.derive_by_array(F_sym, (x_sym,y_sym))

# Calcolo dell'Hessiana con sympy.hessian
H = sym.hessian(F_sym, (x_sym,y_sym))

# Conversione delle espressioni simboliche in funzioni numeriche
grad_f_func = sym.lambdify((x_sym,y_sym), grad_f, 'numpy')
H_func = sym.lambdify((x_sym,y_sym), H, 'numpy')
F_func=sym.lambdify((x_sym,y_sym), F_sym, 'numpy')



#Utili per il metodo di Newton Raphson per risolvere un sistema di equazioni non lineari

#Scrivere in froma simbolica le due equazioni del sistema, ad esempio

f1_sym = lambda x_sym,y_sym: x_sym+y_sym-3   #[-1,1]
f2_sym= lambda x_sym,y_sym: x_sym**2+y_sym**2-9


#Definitre il vettore di Funzioni
def F_sym(f1_sym,f2_sym):
    return Matrix([[f1_sym(x_sym,y_sym)], [f2_sym(x_sym,y_sym)]])   

# Calcolo della matrice Jacobiana simbolicamente
J_sym = F_sym(f1_sym,f2_sym).jacobian(Matrix([x_sym, y_sym]))

# Converte la matrice jacobiana Simbolica in una funzione che può essere valutata numericamente mediante lambdify
J_numerical = lambdify([x_sym, y_sym], J_sym, np)

# Converte il vettore di funzioni Simbolico in una funzione che può essere valutata numericamente mediante lambdify
F_numerical = lambdify([x_sym, y_sym], F_sym(f1_sym,f2_sym), np)

#Disegnare superfici e curve di livello per determinare la stima dell'iterato iniziale
#Esempio 
x = np.arange(-4, 4, 0.1)
y = np.arange(-4, 4, 0.1)
X, Y = np.meshgrid(x, y)
Z=np.zeros_like(X)
superfici=F_numerical(X,Y).squeeze()
 
# Plotta la superficie  
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plotta la superficie
ax.plot_surface(X, Y, superfici[0,:,:], cmap='viridis',alpha=0.5)
# Plotta la superficie
ax.plot_surface(X, Y, superfici[1,:,:], cmap='Reds',alpha=0.5)
ax.plot_surface(X, Y, Z, cmap='gray',alpha=0.5)
plt.show()

#Plot superifici di controllo
plt.contour(X, Y,superfici[0,:,:], levels=[0], colors='black')
plt.contour(X, Y,superfici[1,:,:], levels=[0], colors='red')
plt.show()

