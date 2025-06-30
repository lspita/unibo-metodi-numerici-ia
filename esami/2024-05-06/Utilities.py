import sympy as sym
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify
#Esempio di utilizzo sympy

x=sym.symbols('x')
fs =  sym.exp(-x)-(x+1)
dfs=sym.diff(fs,x,1)
fp=lambdify(x,dfs,np) 

from mpl_toolkits.mplot3d import Axes3D
# Per disegnare una superficie
x = np.arange(-4, 4, 0.1)
y = np.arange(-4, 4, 0.1)
X, Y = np.meshgrid(x, y)
Z=4*X**2+Y**2-4


# Plotta la superficie direttamente
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotta la superficie
ax.plot_surface(X, Y, Z, cmap=plt.cm.viridis)
 
plt.show()

#Per disegnare le curve di livello corrispondente a z=0
plt.contour(X, Y, Z, levels=[0], colors='black')

