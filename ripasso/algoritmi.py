#Zeri di funzione

import math
import numpy as np
import scipy as sp

def sign(x):
  """
  Funzione segno che restituisce 1 se x è positivo, 0 se x è zero e -1 se x è negativo.
  """
  return math.copysign(1, x)

def metodo_bisezione(fname, a, b, tolx, tolf, 
                     maxit=np.ceil(np.log2(np.abs(b-a))/tolx)-1):
    fa=fname(a)
    fb=fname(b)
    if np.sign(fa)*np.sign(fb) > 0:
        print("Non è possibile applicare il metodo di bisezione \n")
        return None, None,None
    
    it = 0
    v_xk = []
    erroreX = tolx+1
    erroreF = tolf+1
    
    while it <= maxit and erroreX >= tolx and erroreF >= tolf:
        xk = a + (b-a)/2 # più stabile rispetto (b+a)/2
        v_xk.append(xk)
        it += 1
        fxk=fname(xk)
        if fxk==0:
            return xk, it, v_xk
        
        if np.sign(fb)*np.sign(fxk) < 0: # la radice si trova nel range [xk, b]
            a = xk
            fa= fxk 
        elif np.sign(fa)*np.sign(fxk) < 0: # la radice si trova nel range [a, xk]
            b = xk
            fb= fxk

        erroreX = np.abs(b-a)
        erroreF = np.abs(fxk)
    
    return xk, it, v_xk

def falsa_posizione(fname,a,b,tolx,tolf,maxit):
    fa=fname(a)
    fb=fname(b)
    if np.sign(fa)*np.sign(fb) > 0:
       print("Metodo di bisezione non applicabile")
       return None,None,None

    it=0
    v_xk=[]
    erroreX=1+tolx
    erroreF=1+tolx
    xprec=a
    while it <= maxit and erroreX >= tolx and erroreF >= tolf:
        xk= a - fa*((b-a)/(fb-fa)) # !!! -fa non +fa
        v_xk.append(xk)
        it+=1
        fxk=fname(xk)
        if fxk==0:
            return xk,it,v_xk

        if np.sign(fa)*np.sign(fxk)<0:
           b= xk
           fb= fxk
        elif  np.sign(fb)*np.sign(fxk)<0:
           a= xk
           fa= fxk
        if xk!=0:
            erroreX=np.abs(xk-xprec)/np.abs(xk)
        else:
            erroreX=np.abs(xk-xprec)

        erroreF = np.abs(fxk)
        xprec=xk
    return xk,it,v_xk

def corde(fname,coeff_ang,x0,tolx,tolf,nmax):
    
    # coeff_ang è il coefficiente angolare della retta che rimane fisso per tutte le iterazioni
    xk=[]
    
    it=0
    errorex=1+tolx
    erroref=1+tolf
    while it <= nmax and errorex >= tolx and erroref >= tolf:
        fx0=fname(x0)
        d=fx0/coeff_ang
      
        x1=x0-d # !!! -d non +d
        fx1=fname(x1)
        if x1!=0:
            errorex=np.abs(d)/np.abs(x1) # !!! abs(d) non abs(x1-x0)
        else:
            errorex=np.abs(d)
       
        erroref=np.abs(fx1)
       
        x0=x1
        it=it+1
        xk.append(x1)
      
        if it==nmax:
            print('Corde : raggiunto massimo numero di iterazioni \n')
    
    return x1,it,xk
    
def newton(fname,fpname,x0,tolx,tolf,nmax):
  
    xk=[]
    
    it=0
    errorex=1+tolx
    erroref=1+tolf
    while it <= nmax and errorex >= tolx and erroref >= tolf:
       
        fx0=fname(x0)
        fpx0=fpname(x0)
        if np.abs(fpx0) <= np.spacing(1):
            print(" derivata prima nulla in x0")
            return None, None,None
        d=fx0/fpx0
    
        x1=x0-d
        fx1=fname(x1)
        erroref=np.abs(fx1)
        if x1!=0:
            errorex=np.abs(d)/np.abs(x1)
        else:
            errorex=np.abs(d)
    
        it=it+1
        x0=x1
        xk.append(x1)
      
        if it==nmax:
            print('Newton: raggiunto massimo numero di iterazioni \n')
        
    
    return x1,it,xk

def newton_modificato(fname,fpname,m,x0,tolx,tolf,nmax):
  
    #m è la molteplicità dello zero
    
    xk=[]
    
    it=0
    errorex=1+tolx
    erroref=1+tolf
    while it <= nmax and errorex >= tolx and errorey >= toly:
       
        fx0=fname(x0)
        fpx0 = fpname(x0)
        if np.abs(fpx0) <= np.spacing(1):
            print(" derivata prima nulla in x0")
            return None, None,None
        d=fx0/fpx0 # !!! non m*
        
        x1=x0-m*d # !!! m*d va qui non sopra
        fx1=fname(x1)
        erroref=np.abs(fx1)
        if x1!=0:
            errorex=np.abs(d)/np.abs(x1)
        else:
            errorex=np.abs(d)
        
        it=it+1
        x0=x1
        xk.append(x1)
          
        if it==nmax:
            print('Newton modificato: raggiunto massimo numero di iterazioni \n')
        
    
    return x1,it,xk
    
def secanti(fname,xm1,x0,tolx,tolf,nmax):
        xk=[]
        
        it=0
        errorex=1+tolx
        erroref=1+tolf
        while it <= nmax and errorex >= tolx and erroref >= tolf:
            
            fxm1=fname(xm1)
            fx0=fname(x0)
            d=fx0*((xm1-x0)/(fxm1-fx0)) # !!! x0-xm1 e (le f di conseguenza) non xm1-x0

            x1=x0-d
          
            
            fx1=fname(x1)
            xk.append(x1);
            if x1!=0:
                errorex=np.abs(d)/np.abs(x1)
            else:
                errorex=np.abs(d)
                
            erroref=np.abs(fx1)
            xm1=x0 
            x0=x1
            
            it=it+1;
           
       
        if it==nmax:
           print('Secanti: raggiunto massimo numero di iterazioni \n')
        
        return x1,it,xk
    
def stima_ordine(xk,iterazioni):
     #Vedi dispensa allegata per la spiegazione

      k=iterazioni-4
      p=np.log(abs(xk[k+2]-xk[k+3])/abs(xk[k+1]-xk[k+2]))/np.log(abs(xk[k+1]-xk[k+2])/abs(xk[k]-xk[k+1]));
     
      ordine=p
      return ordine

#Soluzione di sistemi lineari


#Soluzione di sistemi di equazioni non lineari
def newton_raphson(initial_guess, F_numerical, J_Numerical, tolX, tolF, max_iterations):
    

    X= np.array(initial_guess, dtype=float)
    
   

    it=0
    
    erroreF=1+tolF
    erroreX=1+tolX
    
    errore=[]
    
    while it <= max_iterations and erroreX >= tolX and erroreF >= tolF:
        
        jx = J_Numerical(X[0], X[1])
        
        if np.linalg.det(jx) == 0: # !!! controllo determinante
            print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
            return None, None,None
        
        fx = F_numerical(X[0], X[1])
        fx = fx.squeeze() 
        
        s = np.solve(jx, -fx)
        
        Xnew=X+s
        
        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew !=0:
            erroreX=np.linalg.norm(s, 1)/normaXnew # !!! s non X-Xnew (però è equivalente)
        else:
            erroreX=np.linalg.norm(s, 1)
        
        errore.append(erroreX)
        fxnew=F_numerical(Xnew[0], Xnew[1])
        erroreF= np.linalg.norm(fxnew.squeeze(),1)
        X=Xnew
        it=it+1
    
    return X,it,errore

def def newton_raphson_corde(initial_guess, F_numerical, J_Numerical, tolX, tolF, max_iterations):
    

    X= np.array(initial_guess, dtype=float)
    
   

    it=0
    
    erroreF=1+tolF
    erroreX=1+tolX
    
    errore=[]
    
    while it <= max_iterations and erroreX >= tolX and erroreF >= tolF:
        
        if it==0
            jx = J_Numerical(X[0], X[1])
        
            if np.linalg.det(jx) == 0:
                print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
                return None, None,None
        
        fx = F_numerical(X[0], X[1])
        fx = fx.squeeze() 
        
        s = np.solve(jx, -fx)
        
        Xnew=X+s
        
        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew !=0:
            erroreX=np.linalg.norm(s, 1)/normaXnew 
        else:
            erroreX=np.linalg.norm(s, 1)
        
        errore.append(erroreX)
        fxnew=F_numerical(X[0], X[1])
        erroreF= np.linalg.norm(fxnew.squeeze(),1)
        X=Xnew
        it=it+1
    
    return X,it,errore


def newton_raphson_sham(initial_guess, update, F_numerical, J_Numerical, tolX, tolF, max_iterations):
    
    

    X= np.array(initial_guess, dtype=float)
    
   

    it=0
    
    erroreF=1+tolF
    erroreX=1+tolX
    
    errore=[]
    
    while it <= max_iterations and erroreF >= tolF and erroreX >= tolX:
        
        if it%update == 0:
            jx = J_numerical(X[0], X[1])
        
            if np.linalg.det(jx) == 0:
                print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
                return None, None,None
        
        fx = F_numerical(X[0], X[1])
        fx = fx.squeeze() 
        
        s = np.solve(jx, -fx)
        
        Xnew=X+s
        
        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew !=0:
            erroreX=np.linalg.norm(s, 1) /normaXnew
        else:
            erroreX=np.linalg.norm(s, 1)
        
        errore.append(erroreX)
        fxnew=F_numerical(X[0], X[1])
        erroreF= np.linalg.norm(fxnew.squeeze(),1)
        X=Xnew
        it=it+1
    
    return X,it,errore




#Minimo di una funzion enon lineare

def newton_raphson_minimo(initial_guess, grad_func, Hessian_func, tolX, tolF, max_iterations):
    

    X= np.array(initial_guess, dtype=float)
    
    it=0
    
    erroreF=1+tolX
    erroreX=1+tolF
    
    errore=[]
    
    while it <= max_iterations and erroreF >= tolF and erroreX >= tolX:
        
        Hx = Hessian_func(X[0], X[1])
        
        if np.linalg.det(Hx) == 0:
            print("La matrice Hessiana calcolata nell'iterato precedente non è a rango massimo")
            return None, None,None
        
        gfx =  grad_func(X[0], X[1])
        gfx = gfx.squeeze() 
        
        s = np.solve(Hx, -gfx)
        
        Xnew=X+s
        
        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew!=0:
            erroreX=np.linalg.norm(s, 1)/normaXnew
        else:
            erroreX=np.linalg.norm(s, 1)
            
        errore.append(erroreX)
        gfxnew=grad_func(Xnwew[0], Xnew[1])
        erroreF= np.linalg.norm(gfxnew.squeeze(),1)
        X=Xnew
        it=it+1
    
    return X,it,errore

#Metodi Iterativi basati sullo splitting della matrice: jacobi, gauss-Seidel - Gauss_seidel SOR
def jacobi(A,b,x0,toll,it_max):
    errore=1000
    d=np.diag(A)
    n=A.shape[0]
    D = np.diag(d)
    E=np.tril(A, -1)
    F=np.triu(A, 1)
    M=D
    N=-(E+F) 
    Minv = np.linalg.inv(M)
    T=Minv@N
    autovalori=np.linalg.eigvals(T)
    raggiospettrale=np.max(np.abs(autovalori)) 
    print("raggio spettrale jacobi", raggiospettrale)
    it=0
    
    er_vet=[]
    while it<=it_max and errore>=toll:
        x=T@x0 + Minv@b
        errore=np.linalg.norm(x-x0)/np.linalg.norm(x)
        er_vet.append(errore)
        x0=x.copy()
        it=it+1
    return x,it,er_vet


def gauss_seidel(A,b,x0,toll,it_max):
    errore=1000
    d=np.diag(A)
    D=np.diag(d)
    E=np.tril(A, -1) 
    F=np.triu(A, 1)
    M=(D+E)
    N=-F 
    Minv=np.linalg.inv(M)
    T=Minv@N
    autovalori=np.linalg.eigvals(T)
    raggiospettrale=np.max(np.abs(autovalori))
    print("raggio spettrale Gauss-Seidel ",raggiospettrale)
    it=0
    er_vet=[]
    while it <= max_iterations and errore >= toll:
        x,_=Lsolve(M, b+N@x0) # !!! non T@x0 + Minv@b
        errore=np.linalg.norm(x-x0)/np.linalg.norm(x)
        er_vet.append(errore)
        x0=x.copy()
        it=it+1
    return x,it,er_vet

def gauss_seidel_sor(A,b,x0,toll,it_max,omega):
    errore=1000
    d=np.diag(A)
    D=np.diag(d)
    E=np.tril(A, -1)
    F=np.triu(A, 1)
    Momega=D+omega*E
    Nomega=(1-omega)*D-omega*F
    Minv = np.linalg.inv(M)
    T=Minv@Nomega
    autovalori=np.linalg.eigvals(T)
    raggiospettrale=np.max(np.abs(autovalori))
    print("raggio spettrale Gauss-Seidel SOR ", raggiospettrale)
    
    M=D+E
    N=-F 
    it=0
    xold=x0.copy()
    xnew=x0.copy()
    er_vet=[]
    while it<=it_max and errore>=toll:
        
        xtilde= Lsolve(M, b+N@xold) # !!! non xold + omega*raggiospettrale
        xnew,_=(1-omega)*xold+omega*xtilde # !!! non Lsolve(Momega, b+Nomega@xtilde)
        errore=np.linalg.norm(xnew-xold)/np.linalg.norm(xnew)
        er_vet.append(errore)
        xold=xnew.copy()
        it=it+1
    return xnew,it,er_vet


#Metodi di Discesa

def steepestdescent(A,b,x0,itmax,tol):
 
    n,m=A.shape
    if n!=m:
        print("Matrice non quadrata")
        return [],[]
    
    
   # inizializzare le variabili necessarie
    x = x0

     
    r = A@x-b # !!! vettore resto
    p = -r
    it = 0
    nb=np.linalg.norm(b)
    errore=np.linalg.norm(r)/nb
    vec_sol=[]
    vec_sol.append(x.copy())
    vet_r=[]
    vet_r.append(errore)
     
# utilizzare il metodo del gradiente per trovare la soluzione
    while it <= itmax and errore >= tol: 
        it=it+1
        Ap=A@p
       
        alpha = -(r.T@p)/(p.T@Ap) # !!!
                
        x = r + alpha*p # !!!
        
         
        vec_sol.append(x.copy())
        r=A@x-b
        errore=np.linalg.norm(r)/nb
        vet_r.append(errore)
        p =-r # direzione opposta al gradiente per massima discesa
        
    iterates_array = np.vstack([arr.T for arr in vec_sol])
    return x,vet_r,iterates_array,it

def conjugate_gradient(A,b,x0,itmax,tol):
    n,m=A.shape
    if n!=m:
        print("Matrice non quadrata")
        return [],[]
    
    
   # inizializzare le variabili necessarie
    x = x0
    
    r = A@x-b
    p = -r 
    it = 0
    nb=np.linalg.norm(b)
    errore=np.linalg.norm(r)/nb
    vec_sol=[]
    vec_sol.append(x0.copy())
    vet_r=[]
    vet_r.append(errore)
# utilizzare il metodo del gradiente coniugato per calcolare la soluzione
    while it <= itmax and errore >= tol:
        it=it+1
        Ap=A@p
        alpha = -(r.T@p)/(p.T@Ap) 
        x = x + alpha*p
        vec_sol.append(x.copy())
        rtr_old=r.T@r
        r= r+alpha*Ap # !!! non A@x-b
        gamma=r.T@r/rtr_old 
        errore=np.linalg.norm(r)/nb
        vet_r.append(errore)
        p = -r + gamma*p
   
    iterates_array = np.vstack([arr.T for arr in vec_sol])
    return x,vet_r,iterates_array,it

#Soluzione di sistemi sovradeterminati

def eqnorm(A,b):
 
    G=A.T@A
    f=A.T@b 
    
    L=sp.linalg.cholesky(G, lower=True) # !!!
    U=L.T

    x, flag = Lsolve(L, f)
    if flag == 0:
        x, flag = Usolve(U, x)
    
    residuo = np.linalg.norm(A@x-b)**2
        
    return x


def qrLS(A,b):
    n=A.shape[1]  # numero di colonne di A
    Q,R=sp.linalg.qr(A)
    h=Q.T@b
    x,_ = Usolve(R[:n,:], h[:n])
    residuo=np.linalg.norm(h[n:])**2 # !!! non h[:n]
    return x,residuo



def SVDLS(A,b):
    m,n=A.shape  #numero di righe e  numero di colonne di A
    U,s,VT=spLin.svd(A)  
    
    V=VT.T
    thresh=np.spacing(1)*m*s[0] ##Calcolo del rango della matrice, numero dei valori singolari maggiori di una soglia
    k=np.count_nonzero(s>thresh)
    
    d=U.T@b
    d1=d[:k].reshape((k, 1)) 
    s1=s[:k].reshape((k, 1))
    
    c=d1/s1
    x=V[:,:k]@c # !!! non V@c 
    residuo=np.linalg.norm(d[k:])**2 
    return x,residuo
     

#-----------Interpolazione

def plagr(xnodi,j):
    
    xzeri=np.zeros_like(xnodi)
    n=xnodi.size
    if j==0:
       xzeri==xnodi[1:n]
    else:
       xzeri=np.append(xnodi[:j], xnodi[j+1:n])
    
    num= np.poly(xzeri)
    den= np.polyval(num, xnodi[j])
    
    p= num/den
    
    return p



def InterpL(x, y, xx):
     
     n=x.size # !!! non shape[0] (ma equivalente)
     m=xx.size # !!! non shape[0] (ma equivalente)
     L=np.zeros((m,n))
     for j in range(n):
        p=plagr(x, j) 
        L[:,j]=np.polyval(p, xx) 
    
    
     return L@y