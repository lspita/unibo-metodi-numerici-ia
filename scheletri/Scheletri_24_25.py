#Zeri di funzione

import math
def sign(x):
  """
  Funzione segno che restituisce 1 se x è positivo, 0 se x è zero e -1 se x è negativo.
  """
  return math.copysign(1, x)

def metodo_bisezione(fname, a, b, tolx):
 
 fa=fname(a)
 fb=fname(b)
 if #to do
     print("Non è possibile applicare il metodo di bisezione \n")
     return None, None,None

 it = 0
 v_xk = []

 
 
 while #to do
    xk = #to do
    v_xk.append(xk)
    it += 1
    fxk=fname(xk)
    if fxk==0:
      return xk, it, v_xk

    if # to do
      a = #to do 
      fa= #to do 
    elif # to do
      b =
      fb=

 
 return xk, it, v_xk

def falsa_posizione(fname,a,b,tolx,tolf,maxit):
    fa=fname(a)
    fb=fname(b)
    if #to do:
       print("Metodo di bisezione non applicabile")
       return None,None,None

    it=0
    v_xk=[]
    fxk=1+tolf
    errore=1+tolx
    xprec=a
    while #to do:
        xk= #to do
        v_xk.append(xk)
        it+=1
        fxk=# to do
        if fxk==0:
            return xk,it,v_xk

        if #to do
           b= #to do
           fb= #to do
        elif #to do 
           a=#to do
           fa=#to do
        if xk!=0:
            errore= #to do
        else:
            errore=#to do 
        xprec=xk
    return xk,it,v_xk

def corde(fname,coeff_ang,x0,tolx,tolf,nmax):
    
     # coeff_ang è il coefficiente angolare della retta che rimane fisso per tutte le iterazioni
        xk=[]
        
        it=0
        errorex=1+tolx
        erroref=1+tolf
        while #to do
           
           fx0=# to do
           d=# to do
          
           x1=#to do
           fx1=#
           if x1!=0:
                errorex=#to do 
           else:
                errorex=#to do
           
           erroref=#to do
           
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
        while #to do
           
           fx0=fname(x0)
           if #to do
                print(" derivata prima nulla in x0")
                return None, None,None
           d=#to do 

           x1=#to do
           fx1=fname(x1)
           erroref=np.abs(fx1)
           if x1!=0:
                errore=#to do
           else:
                errore=#to do 

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
        while #to do
           
           fx0=fname(x0)
           if #to do
                print(" derivata prima nulla in x0")
                return None, None,None
           d=#to do 

           x1=#to do
           fx1=fname(x1)
           erroref=np.abs(fx1)
           if x1!=0:
                errore=#to do
           else:
                errore=#to do 

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
        while #to do
            
            fxm1=#to do
            fx0=#to do 
            d=#to do 

            x1=#to do 
          
            
            fx1=fname(x1)
            xk.append(x1);
            if x1!=0:
                errorex=#to do 
            else:
                errorex=#to do
                
            erroref=#to do 
            xm1=#to do 
            x0=#to do
            
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


#Soluzione di sistemi di equazioni non lineari
def newton_raphson(initial_guess, F_numerical, J_Numerical, tolX, tolF, max_iterations):
    

    X= np.array(initial_guess, dtype=float)
    
   

    it=0
    
    erroreF=1+tolF
    erroreX=1+tolX
    
    errore=[]
    
    while #to do
        
        jx = #to do
        
        if #to do
            print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
            return None, None,None
        
        fx = #To do
        fx = fx.squeeze() 
        
        s = #to do 
        
        Xnew=#to do
        
        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew !=0:
            erroreX=#to do 
        else:
            erroreX=#to do 
        
        errore.append(erroreX)
        fxnew=#to do
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
    
    while #to do
        
        if it#to do
            jx = #to do
        
            if #to do
                print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
                return None, None,None
        
        fx = #To do
        fx = fx.squeeze() 
        
        s = #to do 
        
        Xnew=#to do
        
        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew !=0:
            erroreX=#to do 
        else:
            erroreX=#to do 
        
        errore.append(erroreX)
        fxnew=#to do
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
    
    while #to do
        
        if it# to to:
            jx = #to do
        
            if #to do
                print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
                return None, None,None
        
        fx = #To do
        fx = fx.squeeze() 
        
        s = #to do 
        
        Xnew=#to do
        
        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew !=0:
            erroreX=#to do 
        else:
            erroreX=#to do 
        
        errore.append(erroreX)
        fxnew=#to do
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
    
    while #to do:
        
        Hx = #to do
        
        if #to do
            print("La matrice Hessiana calcolata nell'iterato precedente non è a rango massimo")
            return None, None,None
        
        gfx =  #to do 
        gfx = gfx.squeeze() 
        
        s = #to do 
        
        Xnew=#to do 
        
        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew!=0:
            erroreX=#to do
        else:
            erroreX=#to do
            
        errore.append(erroreX)
        gfxnew=#to do 
        erroreF= np.linalg.norm(gfxnew.squeeze(),1)
        X=Xnew
        it=it+1
    
    return X,it,errore

#Metodi Iterativi basati sullo splitting della matrice: jacobi, gauss-Seidel - Gauss_seidel SOR
def jacobi(A,b,x0,toll,it_max):
    errore=1000
    d=np.diag(A)
    n=A.shape[0]
    invM=np.diag(1/d)
    E=#to do 
    F=#to do 
    N=#to do 
    T=#to do 
    autovalori=np.linalg.eigvals(T)
    raggiospettrale=#to do 
    print("raggio spettrale jacobi", raggiospettrale)
    it=0
    
    er_vet=[]
    while it<=it_max and errore>=toll:
        x=#to do 
        errore=#to do 
        er_vet.append(errore)
        x0=x.copy()
        it=it+1
    return x,it,er_vet


def gauss_seidel(A,b,x0,toll,it_max):
    errore=1000
    d=#to do 
    D=#to do 
    E=#to do 
    F=#to do 
    M=#to do 
    N=#to do 
    T=#to do 
    autovalori=np.linalg.eigvals(T)
    raggiospettrale=#to do 
    print("raggio spettrale Gauss-Seidel ",raggiospettrale)
    it=0
    er_vet=[]
    while #to do 
        x=#to do 
        errore=#to do 
        er_vet.append(errore)
        x0=x.copy()
        it=it+1
    return x,it,er_vet

def gauss_seidel_sor(A,b,x0,toll,it_max,omega):
    errore=1000
    d=#to do 
    D=#to do 
    E=#to do 
    F=#to do  
    Momega=D+omega*E
    Nomega=(1-omega)*D-omega*F
    T=#to do
    autovalori=np.linalg.eigvals(T)
    raggiospettrale=#to do
    print("raggio spettrale Gauss-Seidel SOR ", raggiospettrale)
    
    M=#to do 
    N=#to do 
    it=0
    xold=x0.copy()
    xnew=x0.copy()
    er_vet=[]
    while it<=it_max and errore>=toll:
        
        xtilde= #to do 
        xnew=#to do 
        errore=#to do 
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

     
    r = #to do 
    p = #to do 
    it = 0
    nb=np.linalg.norm(b)
    errore=np.linalg.norm(r)/nb
    vec_sol=[]
    vec_sol.append(x.copy())
    vet_r=[]
    vet_r.append(errore)
     
# utilizzare il metodo del gradiente per trovare la soluzione
    while #to do 
        it=it+1
        Ap=#to do 
       
        alpha = # to do
                
        x = #to do 
        
         
        vec_sol.append(x.copy())
        r=#to do 
        errore=np.linalg.norm(r)/nb
        vet_r.append(errore)
        p =#to do 
        
    iterates_array = np.vstack([arr.T for arr in vec_sol])
    return x,vet_r,iterates_array,it

def conjugate_gradient(A,b,x0,itmax,tol):
    n,m=A.shape
    if n!=m:
        print("Matrice non quadrata")
        return [],[]
    
    
   # inizializzare le variabili necessarie
    x = x0
    
    r = #to do 
    p = àto do 
    it = 0
    nb=np.linalg.norm(b)
    errore=np.linalg.norm(r)/nb
    vec_sol=[]
    vec_sol.append(x0.copy())
    vet_r=[]
    vet_r.append(errore)
# utilizzare il metodo del gradiente coniugato per calcolare la soluzione
    while #to do 
        it=it+1
        Ap=#to do A.dot(p)
        alpha = #to do 
        x = #to do 
        vec_sol.append(x.copy())
        rtr_old=#to do
        r= #to do 
        gamma=#to do 
        errore=np.linalg.norm(r)/nb
        vet_r.append(errore)
        p = #to do 
   
    iterates_array = np.vstack([arr.T for arr in vec_sol])
    return x,vet_r,iterates_array,it

#Soluzione di sistemi sovradeterminati

def eqnorm(A,b):
 
    G= #to do  
    f=#to do 
    
    L=
    U=
        
        
    return x


def qrLS(A,b):
    n=A.shape[1]  # numero di colonne di A
    Q,R=spLin.qr(A)
    h=#to do 
    x,_ #to do
    residuo=#to do 
    return x,residuo



def SVDLS(A,b):
    m,n=A.shape  #numero di righe e  numero di colonne di A
    U,s,VT=spLin.svd(A)  
    
    V=VT.T
    thresh=np.spacing(1)*m*s[0] ##Calcolo del rango della matrice, numero dei valori singolari maggiori di una soglia
    k=#to do 
    
    d=#to do 
    d1=#to do 
    s1=#to do 
    
    c=#to do 
    x=#to do 
    residuo=#to do 
    return x,residuo
     

#-----------Interpolazione

def plagr(xnodi,j):
    
    xzeri=np.zeros_like(xnodi)
    n=xnodi.size
    if j==0:
       xzeri==#to do 
    else:
       xzeri=np.append(#to do )
    
    num= 
    den= 
    
    p= 
    
    return p



def InterpL(x, y, xx):
     
     n=#to do
     m=#to do 
     L=np.zeros((m,n))
     for j #to do :
        p=#to do 
        L[:,j]=#to do 
    
    
     return #to do 