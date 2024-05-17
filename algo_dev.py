import numpy as np

def calculate_E(D, c, x):
    # Calculate E = Dx + c
    E = np.dot(D, x) + c
    return E


def calculate_beta(E, JNNO, x, l, u):
    beta = 0
    for j in JNNO:
        if E[j] > 0:
            beta += E[j] * (x[j] - l[j])
        else:
            beta += E[j] * (x[j] - u[j])
    return beta


def calc_theta_j0(u, x, l, E, j0):
    if E[j0] > 0:
        theta_j0 = (u[j0] - x[j0])
    else:
        theta_j0 = (x[j0] - l[j0])
    return theta_j0


def calc_theta_jb(u, x, l, JB, d):
    theta_jb = None
    
    for j in JB:
        if d[j] < 0:
            theta_jb = (l[j] - x[j]) / d[j]
        elif d[j] > 0:
            theta_jb = (u[j] - x[j]) / d[j]
        elif d[j] == 0 :
            theta_jb = float('inf')
    return theta_jb

def calc_theta_F(t,E,j0):
    E_j0 = E[j0-1]
    t_j0 = t[j0-1]
    if E_j0 * t_j0 < 0:
        theta_F = -E_j0 / t_j0
    else:
        theta_F = float('inf')
    return theta_F


def main_algorithm(D, c, l, u, ep):
    # Initialization
    x = np.array([0, 0, 1], dtype=float)
    JB = set()
    JN = set(range(D.shape[1]))
    
    while True:
        E = calculate_E(D, c, x)
    
        # JNNO contient les indices des variables non basiques qui ne satisfont pas les conditions d'optimalité
        JNNO = {j for j in JN if (x[j] == l[j] and E[j] < 0) or (x[j] == u[j] and E[j] > 0)}
        if not JNNO:
            JNNO = JN

        # Sélection de j0 tel que |E_j0| = max{|E_j| pour j dans JNNO}
        j0 = max(JNNO, key=lambda j: abs(E[j]))

        # Calcul de l'estimation de sub-optimalité beta(x, JB)
        beta = calculate_beta(E, JNNO, x, l, u)
        
        print("E =", E)
        print("JNNO =", JNNO)
        print("j0 =", j0)
        print("beta(x, JB) =", beta)
        
        if beta <= ep:
            print("Le critère de suboptimalité est vérifié, l'algorithme s'arrête.")
            return x, JB, JN
        
        # Étape S3: Calcul des directions et du pas optimal
        d = np.array([0, 1, 0])
        t = np.array([2, 3, 0])
        
        theta_j0 = calc_theta_j0(u, x, l, E, j0)
        theta_jb = calc_theta_jb(u, x, l, JB, d)
        theta_F = calc_theta_F(t, E, j0)
        
        theta_list = [theta for theta in [theta_j0, theta_jb, theta_F] if theta is not None]
        theta = min(theta_list)
        
        print("Theta values:", [theta_j0, theta_jb, theta_F])
        print("Theta =", theta)

        # Update new solution and reduced costs
        x = x + theta * d
        E = calculate_E(D, c, x)
        
        # Update support sets
        if theta == theta_j0:
            JN.remove(j0)
            JB.add(j0)
        elif theta == theta_jb:
            for j in JB:
                if d[j] != 0 and (theta == (l[j] - x[j]) / d[j] or theta == (u[j] - x[j]) / d[j]):
                    JB.remove(j)
                    JN.add(j)
                    break
        elif theta == theta_F:
            JN.remove(j0)
            JB.add(j0)
        
        print("Updated x =", x)
        print("Updated E =", E)
        print("Updated JB =", JB)
        print("Updated JN =", JN)

# Given example data
D = np.array([[4, 2, 1], [2, 3, 0], [1, 0, 2]], dtype=float)
c = np.array([1, -2, 2], dtype=float)
l = np.array([0, -1, 1], dtype=float)
u = np.array([3, 4, 5], dtype=float)
ep = 0

# Execute the algorithm
solution, JB, JN = main_algorithm(D, c, l, u, ep)
print(f"Optimal solution: {solution}")
