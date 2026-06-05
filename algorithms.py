import numpy as np
from scipy.special import gamma as scipy_gamma


def clip(x, lb, ub): return np.clip(x, lb, ub)

def levy_step(dim, beta=1.5):
    num = scipy_gamma(1+beta)*np.sin(np.pi*beta/2)
    den = scipy_gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma = (num/den)**(1/beta)
    u = np.random.randn(dim)*sigma
    v = np.random.randn(dim)
    return u / (np.abs(v)**(1/beta))

def init_pop(N, dim, lb, ub):
    return lb + np.random.rand(N, dim)*(ub - lb)


# 1. BASE RPO

def rpo(func, lb, ub, dim, max_fes, pop_size=50):
    lb, ub = np.array(lb, float), np.array(ub, float)
    X   = init_pop(pop_size, dim, lb, ub)
    fit = np.array([func(X[i]) for i in range(pop_size)])
    fes = pop_size
    bi  = int(np.argmin(fit)); bp = X[bi].copy(); bv = fit[bi]
    history = [(fes, bv)]
    T = max(1, (max_fes - pop_size) // (2*pop_size))

    for t in range(1, T+1):
        # Phase 1: Foraging (Eq. 4-6)
        for i in range(pop_size):
            if fes >= max_fes: break
            better = [k for k in range(pop_size) if fit[k] < fit[i]]
            pfs = better + [bi]                          # Eq. 4
            SFS = X[pfs[np.random.randint(len(pfs))]]
            r = np.random.rand(dim)
            I = np.random.choice([1, 2], size=dim)      # Eq. 5
            X_P1 = clip(X[i] + r*(SFS - I*X[i]), lb, ub)
            f = func(X_P1); fes += 1
            if f < fit[i]:                              # Eq. 6
                X[i], fit[i] = X_P1, f
                if f < bv: bv, bp, bi = f, X_P1.copy(), i

        # Phase 2: Climbing (Eq. 7-8)
        for i in range(pop_size):
            if fes >= max_fes: break
            r = np.random.rand(dim)
            X_P2 = clip(X[i] + (lb + r*(ub-lb))/t, lb, ub)  # Eq. 7
            f = func(X_P2); fes += 1
            if f < fit[i]:                              # Eq. 8
                X[i], fit[i] = X_P2, f
                if f < bv: bv, bp, bi = f, X_P2.copy(), i

        history.append((fes, bv))
    return {"best_val": bv, "best_pos": bp, "history": history}


# 2. MRPO 

def mrpo(func, lb, ub, dim, max_fes, pop_size=50):
    
    lb, ub = np.array(lb, float), np.array(ub, float)

    # OBL initialisation
    X_o  = init_pop(pop_size, dim, lb, ub)
    X_op = lb + ub - X_o
    Xa   = np.vstack([X_o, X_op])
    fa   = np.array([func(Xa[i]) for i in range(2*pop_size)])
    fes  = 2*pop_size
    idx  = np.argsort(fa)[:pop_size]
    pop  = Xa[idx].copy(); fitness = fa[idx].copy()
    bi   = int(np.argmin(fitness)); gbp = pop[bi].copy(); gbf = fitness[bi]
    history = [(fes, gbf)]

    max_iter = max(1, (max_fes - 2*pop_size) // (pop_size * 4))

    for t in range(1, max_iter+1):
        sigma = 0.1*(ub-lb)*np.exp(-3*t/max_iter)

        for i in range(pop_size):
            if fes >= max_fes: break

            better = [k for k in range(pop_size) if fitness[k] < fitness[i]]
            pfs    = better + [bi]
            k      = pfs[np.random.randint(len(pfs))]
            SF     = pop[k]
            I      = np.random.choice([1, 2])
            r1     = np.random.rand(dim)
            if fitness[k] < fitness[i]:
                np1 = pop[i] + r1*(SF - I*pop[i])
            else:
                np1 = pop[i] + r1*(pop[i] - SF)
            np1 = clip(np1, lb, ub); f1 = func(np1); fes += 1
            if f1 < fitness[i]: pop[i], fitness[i] = np1, f1

            if fes >= max_fes: break

            # Phase 2: Directional Gaussian 
            alpha = 0.5*(1 - t/max_iter)
            np2   = pop[i] + alpha*(gbp-pop[i]) + np.random.randn(dim)*sigma
            np2   = clip(np2, lb, ub); f2 = func(np2); fes += 1
            if f2 < fitness[i]: pop[i], fitness[i] = np2, f2

            # Hill climbing x2 
            for _ in range(2):
                if fes >= max_fes: break
                cand = clip(pop[i] + np.random.randn(dim)*sigma*0.5, lb, ub)
                fc   = func(cand); fes += 1
                if fc < fitness[i]: pop[i], fitness[i] = cand, fc

            if fitness[i] < gbf: gbf, gbp, bi = fitness[i], pop[i].copy(), i

        # Best refinement x5 
        for _ in range(5):
            if fes >= max_fes: break
            cand = clip(gbp + np.random.randn(dim)*sigma.mean()*0.5, lb, ub)
            fc   = func(cand); fes += 1
            if fc < gbf: gbf, gbp = fc, cand.copy()

        history.append((fes, gbf))
    return {"best_val": gbf, "best_pos": gbp, "history": history}


# 3-9. COMPETITOR ALGORITHMS 

# 3. RIME

def rime(func, lb, ub, dim, max_fes, pop_size=50):
    lb, ub = np.array(lb, float), np.array(ub, float)
    X   = init_pop(pop_size, dim, lb, ub)
    fit = np.array([func(X[i]) for i in range(pop_size)])
    fes = pop_size
    bi  = int(np.argmin(fit)); bp = X[bi].copy(); bv = fit[bi]
    history = [(fes, bv)]
    T = max(1, (max_fes - pop_size) // pop_size)

    for t in range(1, T+1):
        if fes >= max_fes: break
        norm_t = t / T

        for i in range(pop_size):
            if fes >= max_fes: break

            # Soft-rime search strategy: cosine random walk around global best
            r1    = np.random.rand()
            theta = np.random.rand() * 2 * np.pi
            h     = np.random.rand(dim) * (ub - lb) + lb   # environmental noise
            beta  = 1 / np.sqrt(t)                          # freezing coefficient
            X_soft = bp + r1 * np.cos(theta) * beta * h
            X_soft = clip(X_soft, lb, ub)
            f_soft = func(X_soft); fes += 1
            if f_soft < fit[i]:
                X[i], fit[i] = X_soft, f_soft
                if f_soft < bv: bv, bp, bi = f_soft, X_soft.copy(), i

            if fes >= max_fes: break

            # Hard-rime puncture: replace one dimension with global best's value;
            # probability of triggering grows linearly with iteration
            if np.random.rand() < norm_t:
                X_hard          = X[i].copy()
                X_hard[np.random.randint(dim)] = bp[np.random.randint(dim)]
                X_hard = clip(X_hard, lb, ub)
                f_hard = func(X_hard); fes += 1
                if f_hard < fit[i]:
                    X[i], fit[i] = X_hard, f_hard
                    if f_hard < bv: bv, bp, bi = f_hard, X_hard.copy(), i

        history.append((fes, bv))
    return {"best_val": bv, "best_pos": bp, "history": history}


# 4. GO — Growth Optimizer

def go(func, lb, ub, dim, max_fes, pop_size=50):
    lb, ub = np.array(lb, float), np.array(ub, float)
    X   = init_pop(pop_size, dim, lb, ub)
    fit = np.array([func(X[i]) for i in range(pop_size)])
    fes = pop_size
    bv  = fit.min(); bp = X[np.argmin(fit)].copy()
    history = [(fes, bv)]

    while fes < max_fes:
        idx_sorted = np.argsort(fit)

        for i in range(pop_size):
            if fes >= max_fes: break

            # ── Learning phase ──────────────────────────────────────
            better_X = X[idx_sorted[np.random.randint(1, 5)]]
            worse_X  = X[idx_sorted[np.random.randint(pop_size-5, pop_size)]]
            peers    = [k for k in range(pop_size) if k != i]
            L1, L2   = np.random.choice(peers, 2, replace=False)

            d1 = np.linalg.norm(X[idx_sorted[0]] - better_X) + 1e-12
            d2 = np.linalg.norm(X[idx_sorted[0]] - worse_X ) + 1e-12
            d3 = np.linalg.norm(better_X          - worse_X ) + 1e-12
            d4 = np.linalg.norm(X[L1]             - X[L2]   ) + 1e-12
            rate = d1 + d2 + d3 + d4

            r = np.random.rand(dim)
            X_learn = X[i] + r * (
                (d1/rate)*(X[idx_sorted[0]] - better_X) +
                (d2/rate)*(X[idx_sorted[0]] - worse_X ) +
                (d3/rate)*(better_X          - worse_X ) +
                (d4/rate)*(X[L1]             - X[L2]   )
            )
            X_learn = clip(X_learn, lb, ub)
            f_learn = func(X_learn); fes += 1
            if f_learn < fit[i]:
                X[i], fit[i] = X_learn, f_learn
                if f_learn < bv: bv, bp = f_learn, X_learn.copy()

            if fes >= max_fes: break

            # ── Reflection phase ─────────────────────────────────────
            X_ref = clip(X[i] + np.random.rand(dim) * (X[i] - worse_X), lb, ub)
            f_ref = func(X_ref); fes += 1
            if f_ref < fit[i]:
                X[i], fit[i] = X_ref, f_ref
                if f_ref < bv: bv, bp = f_ref, X_ref.copy()

        history.append((fes, bv))
    return {"best_val": bv, "best_pos": bp, "history": history}


# 5. CPO — Crested Porcupine Optimizer
def cpo(func, lb, ub, dim, max_fes, pop_size=50):
    lb, ub = np.array(lb, float), np.array(ub, float)
    X   = init_pop(pop_size, dim, lb, ub)
    fit = np.array([func(X[i]) for i in range(pop_size)])
    fes = pop_size
    bi  = int(np.argmin(fit)); bp = X[bi].copy(); bv = fit[bi]
    history = [(fes, bv)]
    T     = max(1, (max_fes - pop_size) // pop_size)
    N_min = max(10, pop_size // 5)

    for t in range(1, T+1):
        if fes >= max_fes: break
        norm_t = t / T

        # Cyclic population reduction
        cycle_len = max(1, T // 5)
        cycle_pos = ((t-1) % cycle_len) / max(1, cycle_len - 1)
        N_cur = max(N_min, int(pop_size - (pop_size - N_min) * cycle_pos))
        N_cur = min(N_cur, len(X))

        for i in range(N_cur):
            if fes >= max_fes: break
            r = np.random.rand()

            if r < 0.25:
                # Defense 1 — Sight (exploration)
                j     = np.random.randint(N_cur)
                X_new = clip(X[i] + np.random.rand(dim) * (X[j] - X[i]), lb, ub)
            elif r < 0.5:
                # Defense 2 — Sound (exploration)
                j     = np.random.randint(N_cur)
                alpha = np.random.rand(dim)
                X_new = clip(alpha * bp + (1 - alpha) * X[j], lb, ub)
            elif r < 0.75:
                # Defense 3 — Odor (exploitation)
                s     = np.random.randn(dim) * (1 - norm_t)
                X_new = clip(bp + s * (bp - X[i]), lb, ub)
            else:
                # Defense 4 — Physical attack (exploitation, Levy-enhanced)
                X_new = clip(bp + np.random.rand(dim) * levy_step(dim) * (bp - X[i]), lb, ub)

            f_new = func(X_new); fes += 1
            if f_new < fit[i]:
                X[i], fit[i] = X_new, f_new
                if f_new < bv: bv, bp, bi = f_new, X_new.copy(), i

        history.append((fes, bv))
    return {"best_val": bv, "best_pos": bp, "history": history}


# 6. HO — Hippopotamus Optimization Algorithm
def ho(func, lb, ub, dim, max_fes, pop_size=50):
    lb, ub = np.array(lb, float), np.array(ub, float)
    X   = init_pop(pop_size, dim, lb, ub)
    fit = np.array([func(X[i]) for i in range(pop_size)])
    fes = pop_size
    bi  = int(np.argmin(fit)); bp = X[bi].copy(); bv = fit[bi]
    history = [(fes, bv)]
    T = max(1, (max_fes - pop_size) // pop_size)

    for t in range(1, T+1):
        if fes >= max_fes: break
        norm_t = t / T

        for i in range(pop_size):
            if fes >= max_fes: break
            phase = np.random.rand()

            if phase < 1/3:
                # Phase 1 — River/pond positioning: follow dominant hippo
                I     = np.random.choice([1, 2])
                r     = np.random.rand(dim)
                X_new = clip(X[i] + r * (bp - I * X[i]), lb, ub)
            elif phase < 2/3:
                # Phase 2 — Predator defence: random territorial avoidance
                r1    = np.random.rand(dim)
                r2    = np.random.rand(dim)
                X_new = clip(X[i] + (lb + r1*(ub-lb))*(1-norm_t)
                             + r2*(bp - X[i])*norm_t, lb, ub)
            else:
                # Phase 3 — Territorial contest: Levy-enhanced dominance
                j = np.random.randint(pop_size)
                if fit[j] < fit[i]:
                    X_new = clip(X[i] + np.random.rand(dim) * levy_step(dim) * (X[j] - X[i]), lb, ub)
                else:
                    X_new = clip(X[i] + np.random.rand(dim) * levy_step(dim) * (X[i] - X[j]), lb, ub)

            f_new = func(X_new); fes += 1
            if f_new < fit[i]:
                X[i], fit[i] = X_new, f_new
                if f_new < bv: bv, bp, bi = f_new, X_new.copy(), i

        history.append((fes, bv))
    return {"best_val": bv, "best_pos": bp, "history": history}


# 7. WaOA — Walrus Optimization Algorithm
def waoa(func, lb, ub, dim, max_fes, pop_size=50):
    lb, ub = np.array(lb, float), np.array(ub, float)
    X   = init_pop(pop_size, dim, lb, ub)
    fit = np.array([func(X[i]) for i in range(pop_size)])
    fes = pop_size
    bi  = int(np.argmin(fit)); bp = X[bi].copy(); bv = fit[bi]
    history = [(fes, bv)]
    T = max(1, (max_fes - pop_size) // pop_size)

    for t in range(1, T+1):
        if fes >= max_fes: break
        norm_t = t / T

        for i in range(pop_size):
            if fes >= max_fes: break
            phase = np.random.rand()

            if phase < 1/3:
                # Phase 1 — Feeding (exploration): follow random peer
                j     = np.random.randint(pop_size)
                I     = np.random.choice([1, 2])
                X_new = clip(X[i] + np.random.rand(dim) * (X[j] - I*X[i]), lb, ub)
            elif phase < 2/3:
                # Phase 2 — Migration: cosine directional drift toward best
                phi   = 2 * np.pi * np.random.rand(dim)
                dist  = np.random.rand() * (1 - norm_t)
                X_new = clip(X[i] + dist*np.cos(phi)*(bp - X[i])
                             + np.random.rand(dim)*(ub-lb)*(1-norm_t), lb, ub)
            else:
                # Phase 3 — Evasion (exploitation): Levy spiral toward best
                X_new = clip(bp + np.random.rand(dim) * levy_step(dim) * (bp - X[i]), lb, ub)

            f_new = func(X_new); fes += 1
            if f_new < fit[i]:
                X[i], fit[i] = X_new, f_new
                if f_new < bv: bv, bp, bi = f_new, X_new.copy(), i

        history.append((fes, bv))
    return {"best_val": bv, "best_pos": bp, "history": history}


# 8. FOX — Fox Optimizer
def fox(func, lb, ub, dim, max_fes, pop_size=50):
    lb, ub = np.array(lb, float), np.array(ub, float)
    X   = init_pop(pop_size, dim, lb, ub)
    fit = np.array([func(X[i]) for i in range(pop_size)])
    fes = pop_size
    bi  = int(np.argmin(fit)); bp = X[bi].copy(); bv = fit[bi]
    history = [(fes, bv)]
    T  = max(1, (max_fes - pop_size) // pop_size)
    c1 = 0.18   # jump scaling parameter
    c2 = 0.82   # run scaling parameter

    for t in range(1, T+1):
        if fes >= max_fes: break
        a = 2 * (1 - t / T)   # linearly decreasing 2 → 0

        for i in range(pop_size):
            if fes >= max_fes: break

            if np.random.rand() > 0.5:
                # Phase 1 — Jumping (exploration): leap at random angles
                D     = np.abs(bp - X[i])
                tt    = np.random.rand(dim)
                dist  = 0.5 * D / (tt + 1e-12)
                X_new = clip(0.5 * (bp + dist), lb, ub)
            else:
                # Phase 2 — Running (exploitation): fast convergence
                r     = np.random.rand(dim)
                X_new = clip(bp - a * c1 * r * (c2 * bp - X[i]), lb, ub)

            f_new = func(X_new); fes += 1
            if f_new < fit[i]:
                X[i], fit[i] = X_new, f_new
                if f_new < bv: bv, bp, bi = f_new, X_new.copy(), i

        history.append((fes, bv))
    return {"best_val": bv, "best_pos": bp, "history": history}


# 9. GJO — Golden Jackal Optimizer
def gjo(func, lb, ub, dim, max_fes, pop_size=50):
    lb, ub = np.array(lb, float), np.array(ub, float)
    X   = init_pop(pop_size, dim, lb, ub)
    fit = np.array([func(X[i]) for i in range(pop_size)])
    fes = pop_size

    s      = np.argsort(fit)
    male   = X[s[0]].copy(); male_f   = fit[s[0]]
    female = X[s[1]].copy(); female_f = fit[s[1]]
    bv = male_f; bp = male.copy()
    history = [(fes, bv)]
    T = max(1, (max_fes - pop_size) // pop_size)

    for t in range(1, T+1):
        if fes >= max_fes: break
        E1 = 1.5 * (1 - t / T)   # prey energy, decays to 0

        for i in range(pop_size):
            if fes >= max_fes: break
            Esc = E1 * (2 * np.random.rand() - 1)
            r1  = np.random.rand(dim)
            r2  = np.random.rand(dim)

            D_male   = np.abs(r1 * male   - X[i])
            D_female = np.abs(r2 * female - X[i])

            X1 = male   - Esc * D_male
            X2 = female - Esc * D_female
            X_new = clip((X1 + X2) / 2, lb, ub)

            f_new = func(X_new); fes += 1
            if f_new < fit[i]:
                X[i], fit[i] = X_new, f_new
                if f_new < bv:
                    bv, bp = f_new, X_new.copy()
                    female, female_f = male.copy(), male_f
                    male,   male_f   = X_new.copy(), f_new
                elif f_new < female_f:
                    female, female_f = X_new.copy(), f_new

        history.append((fes, bv))
    return {"best_val": bv, "best_pos": bp, "history": history}


ALGORITHMS = {
    "MRPO" : mrpo,
    "RPO"  : rpo,
    "RIME" : rime,
    "GO"   : go,
    "CPO"  : cpo,
    "HO"   : ho,
    "WaOA" : waoa,
    "FOX"  : fox,
    "GJO"  : gjo,
}