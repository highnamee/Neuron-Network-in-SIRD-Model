from numpy import linspace, zeros, asarray


class EulerSIRD:
    def __init__(self, beta, gamma, mu, U_0):
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.U_0 = U_0
        self.dt = 1                            # 1 day
        self.T = 1

    def ode_FE(self):
        S = self.U_0[0]
        I = self.U_0[1]
        R = self.U_0[2]
        D = self.U_0[3]
        N = S + I + R + D
        result = [S-self.beta*S*I, I + self.beta*S*I -
                  (self.gamma+self.mu)*I, R + self.gamma*I, D + self.mu*I]
        return result
