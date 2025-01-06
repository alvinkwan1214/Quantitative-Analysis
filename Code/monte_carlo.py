import numpy as np
import pandas as pd
import math
from scipy.stats import norm

N = norm.cdf


class option_pricing():
    def __init__(self):
        self.S = 79 #starting price
        self.K = 77 #strike price
        self.vol = 0.3 #volitility
        self.r = 0.01 #risk-free rate per annum
        self.N = 19 #number of steps 
        self.M = 1000000 # number of simulations
        self.T = 30/365 # days over the year

        self.deltat = self.T/self.N # time steps
        self.u = math.exp(self.vol*(self.deltat**(1/2))) #CCR up probability
        self.d = 1/self.u #CRR down probability
        self.y = 0 #yield per annum 
        self.p = (math.exp((self.r-self.y)*self.deltat) - self.d) / (self.u - self.d) #risk neutral probaility 
        self.q = 1 - self.p 

    def monte_carlo_stock_price(self):
        dt = self.T/self.N
        nudt = (self.r - 0.5*self.vol**2)*dt
        volsdt = self.vol*np.sqrt(dt)
        lnS = np.log(self.S)

        # Monte Carlo Method
        Z = np.random.normal(size=(self.N, self.M))
        delta_lnSt = nudt + volsdt*Z
        lnSt = lnS + np.cumsum(delta_lnSt, axis=0)
        lnSt = np.concatenate( (np.full(shape=(1, self.M), fill_value=lnS), lnSt ) )
        st= np.exp(lnSt) #find the underlying price 
        return st
    
    def monte_carlo_option_price(self, ST):
        CT = np.maximum(0, ST - self.K) # find the payoff of the call option
        C0 = np.exp(-self.r*self.T)*np.sum(CT[-1])/self.M #find the average pay off then scale it back to the current time-step
        sigma = np.sqrt( np.sum( (CT[-1] - C0)**2) / (self.M-1) )
        SE = sigma/np.sqrt(self.M)
        PT = np.maximum(0,self.K - ST) # find the payoff of the put option
        P0 = np.exp(-self.r*self.T)*np.sum(PT[-1])/self.M #find the average pay off then scale it back to the current time-step
        sigma = np.sqrt( np.sum( (PT[-1] - P0)**2) / (self.M-1) )
        SE = sigma/np.sqrt(self.M)

        return C0, P0
    
    def black_schole(self):
        d1 = (np.log(self.S/self.K) + (self.r + self.vol**2/2)*self.T) / (self.vol*np.sqrt(self.T))
        d2 = d1 - self.vol * np.sqrt(self.T)

        call = self.S * N(d1) - self.K * math.exp(-self.r * self.T) * N(d2)
        put = self.K * math.exp(-self.r * self.T) * N(-d2)  -  self.S * N(-d1)
        return call, put 


    def CRR_option_pricing(self):
        stock_price = np.zeros([self.N+1, self.N+1])
        for i in range(self.N + 1):
            for j in range(i + 1):
                stock_price[j, i] = self.S * (self.u ** (i - j)) * (self.d ** j)

        option_call = np.zeros([self.N + 1, self.N + 1])

        #call option
        option_call[:, self.N] = np.maximum(np.zeros(self.N + 1), (stock_price[:, self.N] - self.K))


        #put option
        option_put = np.zeros([self.N + 1, self.N + 1])
        option_put[:, self.N] = np.maximum(np.zeros(self.N + 1), (self.K -stock_price[:, self.N]))


        for i in range(self.N - 1, -1, -1):
            for j in range(0, i + 1):
                option_call[j, i] = (
                    1 /(math.exp((self.r-self.y)*self.deltat)) * (self.p * option_call[j, i + 1] + self.q * option_call[j + 1, i + 1])
                )
                option_put[j, i] = (
                    1 /(math.exp((self.r-self.y)*self.deltat)) * (self.p * option_put[j, i + 1] + self.q * option_put[j + 1, i + 1])
                )
        
        return option_call, option_put
    

if __name__ == "__main__":
    #check the type of options of each method
    pricing = option_pricing()
    stock_price = pricing.monte_carlo_stock_price()
    call_m, put_m = pricing.monte_carlo_option_price(stock_price)
    call_bs, put_bs  =  pricing.black_schole()
    call_b, put_b = pricing.CRR_option_pricing()
    call_b, put_b = call_b[0,0],put_b[0,0]

    result  = {"Monte Carlo": [call_m, put_m], "BS": [call_bs, put_bs], "Binomal_CRR":[call_b, put_b]}
    df = pd.DataFrame(result)
    df = df.rename(index={0: "Call", 1: "Put"})

    print(df.transpose())