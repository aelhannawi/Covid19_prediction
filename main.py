import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class COVID19Predictor:
    def __init__(self, total_population, initial_infected, contact_rate, recovery_rate):
        self.N = total_population
        self.I0 = initial_infected
        self.S0 = total_population - initial_infected
        self.R0 = 0
        self.beta = contact_rate
        self.gamma = recovery_rate

    def sir_model(self, y, t):
        """
        The SIR model equations
        """
        S, I, R = y
        dSdt = -self.beta * S * I / self.N
        dIdt = self.beta * S * I / self.N - self.gamma * I
        dRdt = self.gamma * I
        return dSdt, dIdt, dRdt

    def predict(self, days):
        """
        Predict the spread of COVID-19 over a specified number of days
        """
        # Time vector (in days)
        t = np.linspace(0, days, days)

        # Initial conditions vector
        y0 = self.S0, self.I0, self.R0

        # Integrate the SIR equations over the time grid, t
        result = odeint(self.sir_model, y0, t)
        S, I, R = result.T
        return t, S, I, R

    def plot(self, t, S, I, R):
        """
        Plot the results of the prediction
        """
        plt.figure(figsize=(10, 6))
        plt.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
        plt.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
        plt.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
        plt.xlabel('Time (days)')
        plt.ylabel('Number of people')
        plt.title('COVID-19 Prediction using SIR Model')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    # Parameters for the SIR model
    total_population = 1000
    initial_infected = 1
    contact_rate = 0.3
    recovery_rate = 0.1
    days_to_predict = 100

    # Create COVID19Predictor object
    predictor = COVID19Predictor(total_population, initial_infected, contact_rate, recovery_rate)

    # Predict the spread of COVID-19
    t, S, I, R = predictor.predict(days_to_predict)

    # Plot the results
    predictor.plot(t, S, I, R)

if __name__ == "__main__":
    main()
