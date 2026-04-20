import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
  
  @staticmethod
  def plot_decision_boundary(X, y, model):

      plt.figure(figsize=(8, 6))
      plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Classe 0 ')
      plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Classe 1')

      # Equação: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b) / w2
      x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
      
      if model.w[1] != 0:
          x1_values = np.linspace(x1_min, x1_max, 100)
          x2_values = -(model.w[0] * x1_values + model.b) / model.w[1]
          plt.plot(x1_values, x2_values, 'k--', label='Hiperplane')
      
      plt.xlim(x1_min, x1_max)
      plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
      plt.xlabel('X1')
      plt.ylabel('X2')
      plt.title('Separação de Classes pelo Perceptron')
      plt.legend()
      plt.grid(True, linestyle='--', alpha=0.6)
      plt.show()