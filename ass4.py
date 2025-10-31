import matplotlib.pyplot as plt

# Function: y = (x + 5)^2
def f(x):
    return (x + 5)**2

# Derivative: f'(x) = 2(x + 5)
def df(x):
    return 2 * (x + 5)

# Gradient Descent parameters
x = 3               # Starting point
learning_rate = 0.1 # Step size
precision = 1e-6    # Convergence criteria
max_iters = 100     # Safety stop

# To visualize convergence
x_values = [x]
y_values = [f(x)]

for i in range(max_iters):
    prev_x = x
    grad = df(prev_x)
    x = prev_x - learning_rate * grad   # Update step
    x_values.append(x)
    y_values.append(f(x))
    
    # Stop when the change is very small
    if abs(x - prev_x) < precision:
        break

print(f"Local minima occurs at x = {x:.6f}")
print(f"Minimum value of function y = {f(x):.6f}")
print(f"Iterations: {i+1}")

# --- Visualization ---
plt.figure(figsize=(8,5))
plt.plot(x_values, y_values, 'bo-', label="Gradient Descent Path")
plt.title("Gradient Descent Convergence for y = (x + 5)^2")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()





