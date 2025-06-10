import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# =====================================
# Configuración del problema
# =====================================
def objetivo(x):
    """Función objetivo a minimizar (ejemplo: función Rastrigin)."""
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Límites de las variables (para Rastrigin: [-5.12, 5.12])
lim_inf = -5.12
lim_sup = 5.12
n_variables = 2  # Número de variables de decisión

# =====================================
# Parámetros del Harmony Search
# =====================================
HMS = 20       # Tamaño de la memoria de armonías
HMCR = 0.95    # Tasa de consideración de memoria
PAR = 0.3      # Tasa de ajuste de paso
BW = 0.1       # Ancho de banda
MaxIter = 100  # Número máximo de iteraciones

# =====================================
# Inicialización de la memoria de armonías (HM)
# =====================================
HM = np.random.uniform(lim_inf, lim_sup, (HMS, n_variables))
HM_fitness = np.array([objetivo(sol) for sol in HM])

# =====================================
# Algoritmo Harmony Search (HS)
# =====================================
mejores_fitness = []  # Para almacenar el mejor fitness en cada iteración

def HS():
    global HM, HM_fitness
    
    for iter in range(MaxIter):
        # 1. Improvisar una nueva armonía
        nueva_armonia = np.zeros(n_variables)
        
        for i in range(n_variables):
            if np.random.rand() < HMCR:
                # Seleccionar de la memoria
                valor = HM[np.random.randint(0, HMS), i]
                
                if np.random.rand() < PAR:
                    # Ajuste de paso
                    valor += np.random.uniform(-1, 1) * BW
                    # Asegurar que esté dentro de los límites
                    valor = np.clip(valor, lim_inf, lim_sup)
            else:
                # Generar un valor aleatorio
                valor = np.random.uniform(lim_inf, lim_sup)
            
            nueva_armonia[i] = valor
        
        # 2. Evaluar la nueva armonía
        nuevo_fitness = objetivo(nueva_armonia)
        
        # 3. Actualizar la HM si la nueva armonía es mejor
        peor_idx = np.argmax(HM_fitness)
        if nuevo_fitness < HM_fitness[peor_idx]:
            HM[peor_idx] = nueva_armonia
            HM_fitness[peor_idx] = nuevo_fitness
        
        # Guardar el mejor fitness de esta iteración
        mejores_fitness.append(np.min(HM_fitness))
    
    # Devolver la mejor solución encontrada
    mejor_idx = np.argmin(HM_fitness)
    return HM[mejor_idx], HM_fitness[mejor_idx]

# Ejecutar el algoritmo
mejor_solucion, mejor_valor = HS()

# =====================================
# Visualización (2D o 3D)
# =====================================
# Crear una gráfica de convergencia
plt.figure(figsize=(10, 5))
plt.plot(mejores_fitness, 'r-', linewidth=2)
plt.title("Convergencia del Harmony Search")
plt.xlabel("Iteración")
plt.ylabel("Mejor Fitness")
plt.grid()
plt.show()

# Gráfica 3D de la función y las soluciones (solo para 2 variables)
if n_variables == 2:
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(lim_inf, lim_sup, 100)
    y = np.linspace(lim_inf, lim_sup, 100)
    X, Y = np.meshgrid(x, y)
    Z = objetivo([X, Y])

    # Guardar la evolución de HM para la animación
    HM_evolucion = []
    HM_fitness_evolucion = []

    # Reiniciar HM y fitness para la simulación animada
    HM_anim = np.random.uniform(lim_inf, lim_sup, (HMS, n_variables))
    HM_fitness_anim = np.array([objetivo(sol) for sol in HM_anim])

    for iter in range(MaxIter):
        HM_evolucion.append(HM_anim.copy())
        HM_fitness_evolucion.append(HM_fitness_anim.copy())

        nueva_armonia = np.zeros(n_variables)
        for i in range(n_variables):
            if np.random.rand() < HMCR:
                valor = HM_anim[np.random.randint(0, HMS), i]
                if np.random.rand() < PAR:
                    valor += np.random.uniform(-1, 1) * BW
                    valor = np.clip(valor, lim_inf, lim_sup)
            else:
                valor = np.random.uniform(lim_inf, lim_sup)
            nueva_armonia[i] = valor

        nuevo_fitness = objetivo(nueva_armonia)
        peor_idx = np.argmax(HM_fitness_anim)
        if nuevo_fitness < HM_fitness_anim[peor_idx]:
            HM_anim[peor_idx] = nueva_armonia
            HM_fitness_anim[peor_idx] = nuevo_fitness

    # Para la última iteración
    HM_evolucion.append(HM_anim.copy())
    HM_fitness_evolucion.append(HM_fitness_anim.copy())

    def update(frame):
        ax.clear()
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
        HM_frame = HM_evolucion[frame]
        fitness_frame = HM_fitness_evolucion[frame]
        mejor_idx = np.argmin(fitness_frame)
        peor_idx = np.argmax(fitness_frame)

        # Graficar todas las soluciones
        for i, sol in enumerate(HM_frame):
            color = 'red'
            size = 50
            if i == mejor_idx:
                color = 'green'
                size = 200
            elif i == peor_idx:
                color = 'orange'
                size = 100
            ax.scatter(sol[0], sol[1], fitness_frame[i], color=color, s=size)

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Fitness')
        ax.set_title(f"Iteración {frame+1} / {MaxIter}")

        # Leyenda manual
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Soluciones', markerfacecolor='red', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Mejor', markerfacecolor='green', markersize=15),
            Line2D([0], [0], marker='o', color='w', label='Peor', markerfacecolor='orange', markersize=10),
        ]
        ax.legend(handles=legend_elements, loc='upper left')

    ani = FuncAnimation(fig, update, frames=len(HM_evolucion), interval=200, repeat=False)
    plt.show()