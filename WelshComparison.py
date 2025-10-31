import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

# --- General Settings ---
st.set_page_config(page_title="Neural Net Learner - Welsh vs Sigmoid Comparison", layout="wide")
sns.set_style("whitegrid")

# --- Initialize Session State ---
if "history_sigmoid" not in st.session_state:
    st.session_state.history_sigmoid = []
if "history_welsh" not in st.session_state:
    st.session_state.history_welsh = []
if "r2_history_sigmoid" not in st.session_state:
    st.session_state.r2_history_sigmoid = []
if "r2_history_welsh" not in st.session_state:
    st.session_state.r2_history_welsh = []
if "trained" not in st.session_state:
    st.session_state.trained = False
if "final_r2_sigmoid" not in st.session_state:
    st.session_state.final_r2_sigmoid = 0
if "final_r2_welsh" not in st.session_state:
    st.session_state.final_r2_welsh = 0

# --- Realistic Plant Growth Model ---
def realistic_plant_growth(sunlight, water, temperature=25):
    """
    A more realistic plant growth model based on:
    - Sunlight follows a bell curve (plants need sufficient light but can get burned)
    - Water shows diminishing returns
    - Temperature has an optimal range
    - All factors interact with each other
    """
    # Normalize inputs to 0-1 range
    sun_norm = sunlight / 10
    water_norm = water / 5
    temp_norm = (temperature - 5) / 30  # Normalize to 0-1 for range 5-35°C
    
    # Sunlight response (bell curve with optimal around 6-7 hours)
    sun_effect = 4 * sun_norm * (1 - sun_norm) ** 0.5
    
    # Water response (sigmoid-like with diminishing returns)
    water_effect = 1 - np.exp(-3 * water_norm)
    
    # Temperature effect (bell curve centered around 23°C)
    temp_effect = np.exp(-((temp_norm - 0.6) ** 2) / 0.1)
    
    # Interaction effects
    sun_water_interaction = 0.2 * sun_norm * water_norm
    temp_water_interaction = -0.1 * temp_norm * (1 - water_norm)
    sun_temp_interaction = -0.2 * (sun_norm - 0.6) * (temp_norm - 0.6)
    
    # Final growth rate calculation
    growth_rate = 0.3 + (
        0.4 * sun_effect + 
        0.3 * water_effect + 
        0.2 * temp_effect +
        sun_water_interaction +
        temp_water_interaction +
        sun_temp_interaction
    )
    
    # Constrain to 0-1 range
    return np.clip(growth_rate, 0, 1)

# --- Data Generation ---
@st.cache_data
def generate_data(n_samples=3000, seed=0):
    np.random.seed(seed)
    
    # Create grid points for stratified sampling
    grid_size = int(np.sqrt(n_samples * 5))  # Generate more points for stratification
    sunlight = np.linspace(0, 10, grid_size)
    water = np.linspace(0, 5, grid_size)
    temperatures = np.linspace(5, 35, max(5, grid_size // 8))  # Fewer temperature points
    
    # Create combinations
    combinations = []
    for s in sunlight:
        for w in water:
            for t in np.random.choice(temperatures, 3):  # Select random temperatures to reduce combinations
                combinations.append([s, w, t])
    
    # Sample from combinations if too many
    if len(combinations) > n_samples * 5:
        sample_indices = np.random.choice(len(combinations), n_samples * 5, replace=False)
        combinations = [combinations[i] for i in sample_indices]
    
    # Create dataframe and calculate growth
    df_samples = pd.DataFrame(combinations, columns=['sunlight', 'water', 'temperature'])
    df_samples['growth'] = df_samples.apply(
        lambda row: realistic_plant_growth(row['sunlight'], row['water'], row['temperature']), 
        axis=1
    )
    
    # Normalize inputs
    df_samples['sunlight_norm'] = df_samples['sunlight'] / 10
    df_samples['water_norm'] = df_samples['water'] / 5
    df_samples['temp_norm'] = (df_samples['temperature'] - 5) / 30
    
    # Stratify samples
    bins = np.linspace(0, 1, 11)  # 10 bins (0.0–0.1, ..., 0.9–1.0)
    df_samples['growth_bin'] = pd.cut(df_samples['growth'], bins)
    
    # Sample equally from each bin
    balanced_dfs = []
    samples_per_bin = n_samples // 10
    
    for bin_range in df_samples['growth_bin'].unique():
        bin_df = df_samples[df_samples['growth_bin'] == bin_range]
        if len(bin_df) >= samples_per_bin:
            sampled = bin_df.sample(samples_per_bin, random_state=seed)
            balanced_dfs.append(sampled)
        else:
            balanced_dfs.append(bin_df)  # Take all if not enough samples
    
    df_final = pd.concat(balanced_dfs).sample(frac=1, random_state=seed).drop(columns=['growth_bin'])
    
    return df_final.reset_index(drop=True)

# --- Enhanced Neural Network Class with Multiple Activation Functions ---
class EnhancedNeuralNet:
    def __init__(self, input_size, hidden_layers, output_size, lr=0.01, seed=42, activation_type="sigmoid"):
        """
        Initialize a neural network with multiple hidden layers
        
        Parameters:
        - input_size: number of input features
        - hidden_layers: list of sizes for each hidden layer
        - output_size: number of output units
        - lr: learning rate
        - seed: random seed for weight initialization
        - activation_type: "sigmoid" or "welsh"
        """
        np.random.seed(seed)
        self.lr = lr
        self.n_layers = len(hidden_layers) + 1  # hidden layers + output layer
        self.activation_type = activation_type
        self.hidden_activations = [activation_type] * len(hidden_layers)
        self.output_activation = "sigmoid"
        
        # Initialize weights and biases for all layers
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_layers[0]) * np.sqrt(2. / input_size))
        self.biases.append(np.zeros((1, hidden_layers[0])))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.weights.append(np.random.randn(hidden_layers[i], hidden_layers[i+1]) * 
                               np.sqrt(2. / hidden_layers[i]))
            self.biases.append(np.zeros((1, hidden_layers[i+1])))
        
        # Last hidden layer to output
        self.weights.append(np.random.randn(hidden_layers[-1], output_size) * 
                           np.sqrt(2. / hidden_layers[-1]))
        self.biases.append(np.zeros((1, output_size)))

    def welsh_function(self, z):
        """Welsh function: f(x) = x * e^(-x^2/2)"""
        return z * np.exp(-z**2/2)
    
    def welsh_derivative(self, z):
        """Derivative of Welsh function: f'(x) = e^(-x^2/2) * (1 - x^2)"""
        return np.exp(-z**2/2) * (1 - z**2)

    def activation(self, z, func="sigmoid"):
        if func == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif func == "relu":
            return np.maximum(0, z)
        elif func == "welsh":
            return self.welsh_function(z)
        else:
            return z  # linear

    def activation_derivative(self, z, output=None, func="sigmoid"):
        if func == "sigmoid":
            if output is not None:
                return output * (1 - output)
            else:
                sig = self.activation(z, func="sigmoid")
                return sig * (1 - sig)
        elif func == "relu":
            return (z > 0).astype(float)
        elif func == "welsh":
            return self.welsh_derivative(z)
        else:
            return np.ones_like(z)  # linear

    def forward(self, X):
        self.layer_inputs = []  # Store inputs to each layer (needed for backprop)
        self.layer_outputs = []  # Store outputs of each layer
        self.layer_z_values = []  # Store z values before activation
        
        # Input layer (no activation)
        current_input = X
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            self.layer_inputs.append(current_input)
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.layer_z_values.append(z)
            a = self.activation(z, func=self.hidden_activations[i])
            self.layer_outputs.append(a)
            current_input = a
        
        # Output layer
        self.layer_inputs.append(current_input)
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        self.layer_z_values.append(z_output)
        output = self.activation(z_output, func=self.output_activation)
        self.layer_outputs.append(output)
        
        return output

    def backward(self, X, y):
        m = len(y)
        output = self.layer_outputs[-1]
        
        # Output layer error
        d_layer = (output - y) * self.activation_derivative(
            self.layer_z_values[-1], output, func=self.output_activation)
        deltas = [d_layer]
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 1, 0, -1):
            if self.activation_type == "welsh":
                # For Welsh function, use z values for derivative calculation
                d_layer = np.dot(deltas[0], self.weights[i].T) * self.activation_derivative(
                    self.layer_z_values[i-1], func=self.hidden_activations[i-1])
            else:
                # For other activations, can use output values
                d_layer = np.dot(deltas[0], self.weights[i].T) * self.activation_derivative(
                    self.layer_z_values[i-1], self.layer_outputs[i-1], func=self.hidden_activations[i-1])
            deltas.insert(0, d_layer)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            layer_input = X if i == 0 else self.layer_outputs[i-1]
            self.weights[i] -= self.lr * np.dot(layer_input.T, deltas[i]) / m
            self.biases[i] -= self.lr * np.sum(deltas[i], axis=0, keepdims=True) / m

    def train_one_epoch(self, X, y):
        self.forward(X)
        self.backward(X, y)
        loss = np.mean((self.layer_outputs[-1] - y) ** 2)
        return loss

    def predict(self, X):
        return self.forward(X)

# --- Visualization Function ---
def draw_network_comparison(nn_sigmoid, nn_welsh, ax1, ax2):
    """Draw both networks side by side for comparison"""
    for nn, ax, title in [(nn_sigmoid, ax1, "Sigmoid Network"), (nn_welsh, ax2, "Welsh Network")]:
        # Count the total number of nodes
        input_size = nn.weights[0].shape[0]
        hidden_layers = [w.shape[1] for w in nn.weights[:-1]]
        output_size = nn.weights[-1].shape[1]
        total_layers = len(hidden_layers) + 2  # +2 for input and output layers
        
        G = nx.DiGraph()
        pos = {}
        layer_nodes = {i: [] for i in range(total_layers)}
        
        # Add input nodes
        for i in range(input_size):
            node = f"I{i}"
            G.add_node(node)
            pos[node] = (0, -i * 2)
            layer_nodes[0].append(node)
        
        # Add hidden layer nodes
        for layer_idx, layer_size in enumerate(hidden_layers):
            for i in range(layer_size):
                node = f"H{layer_idx}_{i}"
                G.add_node(node)
                pos[node] = (layer_idx + 1, -i * 2)
                layer_nodes[layer_idx + 1].append(node)
        
        # Add output nodes
        for i in range(output_size):
            node = f"O{i}"
            G.add_node(node)
            pos[node] = (total_layers - 1, -i * 2)
            layer_nodes[total_layers - 1].append(node)
        
        # Add edges between layers (simplified for complex networks)
        if len(hidden_layers) <= 2:  # Only show edges for simple networks
            # Add edges between input and first hidden layer
            for i in range(input_size):
                for j in range(hidden_layers[0]):
                    G.add_edge(f"I{i}", f"H0_{j}", weight=nn.weights[0][i, j])
            
            # Add edges between hidden layers
            for layer_idx in range(len(hidden_layers) - 1):
                for i in range(hidden_layers[layer_idx]):
                    for j in range(hidden_layers[layer_idx + 1]):
                        G.add_edge(f"H{layer_idx}_{i}", f"H{layer_idx+1}_{j}", 
                                   weight=nn.weights[layer_idx + 1][i, j])
            
            # Add edges between last hidden layer and output
            for i in range(hidden_layers[-1]):
                for j in range(output_size):
                    G.add_edge(f"H{len(hidden_layers)-1}_{i}", f"O{j}", 
                               weight=nn.weights[-1][i, j])
            
            # Edge weights
            edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
            norm = plt.Normalize(-1, 1)
            cmap = plt.cm.seismic
            
            # Draw network
            nx.draw(G, pos, ax=ax, with_labels=True, edge_color=edge_colors, edge_cmap=cmap,
                    node_color='lightblue', node_size=800, width=1.5, arrows=True,
                    font_size=6)
        else:
            # Simplified view for complex networks
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', 
                    node_size=800, width=1, arrows=True, font_size=6, edge_color='gray', alpha=0.5)
            
        ax.set_title(f"{title} Architecture")

# --- App Interface ---
st.title("🌱 Neural Network Comparison: Welsh vs Sigmoid Activation")
st.markdown("Compare the performance of neural networks using Welsh function vs Sigmoid activation on realistic plant growth data.")

# Add information about Welsh function
with st.expander("📚 About the Welsh and Sigmoid Functions"):
    st.markdown("""
    **Welsh Function**: f(x) = x * e^(-x²/2)
    **Sigmoid Function**: f(x) = 1 / (1 + e⁻ˣ)
    
    **Properties**:
    **Welsh Function**:
    - **Range**: Approximately (-0.43, 0.43)
    - **Derivative**: f'(x) = e^(-x²/2) * (1 - x²)
    - **Advantages**: 
      - Non-monotonic derivative (can help with gradient flow)
      - Smoother than ReLU
      - More balanced gradient distribution
      - Can potentially avoid some vanishing gradient issues
      
    **Sigmoid Function**:
    - **Range**: (0, 1)
    - **Derivative**: f'(x) = f(x)(1 - f(x))
    - **Advantages**:
      - Smooth gradient
      - Clear probability interpretation
      - Stable and bounded output
      - Widely used in binary classification
    """)
    
    # Plot Welsh function vs Sigmoid for comparison
    x = np.linspace(-5, 5, 1000)
    welsh_y = x * np.exp(-x**2/2)
    sigmoid_y = 1 / (1 + np.exp(-x))
    
    fig_act, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Activation functions
    ax1.plot(x, welsh_y, 'r-', linewidth=2, label='Welsh: x*e^(-x²/2)')
    ax1.plot(x, sigmoid_y, 'b-', linewidth=2, label='Sigmoid: 1/(1+e⁻ˣ)')
    ax1.set_xlabel('Input (x)')
    ax1.set_ylabel('Output')
    ax1.set_title('Activation Functions Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Derivatives
    welsh_deriv = np.exp(-x**2/2) * (1 - x**2)
    sigmoid_deriv = sigmoid_y * (1 - sigmoid_y)
    
    ax2.plot(x, welsh_deriv, 'r-', linewidth=2, label="Welsh derivative")
    ax2.plot(x, sigmoid_deriv, 'b-', linewidth=2, label="Sigmoid derivative")
    ax2.set_xlabel('Input (x)')
    ax2.set_ylabel('Derivative')
    ax2.set_title('Derivatives Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig_act)

with st.sidebar:
    st.header("🧠 Network Config")
    
    # Network architecture settings
    st.subheader("Architecture")
    num_hidden_layers = st.slider("Number of Hidden Layers", 1, 4, 2)
    
    # Create layer size inputs
    hidden_layers = []
    for i in range(num_hidden_layers):
        layer_size = st.slider(f"Neurons in Hidden Layer {i+1}", 3, 20, 8)
        hidden_layers.append(layer_size)
    
    # Input features selection
    st.subheader("Input Features")
    use_temperature = st.checkbox("Include Temperature as Input", True)
    input_size = 3 if use_temperature else 2
    
    # Training parameters
    st.subheader("Training Parameters")
    learning_rate = st.slider("Learning Rate", 0.001, 0.05, 0.01, step=0.001)
    epochs = st.slider("Auto-Train Epochs", 100, 2000, 500, step=100)
    seed = st.number_input("Random Seed", value=42)
    
    if st.button("🔄 Initialize Both Networks"):
        st.session_state.nn_sigmoid = EnhancedNeuralNet(input_size, hidden_layers, 1, lr=learning_rate, seed=seed, activation_type="sigmoid")
        st.session_state.nn_welsh = EnhancedNeuralNet(input_size, hidden_layers, 1, lr=learning_rate, seed=seed, activation_type="welsh")
        st.session_state.history_sigmoid = []
        st.session_state.history_welsh = []
        st.session_state.r2_history_sigmoid = []
        st.session_state.r2_history_welsh = []
        st.session_state.epoch = 0
        st.session_state.trained = False
        st.session_state.use_temperature = use_temperature
        st.success(f"Both networks initialized with {len(hidden_layers)} hidden layers!")

# --- Initialization ---
if "nn_sigmoid" not in st.session_state:
    # Default initialization
    st.session_state.nn_sigmoid = EnhancedNeuralNet(3, [8, 5], 1, activation_type="sigmoid")
    st.session_state.nn_welsh = EnhancedNeuralNet(3, [8, 5], 1, activation_type="welsh")
    st.session_state.history_sigmoid = []
    st.session_state.history_welsh = []
    st.session_state.r2_history_sigmoid = []
    st.session_state.r2_history_welsh = []
    st.session_state.epoch = 0
    st.session_state.trained = False
    st.session_state.use_temperature = True

# --- Data ---
data = generate_data()

# Select features based on user choice
if st.session_state.use_temperature:
    X = data[["sunlight_norm", "water_norm", "temp_norm"]].values
else:
    X = data[["sunlight_norm", "water_norm"]].values

y = data["growth"].values.reshape(-1, 1)

# --- Training Comparison ---
st.subheader("📊 Train Both Networks")

col_train, col_metrics = st.columns([1, 1])

with col_train:
    if st.button("👣 Step Train (1 Epoch Both)"):
        loss_sigmoid = st.session_state.nn_sigmoid.train_one_epoch(X, y)
        loss_welsh = st.session_state.nn_welsh.train_one_epoch(X, y)
        
        st.session_state.history_sigmoid.append(loss_sigmoid)
        st.session_state.history_welsh.append(loss_welsh)
        
        # Calculate R² scores
        pred_sigmoid = st.session_state.nn_sigmoid.predict(X)
        pred_welsh = st.session_state.nn_welsh.predict(X)
        
        r2_sigmoid = 1 - np.sum((y - pred_sigmoid) ** 2) / np.sum((y - y.mean()) ** 2)
        r2_welsh = 1 - np.sum((y - pred_welsh) ** 2) / np.sum((y - y.mean()) ** 2)
        
        st.session_state.r2_history_sigmoid.append(r2_sigmoid)
        st.session_state.r2_history_welsh.append(r2_welsh)
        
        st.session_state.epoch += 1
        st.session_state.trained = True
        st.rerun()
    
    if st.button("🚀 Train Both Automatically"):
        with st.spinner("Training both networks in progress..."):
            progress_bar = st.progress(0)
            for i in range(epochs):
                loss_tanh = st.session_state.nn_tanh.train_one_epoch(X, y)
                loss_welsh = st.session_state.nn_welsh.train_one_epoch(X, y)
                
                st.session_state.history_tanh.append(loss_tanh)
                st.session_state.history_welsh.append(loss_welsh)
                
                # Calculate R² scores every 10 epochs to avoid slowdown
                if i % 10 == 0 or i == epochs - 1:
                    pred_tanh = st.session_state.nn_tanh.predict(X)
                    pred_welsh = st.session_state.nn_welsh.predict(X)
                    
                    r2_tanh = 1 - np.sum((y - pred_tanh) ** 2) / np.sum((y - y.mean()) ** 2)
                    r2_welsh = 1 - np.sum((y - pred_welsh) ** 2) / np.sum((y - y.mean()) ** 2)
                    
                    # Fill in intermediate values
                    for _ in range(len(st.session_state.r2_history_tanh), len(st.session_state.history_tanh)):
                        st.session_state.r2_history_tanh.append(r2_tanh)
                        st.session_state.r2_history_welsh.append(r2_welsh)
                
                st.session_state.epoch += 1
                progress_bar.progress((i + 1) / epochs)
            
            st.session_state.trained = True
        st.success(f"Auto Training Complete! ✅ Trained both networks for {epochs} more epochs.")
        st.rerun()

with col_metrics:
    if st.session_state.trained:
        pred_sigmoid = st.session_state.nn_sigmoid.predict(X)
        pred_welsh = st.session_state.nn_welsh.predict(X)
        
        r2_sigmoid = 1 - np.sum((y - pred_sigmoid) ** 2) / np.sum((y - y.mean()) ** 2)
        r2_welsh = 1 - np.sum((y - pred_welsh) ** 2) / np.sum((y - y.mean()) ** 2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sigmoid R² Score", f"{r2_sigmoid:.4f}")
            st.metric("Sigmoid Final Loss", f"{st.session_state.history_sigmoid[-1]:.6f}")
        with col2:
            st.metric("Welsh R² Score", f"{r2_welsh:.4f}")
            st.metric("Welsh Final Loss", f"{st.session_state.history_welsh[-1]:.6f}")
        
        # Winner indicator
        if r2_tanh > r2_welsh:
            st.success("🏆 Tanh is currently performing better!")
        elif r2_welsh > r2_tanh:
            st.success("🏆 Welsh is currently performing better!")
        else:
            st.info("🤝 Both networks are performing equally!")
        
        st.metric("Current Epoch", st.session_state.epoch)

# --- Loss Comparison Plots ---
if st.session_state.history_tanh and st.session_state.history_welsh:
    st.subheader("📈 Training Progress Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
        epochs_range = range(len(st.session_state.history_tanh))
        ax_loss.plot(epochs_range, st.session_state.history_tanh, 'b-', linewidth=2, label="Tanh Network", alpha=0.8)
        ax_loss.plot(epochs_range, st.session_state.history_welsh, 'r-', linewidth=2, label="Welsh Network", alpha=0.8)
        ax_loss.set_title("Loss Comparison Over Time")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("MSE Loss")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        st.pyplot(fig_loss)
    
    with col2:
        if len(st.session_state.r2_history_tanh) > 0:
            fig_r2, ax_r2 = plt.subplots(figsize=(8, 5))
            epochs_range = range(len(st.session_state.r2_history_tanh))
            ax_r2.plot(epochs_range, st.session_state.r2_history_tanh, 'b-', linewidth=2, label="Tanh Network", alpha=0.8)
            ax_r2.plot(epochs_range, st.session_state.r2_history_welsh, 'r-', linewidth=2, label="Welsh Network", alpha=0.8)
            ax_r2.set_title("R² Score Comparison Over Time")
            ax_r2.set_xlabel("Epoch")
            ax_r2.set_ylabel("R² Score")
            ax_r2.set_ylim(0, 1)
            ax_r2.legend()
            ax_r2.grid(True, alpha=0.3)
            st.pyplot(fig_r2)

# --- Network Architecture Visualization ---
if st.session_state.trained and len(st.session_state.nn_sigmoid.weights) <= 3:
    st.subheader("🧠 Network Architecture Comparison")
    fig_nets, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    draw_network_comparison(st.session_state.nn_sigmoid, st.session_state.nn_welsh, ax1, ax2)
    st.pyplot(fig_nets)

# --- Prediction Interface ---
st.subheader("🔮 Compare Predictions")
sun = st.slider("Sunlight (hours)", 0.0, 10.0, 5.0)
water = st.slider("Water (liters)", 0.0, 5.0, 2.5)

# Add temperature input if enabled
if st.session_state.use_temperature:
    temperature = st.slider("Temperature (°C)", 5.0, 35.0, 25.0)
    input_val = np.array([[sun / 10, water / 5, (temperature - 5) / 30]])
else:
    temperature = 25  # Default
    input_val = np.array([[sun / 10, water / 5]])

if st.session_state.trained:
    pred_sigmoid = st.session_state.nn_sigmoid.predict(input_val)[0][0]
    pred_welsh = st.session_state.nn_welsh.predict(input_val)[0][0]
    
    # Get the actual value from the realistic model
    true_val = realistic_plant_growth(sun, water, temperature)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"🌿 Tanh Prediction: **{pred_tanh:.4f}**")
    with col2:
        st.info(f"🌿 Welsh Prediction: **{pred_welsh:.4f}**")
    with col3:
        st.warning(f"🔬 True Value: **{true_val:.4f}**")
    
    # Show which prediction is closer to truth
    tanh_error = abs(pred_tanh - true_val)
    welsh_error = abs(pred_welsh - true_val)
    
    if tanh_error < welsh_error:
        st.success(f"🎯 Tanh is closer to truth (error: {tanh_error:.4f} vs {welsh_error:.4f})")
    elif welsh_error < tanh_error:
        st.success(f"🎯 Welsh is closer to truth (error: {welsh_error:.4f} vs {tanh_error:.4f})")
    else:
        st.info(f"🤝 Both predictions are equally close to truth (error: {tanh_error:.4f})")
else:
    st.warning("Train the networks to see predictions.")

# --- Additional Comparison Visualizations ---
if st.session_state.trained:
    st.subheader("📈 Detailed Performance Analysis")

    pred_tanh_all = st.session_state.nn_tanh.predict(X)
    pred_welsh_all = st.session_state.nn_welsh.predict(X)
    
    # Calculate various metrics
    r2_tanh = 1 - np.sum((y - pred_tanh_all) ** 2) / np.sum((y - y.mean()) ** 2)
    r2_welsh = 1 - np.sum((y - pred_welsh_all) ** 2) / np.sum((y - y.mean()) ** 2)
    
    mae_tanh = np.mean(np.abs(y - pred_tanh_all))
    mae_welsh = np.mean(np.abs(y - pred_welsh_all))
    
    rmse_tanh = np.sqrt(np.mean((y - pred_tanh_all) ** 2))
    rmse_welsh = np.sqrt(np.mean((y - pred_welsh_all) ** 2))

    # 1. Predicted vs Actual Comparison
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Tanh predictions
    ax1.scatter(y, pred_tanh_all, alpha=0.6, color="blue", s=20)
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
    ax1.set_xlabel("Actual Growth")
    ax1.set_ylabel("Predicted Growth")
    ax1.set_title(f"Tanh Network (R² = {r2_tanh:.4f})")
    ax1.grid(True, alpha=0.3)
    
    # Welsh predictions
    ax2.scatter(y, pred_welsh_all, alpha=0.6, color="red", s=20)
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
    ax2.set_xlabel("Actual Growth")
    ax2.set_ylabel("Predicted Growth")
    ax2.set_title(f"Welsh Network (R² = {r2_welsh:.4f})")
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig1)

    # 2. Error Distribution Comparison
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    errors_tanh = (pred_tanh_all - y).flatten()
    errors_welsh = (pred_welsh_all - y).flatten()
    
    # Error histograms
    ax1.hist(errors_tanh, bins=30, alpha=0.7, color='blue', label=f'Tanh (std={np.std(errors_tanh):.4f})')
    ax1.hist(errors_welsh, bins=30, alpha=0.7, color='red', label=f'Welsh (std={np.std(errors_welsh):.4f})')
    ax1.set_xlabel("Prediction Error")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Error Distribution Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error vs actual values
    ax2.scatter(y, errors_tanh, alpha=0.5, color='blue', s=15, label='Tanh')
    ax2.scatter(y, errors_welsh, alpha=0.5, color='red', s=15, label='Welsh')
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.set_xlabel("Actual Growth")
    ax2.set_ylabel("Prediction Error")
    ax2.set_title("Error vs Actual Values")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig2)

    # 3. Performance Metrics Summary
    st.subheader("📊 Performance Metrics Summary")
    
    metrics_df = pd.DataFrame({
        'Metric': ['R² Score', 'MAE', 'RMSE', 'Final Loss'],
        'Tanh Network': [f"{r2_tanh:.6f}", f"{mae_tanh:.6f}", f"{rmse_tanh:.6f}", f"{st.session_state.history_tanh[-1]:.6f}"],
        'Welsh Network': [f"{r2_welsh:.6f}", f"{mae_welsh:.6f}", f"{rmse_welsh:.6f}", f"{st.session_state.history_welsh[-1]:.6f}"],
        'Winner': [
            'Tanh' if r2_tanh > r2_welsh else 'Welsh' if r2_welsh > r2_tanh else 'Tie',
            'Tanh' if mae_tanh < mae_welsh else 'Welsh' if mae_welsh < mae_tanh else 'Tie',
            'Tanh' if rmse_tanh < rmse_welsh else 'Welsh' if rmse_welsh < rmse_tanh else 'Tie',
            'Tanh' if st.session_state.history_tanh[-1] < st.session_state.history_welsh[-1] else 'Welsh' if st.session_state.history_welsh[-1] < st.session_state.history_tanh[-1] else 'Tie'
        ]
    })
    
    st.dataframe(metrics_df, use_container_width=True)

    # 4. Learning Curve Analysis
    if len(st.session_state.history_tanh) > 50:  # Only show if we have enough data points
        st.subheader("📈 Learning Curve Analysis")
        
        # Calculate moving averages for smoother curves
        window_size = max(10, len(st.session_state.history_tanh) // 20)
        
        def moving_average(data, window):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        tanh_smooth = moving_average(st.session_state.history_tanh, window_size)
        welsh_smooth = moving_average(st.session_state.history_welsh, window_size)
        
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Raw loss curves
        epochs_range = range(len(st.session_state.history_tanh))
        ax1.plot(epochs_range, st.session_state.history_tanh, 'b-', alpha=0.3, linewidth=1)
        ax1.plot(epochs_range, st.session_state.history_welsh, 'r-', alpha=0.3, linewidth=1)
        
        # Smoothed curves
        smooth_epochs = range(window_size-1, len(st.session_state.history_tanh))
        ax1.plot(smooth_epochs, tanh_smooth, 'b-', linewidth=2, label='Tanh (smoothed)')
        ax1.plot(smooth_epochs, welsh_smooth, 'r-', linewidth=2, label='Welsh (smoothed)')
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Learning Curves (Raw + Smoothed)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for better visualization
        
        # Learning rate analysis (loss improvement rate)
        if len(st.session_state.history_tanh) > 20:
            tanh_improvement = np.diff(st.session_state.history_tanh[-100:])  # Last 100 epochs
            welsh_improvement = np.diff(st.session_state.history_welsh[-100:])
            
            ax2.plot(tanh_improvement, 'b-', alpha=0.7, label='Tanh improvement rate')
            ax2.plot(welsh_improvement, 'r-', alpha=0.7, label='Welsh improvement rate')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_xlabel("Recent Epochs")
            ax2.set_ylabel("Loss Change")
            ax2.set_title("Recent Learning Progress")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig3)
        
        # Learning insights
        tanh_final_improvement = np.mean(st.session_state.history_tanh[-10:]) - np.mean(st.session_state.history_tanh[-20:-10])
        welsh_final_improvement = np.mean(st.session_state.history_welsh[-10:]) - np.mean(st.session_state.history_welsh[-20:-10])
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Tanh Learning Status:**\n- Final loss: {st.session_state.history_tanh[-1]:.6f}\n- Recent improvement: {tanh_final_improvement:.6f}\n- {'Still learning' if tanh_final_improvement < -0.001 else 'Converged' if abs(tanh_final_improvement) < 0.001 else 'Possibly overfitting'}")
        
        with col2:
            st.info(f"**Welsh Learning Status:**\n- Final loss: {st.session_state.history_welsh[-1]:.6f}\n- Recent improvement: {welsh_final_improvement:.6f}\n- {'Still learning' if welsh_final_improvement < -0.001 else 'Converged' if abs(welsh_final_improvement) < 0.001 else 'Possibly overfitting'}")

    # 5. Decision Surface Comparison (2D case)
    if not st.session_state.use_temperature:
        st.subheader("🌐 Decision Surface Comparison")
        
        try:
            # Create a mesh grid for visualization
            resolution = 30
            sun_range = np.linspace(0, 1, resolution)
            water_range = np.linspace(0, 1, resolution)
            sun_grid, water_grid = np.meshgrid(sun_range, water_range)
            
            # Reshape for prediction
            X_mesh = np.c_[sun_grid.ravel(), water_grid.ravel()]
            
            # Make predictions
            Z_tanh = st.session_state.nn_tanh.predict(X_mesh).reshape(sun_grid.shape)
            Z_welsh = st.session_state.nn_welsh.predict(X_mesh).reshape(sun_grid.shape)
            
            # Plot comparison
            fig4, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            
            # Tanh surface
            contour1 = ax1.contourf(sun_grid, water_grid, Z_tanh, cmap='viridis', alpha=0.8, levels=15)
            ax1.scatter([input_val[0, 0]], [input_val[0, 1]], c='red', marker='*', s=200, 
                       edgecolor='white', linewidth=1.5)
            ax1.set_xlabel('Normalized Sunlight')
            ax1.set_ylabel('Normalized Water')
            ax1.set_title('Tanh Network Decision Surface')
            plt.colorbar(contour1, ax=ax1)
            
            # Welsh surface
            contour2 = ax2.contourf(sun_grid, water_grid, Z_welsh, cmap='viridis', alpha=0.8, levels=15)
            ax2.scatter([input_val[0, 0]], [input_val[0, 1]], c='red', marker='*', s=200, 
                       edgecolor='white', linewidth=1.5)
            ax2.set_xlabel('Normalized Sunlight')
            ax2.set_ylabel('Normalized Water')
            ax2.set_title('Welsh Network Decision Surface')
            plt.colorbar(contour2, ax=ax2)
            
            # Difference map
            Z_diff = Z_welsh - Z_tanh
            contour3 = ax3.contourf(sun_grid, water_grid, Z_diff, cmap='RdBu_r', alpha=0.8, levels=15)
            ax3.scatter([input_val[0, 0]], [input_val[0, 1]], c='black', marker='*', s=200, 
                       edgecolor='white', linewidth=1.5)
            ax3.set_xlabel('Normalized Sunlight')
            ax3.set_ylabel('Normalized Water')
            ax3.set_title('Difference (Welsh - Tanh)')
            plt.colorbar(contour3, ax=ax3)
            
            st.pyplot(fig4)
            
        except Exception as e:
            st.error(f"Could not generate decision surface comparison: {str(e)}")

    # 6. Feature Sensitivity Comparison (3D case)
    elif st.session_state.use_temperature:
        st.subheader("🔍 Feature Sensitivity Comparison")
        
        try:
            base_input = np.array([[sun/10, water/5, (temperature-5)/30]])
            
            # Test sensitivity to each input
            n_points = 25
            delta_range = 0.3
            
            # Create variable ranges
            sun_range = np.linspace(max(0, base_input[0, 0]-delta_range), min(1, base_input[0, 0]+delta_range), n_points)
            water_range = np.linspace(max(0, base_input[0, 1]-delta_range), min(1, base_input[0, 1]+delta_range), n_points)
            temp_range = np.linspace(max(0, base_input[0, 2]-delta_range), min(1, base_input[0, 2]+delta_range), n_points)
            
            # Get predictions for both networks
            def get_sensitivity_data(network, feature_idx, feature_range):
                preds = []
                for val in feature_range:
                    test_input = base_input.copy()
                    test_input[0, feature_idx] = val
                    preds.append(network.predict(test_input)[0][0])
                return preds
            
            # Calculate sensitivities
            sun_preds_sigmoid = get_sensitivity_data(st.session_state.nn_sigmoid, 0, sun_range)
            sun_preds_welsh = get_sensitivity_data(st.session_state.nn_welsh, 0, sun_range)
            
            water_preds_sigmoid = get_sensitivity_data(st.session_state.nn_sigmoid, 1, water_range)
            water_preds_welsh = get_sensitivity_data(st.session_state.nn_welsh, 1, water_range)
            
            temp_preds_sigmoid = get_sensitivity_data(st.session_state.nn_sigmoid, 2, temp_range)
            temp_preds_welsh = get_sensitivity_data(st.session_state.nn_welsh, 2, temp_range)
            
            # Plot sensitivity comparison
            fig5, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            
            # Sunlight sensitivity
            ax1.plot(sun_range*10, sun_preds_sigmoid, 'b-', linewidth=2, label='Sigmoid', alpha=0.8)
            ax1.plot(sun_range*10, sun_preds_welsh, 'r-', linewidth=2, label='Welsh', alpha=0.8)
            ax1.axvline(sun, color='gray', linestyle='--', alpha=0.5)
            ax1.set_xlabel('Sunlight (hours)')
            ax1.set_ylabel('Predicted Growth')
            ax1.set_title('Sunlight Sensitivity')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Water sensitivity
            ax2.plot(water_range*5, water_preds_sigmoid, 'b-', linewidth=2, label='Sigmoid', alpha=0.8)
            ax2.plot(water_range*5, water_preds_welsh, 'r-', linewidth=2, label='Welsh', alpha=0.8)
            ax2.axvline(water, color='gray', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Water (liters)')
            ax2.set_ylabel('Predicted Growth')
            ax2.set_title('Water Sensitivity')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Temperature sensitivity
            ax3.plot(temp_range*30+5, temp_preds_sigmoid, 'b-', linewidth=2, label='Sigmoid', alpha=0.8)
            ax3.plot(temp_range*30+5, temp_preds_welsh, 'r-', linewidth=2, label='Welsh', alpha=0.8)
            ax3.axvline(temperature, color='gray', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Temperature (°C)')
            ax3.set_ylabel('Predicted Growth')
            ax3.set_title('Temperature Sensitivity')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            st.pyplot(fig5)
            
            # Calculate and display sensitivity scores
            sun_sens_sigmoid = np.std(sun_preds_sigmoid)
            sun_sens_welsh = np.std(sun_preds_welsh)
            water_sens_sigmoid = np.std(water_preds_sigmoid)
            water_sens_welsh = np.std(water_preds_welsh)
            temp_sens_sigmoid = np.std(temp_preds_sigmoid)
            temp_sens_welsh = np.std(temp_preds_welsh)
            
            sensitivity_df = pd.DataFrame({
                'Feature': ['Sunlight', 'Water', 'Temperature'],
                'Sigmoid Sensitivity': [f"{sun_sens_sigmoid:.4f}", f"{water_sens_sigmoid:.4f}", f"{temp_sens_sigmoid:.4f}"],
                'Welsh Sensitivity': [f"{sun_sens_welsh:.4f}", f"{water_sens_welsh:.4f}", f"{temp_sens_welsh:.4f}"],
                'More Sensitive': [
                    'Sigmoid' if sun_sens_sigmoid > sun_sens_welsh else 'Welsh',
                    'Sigmoid' if water_sens_sigmoid > water_sens_welsh else 'Welsh',
                    'Sigmoid' if temp_sens_sigmoid > temp_sens_welsh else 'Welsh'
                ]
            })
            
            st.dataframe(sensitivity_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Could not perform sensitivity analysis: {str(e)}")

# --- Conclusion and Insights ---
if st.session_state.trained and len(st.session_state.history_tanh) > 10:
    st.subheader("🎯 Key Insights & Conclusions")
    
    # Calculate final performance metrics
    final_r2_tanh = st.session_state.r2_history_tanh[-1] if st.session_state.r2_history_tanh else 0
    final_r2_welsh = st.session_state.r2_history_welsh[-1] if st.session_state.r2_history_welsh else 0
    
    # Determine overall winner
    tanh_wins = 0
    welsh_wins = 0
    
    # R² comparison
    if final_r2_tanh > final_r2_welsh:
        tanh_wins += 1
    elif final_r2_welsh > final_r2_tanh:
        welsh_wins += 1
    
    # Loss comparison
    if st.session_state.history_tanh[-1] < st.session_state.history_welsh[-1]:
        tanh_wins += 1
    elif st.session_state.history_welsh[-1] < st.session_state.history_tanh[-1]:
        welsh_wins += 1
    
    # Convergence speed (which reached 90% of final performance first)
    tanh_target = st.session_state.history_tanh[-1] * 1.1
    welsh_target = st.session_state.history_welsh[-1] * 1.1
    
    tanh_convergence = next((i for i, loss in enumerate(st.session_state.history_tanh) if loss <= tanh_target), len(st.session_state.history_tanh))
    welsh_convergence = next((i for i, loss in enumerate(st.session_state.history_welsh) if loss <= welsh_target), len(st.session_state.history_welsh))
    
    if tanh_convergence < welsh_convergence:
        tanh_wins += 1
    elif welsh_convergence < tanh_convergence:
        welsh_wins += 1
    
    # Display results
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if tanh_wins > welsh_wins:
            st.success("🏆 **Overall Winner: Tanh Activation**")
            st.write(f"Tanh won in {tanh_wins} out of 3 key metrics")
        elif welsh_wins > tanh_wins:
            st.success("🏆 **Overall Winner: Welsh Activation**")
            st.write(f"Welsh won in {welsh_wins} out of 3 key metrics")
        else:
            st.info("🤝 **Result: It's a Tie!**")
            st.write("Both activation functions performed equally well")
    
    # Detailed insights
    st.markdown("### 📝 Detailed Analysis:")
    
    insights = []
    
    if final_r2_tanh > final_r2_welsh + 0.01:
        insights.append("• **Tanh** achieved better overall accuracy (R² score)")
    elif final_r2_welsh > final_r2_tanh + 0.01:
        insights.append("• **Welsh** achieved better overall accuracy (R² score)")
    else:
        insights.append("• Both networks achieved similar accuracy levels")
    
    if st.session_state.history_tanh[-1] < st.session_state.history_welsh[-1] * 0.9:
        insights.append("• **Tanh** converged to a lower final loss")
    elif st.session_state.history_welsh[-1] < st.session_state.history_tanh[-1] * 0.9:
        insights.append("• **Welsh** converged to a lower final loss")
    else:
        insights.append("• Both networks achieved similar final loss values")
    
    if tanh_convergence < welsh_convergence * 0.8:
        insights.append("• **Tanh** converged faster during training")
    elif welsh_convergence < tanh_convergence * 0.8:
        insights.append("• **Welsh** converged faster during training")
    else:
        insights.append("• Both networks had similar convergence speeds")
    
    # Add learning behavior insights
    if len(st.session_state.history_tanh) > 50:
        tanh_stability = np.std(st.session_state.history_tanh[-20:])
        welsh_stability = np.std(st.session_state.history_welsh[-20:])
        
        if tanh_stability < welsh_stability * 0.8:
            insights.append("• **Tanh** showed more stable training in recent epochs")
        elif welsh_stability < tanh_stability * 0.8:
            insights.append("• **Welsh** showed more stable training in recent epochs")
    
    for insight in insights:
        st.write(insight)
    
    # Recommendations
    st.markdown("### 💡 Recommendations:")
    
    if tanh_wins > welsh_wins:
        st.write("Based on this experiment, **Tanh activation** appears to be better suited for this plant growth prediction task.")
    elif welsh_wins > tanh_wins:
        st.write("Based on this experiment, **Welsh activation** appears to be better suited for this plant growth prediction task.")
    else:
        st.write("Both activation functions performed similarly. The choice between them may depend on other factors like computational efficiency or specific use case requirements.")
    
    st.write("**Note:** Results may vary with different:")
    st.write("- Network architectures (number of layers, neurons)")
    st.write("- Learning rates and training parameters")
    st.write("- Dataset characteristics and size")
    st.write("- Random initialization seeds")
    
    st.write("Try experimenting with different configurations to see how the activation functions perform under various conditions!")

st.markdown("---")
st.markdown("**Enhanced Neural Network Comparison Tool** - Compare Welsh vs Tanh activation functions and discover which works better for your specific problem! 🚀")