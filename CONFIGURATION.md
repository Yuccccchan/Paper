# Configuration Files for TSN-WiFi Scheduling

This document provides configuration templates for different network scenarios.

## 1. Small Test Network (10 flows)

```yaml
# config/small_network.yaml
network:
  name: "small_test_network"
  cycle_time: 1000  # microseconds
  
topology:
  switches:
    - id: "s0"
      ports: 8
      processing_delay: 5  # microseconds
    - id: "s1"
      ports: 8
      processing_delay: 5
  
  end_stations:
    - "es0"
    - "es1"
    - "es2"
    - "es3"
  
  links:
    - src: "es0"
      dst: "s0"
      bandwidth: 1000  # Mbps
      latency: 5      # microseconds
    - src: "es1"
      dst: "s0"
      bandwidth: 1000
      latency: 5
    - src: "s0"
      dst: "s1"
      bandwidth: 1000
      latency: 10
    - src: "s1"
      dst: "s0"
      bandwidth: 1000
      latency: 10
    - src: "es2"
      dst: "s1"
      bandwidth: 1000
      latency: 5
    - src: "es3"
      dst: "s1"
      bandwidth: 1000
      latency: 5

flows:
  num_flows: 10
  generation_mode: "random"  # or "fixed"
  
  constraints:
    min_size: 64      # bytes
    max_size: 1500    # bytes (MTU)
    min_deadline: 400 # microseconds
    max_deadline: 900
    min_offset: 0
    max_offset: 400

wifi:
  bands:
    2.4GHz:
      frequency: 2.4
      bandwidth: 20    # MHz
      max_airtime: 1000  # microseconds per cycle
    5GHz:
      frequency: 5.0
      bandwidth: 80
      max_airtime: 1000
    6GHz:
      frequency: 6.0
      bandwidth: 160
      max_airtime: 1000
  
  channel_model:
    type: "rayleigh_fading"
    base_snr:
      2.4GHz: 25  # dB
      5GHz: 30
      6GHz: 35
    snr_std:
      2.4GHz: 5
      5GHz: 4
      6GHz: 3
```

## 2. Medium Industrial Network (50 flows)

```yaml
# config/industrial_network.yaml
network:
  name: "industrial_factory"
  cycle_time: 1000
  
topology:
  switches:
    - id: "core_s0"
      ports: 16
      processing_delay: 3
    - id: "core_s1"
      ports: 16
      processing_delay: 3
    - id: "edge_s0"
      ports: 8
      processing_delay: 5
    - id: "edge_s1"
      ports: 8
      processing_delay: 5
    - id: "edge_s2"
      ports: 8
      processing_delay: 5
    - id: "edge_s3"
      ports: 8
      processing_delay: 5
  
  end_stations:
    # Sensors and actuators
    - "sensor_0"
    - "sensor_1"
    - "sensor_2"
    - "sensor_3"
    - "sensor_4"
    - "actuator_0"
    - "actuator_1"
    - "actuator_2"
    # Controllers
    - "plc_0"
    - "plc_1"
    # AGVs
    - "agv_0"
    - "agv_1"
    - "agv_2"
  
  links:
    # Core backbone
    - {src: "core_s0", dst: "core_s1", bandwidth: 10000, latency: 5}
    - {src: "core_s1", dst: "core_s0", bandwidth: 10000, latency: 5}
    
    # Core to edge
    - {src: "core_s0", dst: "edge_s0", bandwidth: 1000, latency: 8}
    - {src: "core_s0", dst: "edge_s1", bandwidth: 1000, latency: 8}
    - {src: "core_s1", dst: "edge_s2", bandwidth: 1000, latency: 8}
    - {src: "core_s1", dst: "edge_s3", bandwidth: 1000, latency: 8}
    
    # Edge to devices (abbreviated for brevity)
    - {src: "sensor_0", dst: "edge_s0", bandwidth: 1000, latency: 5}
    - {src: "sensor_1", dst: "edge_s0", bandwidth: 1000, latency: 5}
    # ... more connections

flows:
  num_flows: 50
  generation_mode: "scenario"
  
  scenarios:
    periodic_control:
      count: 20
      size: 128
      period: 1000
      deadline: 500
      priority: "high"
    
    event_triggered:
      count: 15
      size: 512
      deadline: 700
      priority: "medium"
    
    monitoring:
      count: 15
      size: 1500
      deadline: 900
      priority: "low"

wifi:
  bands:
    2.4GHz:
      frequency: 2.4
      bandwidth: 40
      max_airtime: 1000
    5GHz:
      frequency: 5.0
      bandwidth: 160
      max_airtime: 1000
    6GHz:
      frequency: 6.0
      bandwidth: 320
      max_airtime: 1000
  
  channel_model:
    type: "industrial_environment"
    interference_level: "high"
    base_snr:
      2.4GHz: 20
      5GHz: 28
      6GHz: 33
```

## 3. Large Scale Network (100 flows)

```yaml
# config/large_network.yaml
network:
  name: "large_campus"
  cycle_time: 1000
  
topology:
  # Generated programmatically
  switches: 20
  end_stations: 40
  connectivity: "hierarchical"
  redundancy: "dual_path"

flows:
  num_flows: 100
  generation_mode: "mixed"
  distribution:
    control: 0.3
    monitoring: 0.4
    video: 0.2
    data: 0.1

wifi:
  # Full tri-band configuration
  bands:
    2.4GHz:
      frequency: 2.4
      bandwidth: 40
      max_airtime: 1000
    5GHz:
      frequency: 5.0
      bandwidth: 160
      max_airtime: 1000
    6GHz:
      frequency: 6.0
      bandwidth: 320
      max_airtime: 1000
```

## 4. Training Hyperparameters

```yaml
# config/training_params.yaml
training:
  num_episodes: 1000
  max_steps_per_episode: 100
  
  agent:
    type: "AttentionDDPG"
    
    actor:
      embed_dim: 128
      num_attention_heads: 8
      num_attention_layers: 3
      learning_rate: 0.0001
      
    critic:
      embed_dim: 128
      num_attention_heads: 8
      num_attention_layers: 3
      learning_rate: 0.001
    
    optimization:
      discount_factor: 0.99
      soft_update_tau: 0.001
      batch_size: 64
      replay_buffer_size: 100000
      
    exploration:
      initial_noise: 0.1
      final_noise: 0.01
      decay_rate: "linear"
      
  reward:
    weights:
      reliability: 100
      deadline_violation: -50
      target_bonus: 50
    target_reliability: 0.99
    
  logging:
    log_interval: 10
    save_interval: 100
    checkpoint_dir: "checkpoints/"
    
  early_stopping:
    enabled: true
    patience: 100
    min_delta: 0.01
```

## 5. Evaluation Configuration

```yaml
# config/evaluation_params.yaml
evaluation:
  num_episodes: 100
  
  metrics:
    - average_reliability
    - deadline_satisfaction_rate
    - resource_utilization
    - end_to_end_delay
    - load_variance
    
  comparison_baselines:
    - name: "Random"
      enabled: true
    - name: "SLCI"
      enabled: true
    - name: "Policy_Gradient"
      enabled: true
    - name: "MLP_DRL"
      enabled: true
    - name: "FlowWise_DRL"
      enabled: true
      
  visualization:
    generate_plots: true
    plot_types:
      - training_curves
      - reliability_comparison
      - gantt_chart
      - band_allocation
    output_dir: "results/"
    
  export:
    format: ["csv", "json"]
    include_raw_data: true
```

## 6. Example Python Configuration Loader

```python
# config/config_loader.py
import yaml
import os

class Config:
    """Configuration loader for TSN-WiFi scheduling"""
    
    def __init__(self, config_dir="config"):
        self.config_dir = config_dir
        self.network = None
        self.training = None
        self.evaluation = None
    
    def load(self, network_config="small_network.yaml",
             training_config="training_params.yaml",
             eval_config="evaluation_params.yaml"):
        """Load all configuration files"""
        
        # Load network configuration
        network_path = os.path.join(self.config_dir, network_config)
        with open(network_path, 'r') as f:
            self.network = yaml.safe_load(f)
        
        # Load training configuration
        training_path = os.path.join(self.config_dir, training_config)
        with open(training_path, 'r') as f:
            self.training = yaml.safe_load(f)
        
        # Load evaluation configuration
        eval_path = os.path.join(self.config_dir, eval_config)
        with open(eval_path, 'r') as f:
            self.evaluation = yaml.safe_load(f)
        
        return self
    
    def get_topology_config(self):
        """Get topology configuration in expected format"""
        return {
            'switches': self.network['topology']['switches'],
            'end_stations': self.network['topology']['end_stations'],
            'links': self.network['topology']['links']
        }
    
    def get_flow_config(self):
        """Get flow generation configuration"""
        return self.network['flows']
    
    def get_wifi_config(self):
        """Get WiFi configuration"""
        return self.network['wifi']
    
    def get_training_params(self):
        """Get training hyperparameters"""
        return self.training['training']
    
    def get_agent_config(self):
        """Get agent architecture configuration"""
        return self.training['training']['agent']
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 60)
        print("Configuration Summary")
        print("=" * 60)
        print(f"\nNetwork: {self.network['network']['name']}")
        print(f"Number of switches: {len(self.network['topology']['switches'])}")
        print(f"Number of end stations: {len(self.network['topology']['end_stations'])}")
        print(f"Number of flows: {self.network['flows']['num_flows']}")
        print(f"\nTraining episodes: {self.training['training']['num_episodes']}")
        print(f"Batch size: {self.training['training']['agent']['optimization']['batch_size']}")
        print(f"Actor LR: {self.training['training']['agent']['actor']['learning_rate']}")
        print(f"Critic LR: {self.training['training']['agent']['critic']['learning_rate']}")
        print("=" * 60)

# Usage example
if __name__ == '__main__':
    config = Config()
    config.load(
        network_config="small_network.yaml",
        training_config="training_params.yaml",
        eval_config="evaluation_params.yaml"
    )
    
    config.print_summary()
    
    # Access configurations
    topology = config.get_topology_config()
    agent_params = config.get_agent_config()
    
    print("\nTopology:", topology)
    print("\nAgent params:", agent_params)
```

## 7. Environment Variable Configuration

```bash
# .env
# Environment variables for TSN-WiFi scheduling

# Paths
PROJECT_ROOT=/path/to/project
DATA_DIR=${PROJECT_ROOT}/data
CONFIG_DIR=${PROJECT_ROOT}/config
CHECKPOINT_DIR=${PROJECT_ROOT}/checkpoints
RESULTS_DIR=${PROJECT_ROOT}/results

# Training
NUM_EPISODES=1000
BATCH_SIZE=64
LEARNING_RATE_ACTOR=0.0001
LEARNING_RATE_CRITIC=0.001

# Device
DEVICE=cuda  # or cpu
NUM_WORKERS=4

# Logging
LOG_LEVEL=INFO
WANDB_PROJECT=tsn-wifi-scheduling
WANDB_ENTITY=your-team

# Reproducibility
RANDOM_SEED=42
```

## 8. Docker Configuration

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories
RUN mkdir -p checkpoints results logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cuda

# Default command
CMD ["python", "train.py", "--config", "config/industrial_network.yaml"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  trainer:
    build: .
    container_name: tsn-wifi-trainer
    volumes:
      - ./config:/app/config
      - ./checkpoints:/app/checkpoints
      - ./results:/app/results
    environment:
      - DEVICE=cuda
      - NUM_EPISODES=1000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: python train.py --config config/industrial_network.yaml
  
  evaluator:
    build: .
    container_name: tsn-wifi-evaluator
    volumes:
      - ./config:/app/config
      - ./checkpoints:/app/checkpoints
      - ./results:/app/results
    environment:
      - DEVICE=cpu
    command: python evaluate.py --checkpoint checkpoints/best_model.pt
```

## Usage Examples

### Load configuration in Python

```python
from config.config_loader import Config

# Load configuration
config = Config().load(
    network_config="industrial_network.yaml",
    training_config="training_params.yaml"
)

# Use in training
from environment.tsn_wifi_env import TSNWiFiEnv
from drl_agent.ddpg_agent import AttentionDDPGAgent

env = TSNWiFiEnv(
    topology_config=config.get_topology_config(),
    num_flows=config.network['flows']['num_flows']
)

agent_config = config.get_agent_config()
agent = AttentionDDPGAgent(
    flow_feature_dim=15,
    action_dim=4,
    lr_actor=agent_config['actor']['learning_rate'],
    lr_critic=agent_config['critic']['learning_rate']
)
```

### Run with specific configuration

```bash
# Train with small network
python train.py --config config/small_network.yaml

# Train with industrial network
python train.py --config config/industrial_network.yaml

# Evaluate trained model
python evaluate.py --config config/industrial_network.yaml --checkpoint checkpoints/best_model.pt

# Run with Docker
docker-compose up trainer
```

### Sweep hyperparameters

```python
# hyperparameter_sweep.py
import itertools
from config.config_loader import Config

# Define sweep parameters
learning_rates = [1e-3, 1e-4, 1e-5]
batch_sizes = [32, 64, 128]
embed_dims = [64, 128, 256]

# Grid search
for lr, bs, ed in itertools.product(learning_rates, batch_sizes, embed_dims):
    print(f"Training with lr={lr}, batch_size={bs}, embed_dim={ed}")
    
    # Load base configuration
    config = Config().load()
    
    # Modify parameters
    config.training['training']['agent']['actor']['learning_rate'] = lr
    config.training['training']['agent']['optimization']['batch_size'] = bs
    config.training['training']['agent']['actor']['embed_dim'] = ed
    
    # Train model
    # ... training code ...
```

## Configuration Best Practices

1. **Version Control**: Keep configurations in version control
2. **Environment Specific**: Use different configs for dev/test/prod
3. **Validation**: Validate configurations before use
4. **Documentation**: Document all configuration parameters
5. **Defaults**: Provide sensible defaults
6. **Modularity**: Separate concerns (network, training, evaluation)
7. **Reproducibility**: Include random seeds and versions

