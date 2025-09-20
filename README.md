# Alibaba DQN-LSTM Recommender System

Implementation of "Deep Reinforcement Learning-Based Recommender Algorithm Optimization and Intelligent Systems Construction for Business Data Analysis" using the Alibaba User Behaviour Dataset.

## Overview

This system combines Deep Q-Networks (DQN) with Long Short-Term Memory (LSTM) networks to create a reinforcement learning-based recommendation engine. The model learns optimal recommendation strategies by replacing traditional CNN components with LSTM layers for better sequential pattern modeling in user behavior data.

## Dataset

**Source**: Alibaba User Behaviour Dataset from Tianchi  
**URL**: https://tianchi.aliyun.com/dataset/dataDetail?dataId=81505  
**Format**: CSV with columns: `user_id`, `item_id`, `category_id`, `behavior_type`, `timestamp`  
**Size**: 86,953,525 records  
**Behaviors**: `pv` (pageview), `cart` (add-to-cart), `fav` (favorite), `buy` (purchase)

## Requirements

```
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

Install dependencies:
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

## Architecture

### Core Components

- **AlibabaDataProcessor**: Data loading, preprocessing, and profile generation
- **StateEncoder**: Converts user sequences to 20-dimensional state vectors
- **AlibabaDQNLSTMRecommender**: Bidirectional LSTM + Attention + Dueling DQN
- **AlibabaRecommenderAgent**: Training agent with priority experience replay
- **AlibabaRecommenderEnvironment**: User behavior simulation environment

### Model Features

- **Bidirectional LSTM**: Captures temporal patterns in both directions
- **Multi-head Attention**: Focuses on relevant sequence components
- **Dueling DQN**: Separates state value and action advantage estimation
- **Priority Replay Buffer**: Emphasizes informative experiences during training
- **Business-aware Rewards**: Weighted rewards for different user behaviors

## Quick Start

### 1. Data Preparation
Download the Alibaba dataset and place it in your project directory:
```bash
# Update DATA_PATH in the script to point to your dataset file
DATA_PATH = "alibaba_user_behavior.csv"
```

### 2. Training
```python
python alibaba_recommender.py
```

### 3. Configuration
Key parameters in the training function:
```python
NUM_EPISODES = 1500      # Training episodes
SAMPLE_SIZE = 50000      # Dataset sample size (for testing)
SEQUENCE_LENGTH = 20     # User sequence length
```

## State Representation

The system encodes user interactions into 20-dimensional vectors containing:

| Dimensions | Features |
|------------|----------|
| 0-3 | Item features (ID, popularity, conversion rates) |
| 4-5 | Category features (ID, purchase rate) |
| 6-9 | Behavior encoding (one-hot for pv/cart/fav/buy) |
| 10 | Temporal features (time since last action) |
| 11-13 | User context (purchase rate, activity level, category diversity) |
| 14 | Sequence position |
| 15-19 | Reserved for extensions |

## Reward System

| Behavior | Base Reward | Description |
|----------|-------------|-------------|
| `pv` (pageview) | 1.0 | Basic engagement |
| `cart` (add-to-cart) | 3.0 | Strong purchase intent |
| `fav` (favorite) | 2.5 | Interest indication |
| `buy` (purchase) | 10.0 | Target conversion |

Additional bonuses:
- **Diversity bonus**: +2.0 for recommending less popular items
- **Preference bonus**: +2.0 for category preference matching

## Training Process

1. **Data Processing**: Load and analyze user/item/category profiles
2. **Environment Setup**: Initialize user behavior simulation
3. **Agent Training**: DQN-LSTM with priority experience replay
4. **Evaluation**: Performance assessment on business metrics

### Training Output
```
Episode  100 | Avg Reward:  12.45 | CTR: 0.156 | Conversion: 0.023 | Diversity: 0.451
Episode  200 | Avg Reward:  18.72 | CTR: 0.203 | Conversion: 0.031 | Diversity: 0.467
...
```

## Evaluation Metrics

### Business Metrics
- **Click-Through Rate (CTR)**: Engagement measurement
- **Conversion Rate**: Purchase completion rate
- **Diversity Score**: Recommendation variety
- **Coverage**: Unique items recommended

### Technical Metrics
- **Average Reward**: Agent performance indicator
- **Training Loss**: Model convergence
- **Episode Length**: Interaction session duration

## Model Files

### Generated Files
- `best_alibaba_recommender.pth`: Best performing model
- `final_alibaba_recommender.pth`: Final trained model
- `alibaba_recommender_episode_X.pth`: Checkpoint saves
- `training_progress_episode_X.pkl`: Training metrics
- `alibaba_recommender_results.png`: Performance visualization

### Model Loading
```python
# Load trained model
agent.load_model("best_alibaba_recommender.pth")

# Make recommendations
state = state_encoder.encode_sequence(user_sequence, user_id)
action = agent.select_action(state, user_id, training=False)
recommended_item = idx_to_item[action]
```

## Performance Benchmarks

Expected results on Alibaba dataset:
- **CTR**: 3-8% (baseline: ~2%)
- **Conversion Rate**: 0.5-2% (baseline: ~0.1%)
- **Diversity Score**: 0.4-0.7
- **Training Episodes**: Convergence within 1000-1500 episodes

## Production Deployment

### Batch Inference
Pre-compute recommendations for active users:
```python
# Generate recommendations for user list
recommendations = {}
for user_id in active_users:
    user_sequence = get_user_sequence(user_id)
    state = state_encoder.encode_sequence(user_sequence, user_id)
    top_items = agent.get_top_k_recommendations(state, k=10)
    recommendations[user_id] = top_items
```

### Real-time Inference
On-demand recommendation serving:
```python
def get_realtime_recommendations(user_id, num_recs=10):
    user_sequence = get_recent_user_sequence(user_id)
    state = state_encoder.encode_sequence(user_sequence, user_id)
    return agent.select_top_actions(state, num_recs)
```

## Configuration Options

### Model Architecture
```python
AlibabaDQNLSTMRecommender(
    state_dim=20,           # State vector dimension
    action_dim=n_items,     # Number of items to recommend
    lstm_hidden_size=256,   # LSTM hidden units
    lstm_layers=3,          # LSTM layer count
    dropout_rate=0.3,       # Regularization
    attention_heads=8       # Multi-head attention
)
```

### Training Parameters
```python
AlibabaRecommenderAgent(
    learning_rate=0.0001,      # Adam learning rate
    gamma=0.99,                # Discount factor
    epsilon=1.0,               # Exploration rate
    epsilon_decay=0.9995,      # Exploration decay
    batch_size=64,             # Training batch size
    memory_size=50000,         # Replay buffer size
    priority_replay=True       # Use priority experience replay
)
```

## Troubleshooting

### Common Issues

**CUDA Memory Error**:
- Reduce `batch_size` or `lstm_hidden_size`
- Use `torch.cuda.empty_cache()` between episodes

**Low Performance**:
- Increase `sequence_length` for better context
- Adjust reward weights for business objectives
- Ensure sufficient training episodes

**Dataset Loading Error**:
- Verify file path and format
- Check column names match expected format
- Ensure sufficient disk space for processing

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Extensions

### Multi-Objective Optimization
Extend rewards to include multiple business objectives:
```python
def calculate_multi_objective_reward(self, action, behavior, context):
    ctr_reward = self.calculate_ctr_reward(action, behavior)
    diversity_reward = self.calculate_diversity_reward(action, context)
    revenue_reward = self.calculate_revenue_reward(action, behavior)
    return ctr_reward + diversity_reward + revenue_reward
```

### Real-time Learning
Implement online learning for production:
```python
def update_model_online(self, user_feedback):
    # Incremental learning from real user interactions
    state = self.get_current_state()
    action = self.last_recommendation
    reward = self.calculate_reward(user_feedback)
    next_state = self.get_next_state()
    
    self.agent.memory.push(state, action, reward, next_state, False)
    if len(self.agent.memory) >= self.agent.batch_size:
        self.agent.train()
```

## Citation

```bibtex
@inproceedings{dqn_lstm_recommender_2022,
    title={Deep Reinforcement Learning-Based Recommender Algorithm Optimization and Intelligent Systems Construction for Business Data Analysis},
    booktitle={2022 IEEE Asia-Pacific Conference on Image Processing, Electronics and Computers (IPEC)},
    year={2022},
    pages={},
    doi={10.1109/IPEC54454.2022.9777623},
    publisher={IEEE}
}
```

## License

This implementation is provided for research and educational purposes. The Alibaba dataset is subject to its original license terms from Tianchi platform.

## Contact

For technical issues or questions about the implementation, please create an issue in the repository or contact the development team.
