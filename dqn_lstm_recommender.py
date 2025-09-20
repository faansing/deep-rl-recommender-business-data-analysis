import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Any, Optional
import torch.nn.functional as F
from datetime import datetime, timedelta
import pickle
import json

class AlibabaDataProcessor:
    """
    Data processor specifically for Alibaba User Behaviour Dataset
    Dataset structure: user_id, item_id, category_id, behavior_type, timestamp
    Behavior types: 'pv' (pageview), 'cart' (add-to-cart), 'fav' (favorite), 'buy' (purchase)
    """
    
    def __init__(self, data_path: str = None, sample_size: int = None):
        self.behavior_mapping = {
            'pv': 0,      # Page view (browse)
            'cart': 1,    # Add to cart
            'fav': 2,     # Add to favorite
            'buy': 3      # Purchase (target behavior)
        }
        self.reverse_behavior_mapping = {v: k for k, v in self.behavior_mapping.items()}
        
        # Reward weights for different behaviors (higher for more valuable actions)
        self.behavior_rewards = {
            'pv': 1.0,    # Base reward for engagement
            'cart': 3.0,  # Higher intent
            'fav': 2.5,   # Moderate intent
            'buy': 10.0   # Highest value - target behavior
        }
        
        self.data = None
        self.user_profiles = {}
        self.item_profiles = {}
        self.category_profiles = {}
        
        if data_path:
            self.load_and_preprocess_data(data_path, sample_size)
    
    def load_and_preprocess_data(self, data_path: str, sample_size: int = None):
        """Load and preprocess the Alibaba dataset"""
        print("Loading Alibaba User Behaviour Dataset...")
        
        # Load data
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        else:
            # Assuming space-separated or tab-separated format
            column_names = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']
            self.data = pd.read_csv(data_path, sep='\t', names=column_names, header=None)
        
        # Sample data if specified
        if sample_size and len(self.data) > sample_size:
            self.data = self.data.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} records from the dataset")
        
        print(f"Loaded {len(self.data)} records")
        print(f"Unique users: {self.data['user_id'].nunique()}")
        print(f"Unique items: {self.data['item_id'].nunique()}")
        print(f"Unique categories: {self.data['category_id'].nunique()}")
        
        # Convert timestamp to datetime
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        # Map behavior types to numbers
        self.data['behavior_code'] = self.data['behavior_type'].map(self.behavior_mapping)
        
        # Sort by user and timestamp
        self.data = self.data.sort_values(['user_id', 'timestamp'])
        
        # Create user profiles
        self._create_user_profiles()
        self._create_item_profiles()
        self._create_category_profiles()
        
        print("Data preprocessing completed!")
        print(f"Behavior distribution:")
        print(self.data['behavior_type'].value_counts())
    
    def _create_user_profiles(self):
        """Create comprehensive user profiles"""
        print("Creating user profiles...")
        
        for user_id in self.data['user_id'].unique():
            user_data = self.data[self.data['user_id'] == user_id]
            
            # Basic statistics
            total_actions = len(user_data)
            unique_items = user_data['item_id'].nunique()
            unique_categories = user_data['category_id'].nunique()
            
            # Behavior patterns
            behavior_counts = user_data['behavior_type'].value_counts()
            behavior_ratios = behavior_counts / total_actions
            
            # Temporal patterns
            time_span = (user_data['timestamp'].max() - user_data['timestamp'].min()).days
            avg_actions_per_day = total_actions / max(1, time_span)
            
            # Purchase patterns
            purchases = user_data[user_data['behavior_type'] == 'buy']
            purchase_rate = len(purchases) / total_actions if total_actions > 0 else 0
            
            # Category preferences
            category_prefs = user_data.groupby('category_id').size().sort_values(ascending=False)
            top_categories = category_prefs.head(5).index.tolist()
            
            self.user_profiles[user_id] = {
                'total_actions': total_actions,
                'unique_items': unique_items,
                'unique_categories': unique_categories,
                'behavior_ratios': behavior_ratios.to_dict(),
                'time_span_days': time_span,
                'avg_actions_per_day': avg_actions_per_day,
                'purchase_rate': purchase_rate,
                'top_categories': top_categories,
                'last_activity': user_data['timestamp'].max()
            }
    
    def _create_item_profiles(self):
        """Create item profiles with popularity and interaction patterns"""
        print("Creating item profiles...")
        
        for item_id in self.data['item_id'].unique():
            item_data = self.data[self.data['item_id'] == item_id]
            
            # Popularity metrics
            total_interactions = len(item_data)
            unique_users = item_data['user_id'].nunique()
            
            # Behavior breakdown
            behavior_counts = item_data['behavior_type'].value_counts()
            
            # Conversion metrics
            views = behavior_counts.get('pv', 0)
            carts = behavior_counts.get('cart', 0)
            favs = behavior_counts.get('fav', 0)
            buys = behavior_counts.get('buy', 0)
            
            cart_conversion = carts / views if views > 0 else 0
            buy_conversion = buys / views if views > 0 else 0
            
            # Category information
            category_id = item_data['category_id'].iloc[0]
            
            self.item_profiles[item_id] = {
                'total_interactions': total_interactions,
                'unique_users': unique_users,
                'category_id': category_id,
                'views': views,
                'carts': carts,
                'favs': favs,
                'buys': buys,
                'cart_conversion_rate': cart_conversion,
                'buy_conversion_rate': buy_conversion,
                'popularity_score': total_interactions / self.data['item_id'].nunique()
            }
    
    def _create_category_profiles(self):
        """Create category profiles"""
        print("Creating category profiles...")
        
        for category_id in self.data['category_id'].unique():
            category_data = self.data[self.data['category_id'] == category_id]
            
            total_interactions = len(category_data)
            unique_users = category_data['user_id'].nunique()
            unique_items = category_data['item_id'].nunique()
            
            # Behavior patterns in this category
            behavior_counts = category_data['behavior_type'].value_counts()
            behavior_ratios = behavior_counts / total_interactions
            
            self.category_profiles[category_id] = {
                'total_interactions': total_interactions,
                'unique_users': unique_users,
                'unique_items': unique_items,
                'behavior_ratios': behavior_ratios.to_dict(),
                'avg_interactions_per_item': total_interactions / unique_items
            }
    
    def get_user_sequence(self, user_id: int, sequence_length: int = 20, 
                         end_time: datetime = None) -> List[Dict]:
        """Get user's recent interaction sequence"""
        user_data = self.data[self.data['user_id'] == user_id]
        
        if end_time:
            user_data = user_data[user_data['timestamp'] <= end_time]
        
        # Get most recent interactions
        recent_data = user_data.tail(sequence_length)
        
        sequence = []
        for _, row in recent_data.iterrows():
            sequence.append({
                'item_id': row['item_id'],
                'category_id': row['category_id'],
                'behavior_type': row['behavior_type'],
                'behavior_code': row['behavior_code'],
                'timestamp': row['timestamp']
            })
        
        return sequence
    
    def calculate_reward(self, action_item_id: int, actual_behavior: str, 
                        user_id: int, context: Dict = None) -> float:
        """Calculate reward based on user action and business objectives"""
        base_reward = self.behavior_rewards[actual_behavior]
        
        # Bonus for successful recommendations
        if actual_behavior in ['cart', 'fav', 'buy']:
            # Additional reward based on item popularity (recommend diverse items)
            item_popularity = self.item_profiles[action_item_id]['popularity_score']
            diversity_bonus = max(0, 1 - item_popularity) * 2  # Bonus for less popular items
            
            # User preference alignment bonus
            user_profile = self.user_profiles.get(user_id, {})
            item_category = self.item_profiles[action_item_id]['category_id']
            
            if item_category in user_profile.get('top_categories', []):
                preference_bonus = 2.0
            else:
                preference_bonus = 0.5  # Exploration bonus
            
            total_reward = base_reward + diversity_bonus + preference_bonus
        else:
            # Small reward for page views (engagement)
            total_reward = base_reward
        
        # Normalize reward to [-5, 15] range for stable training
        return min(15, max(-5, total_reward))


class StateEncoder:
    """
    Encode user interaction sequences and context into state vectors
    for the DQN-LSTM model
    """
    
    def __init__(self, data_processor: AlibabaDataProcessor, 
                 embedding_dim: int = 64, max_items: int = 50000, 
                 max_categories: int = 5000):
        self.data_processor = data_processor
        self.embedding_dim = embedding_dim
        self.max_items = max_items
        self.max_categories = max_categories
        
        # Create mappings for items and categories
        unique_items = list(data_processor.data['item_id'].unique())
        unique_categories = list(data_processor.data['category_id'].unique())
        
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.category_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_categories)}
        
        self.n_items = len(unique_items)
        self.n_categories = len(unique_categories)
        self.n_behaviors = len(data_processor.behavior_mapping)
        
        print(f"State encoder initialized: {self.n_items} items, {self.n_categories} categories")
    
    def encode_sequence(self, sequence: List[Dict], user_id: int, 
                       sequence_length: int = 20) -> np.ndarray:
        """
        Encode user interaction sequence into state vector
        Returns: (sequence_length, state_dim) array
        """
        # State dimensions:
        # - Item embedding: item_id (normalized) + item features
        # - Category embedding: category_id (normalized) + category features  
        # - Behavior: one-hot encoding of behavior type
        # - Temporal: time since last interaction
        # - User context: user profile features
        
        state_dim = 20  # Fixed state dimension
        state = np.zeros((sequence_length, state_dim))
        
        user_profile = self.data_processor.user_profiles.get(user_id, {})
        
        # Pad sequence if shorter than required
        padded_sequence = sequence + [None] * (sequence_length - len(sequence))
        padded_sequence = padded_sequence[-sequence_length:]  # Take last sequence_length items
        
        for i, interaction in enumerate(padded_sequence):
            if interaction is None:
                continue
                
            # Item features
            item_id = interaction['item_id']
            item_idx = self.item_to_idx.get(item_id, 0)
            item_profile = self.data_processor.item_profiles.get(item_id, {})
            
            state[i, 0] = item_idx / self.n_items  # Normalized item ID
            state[i, 1] = item_profile.get('popularity_score', 0)
            state[i, 2] = item_profile.get('cart_conversion_rate', 0)
            state[i, 3] = item_profile.get('buy_conversion_rate', 0)
            
            # Category features
            category_id = interaction['category_id']
            category_idx = self.category_to_idx.get(category_id, 0)
            category_profile = self.data_processor.category_profiles.get(category_id, {})
            
            state[i, 4] = category_idx / self.n_categories  # Normalized category ID
            state[i, 5] = category_profile.get('behavior_ratios', {}).get('buy', 0)
            
            # Behavior encoding (one-hot)
            behavior_code = interaction['behavior_code']
            state[i, 6 + behavior_code] = 1.0  # Positions 6-9 for behavior one-hot
            
            # Temporal features
            if i > 0 and padded_sequence[i-1] is not None:
                time_diff = (interaction['timestamp'] - padded_sequence[i-1]['timestamp']).seconds / 3600
                state[i, 10] = min(time_diff / 24, 1.0)  # Hours since last action (normalized)
            
            # User context features
            state[i, 11] = user_profile.get('purchase_rate', 0)
            state[i, 12] = min(user_profile.get('avg_actions_per_day', 0) / 100, 1.0)
            state[i, 13] = min(user_profile.get('unique_categories', 0) / 100, 1.0)
            
            # Position in sequence
            state[i, 14] = i / sequence_length
            
            # Remaining features for future extensions
            state[i, 15:20] = 0  # Reserved for additional features
        
        return state


class AlibabaDQNLSTMRecommender(nn.Module):
    """
    Enhanced DQN-LSTM model specifically for Alibaba recommendation task
    Incorporates business-specific features and multi-behavior modeling
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 lstm_hidden_size: int = 256,
                 lstm_layers: int = 3,
                 fc_hidden_size: int = 512,
                 dropout_rate: float = 0.3,
                 attention_heads: int = 8):
        super(AlibabaDQNLSTMRecommender, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        
        # Enhanced LSTM with bidirectional processing
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism for better sequence modeling
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size * 2,  # *2 for bidirectional
            num_heads=attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_hidden_size * 2)
        
        # Enhanced fully connected layers with skip connections
        self.fc1 = nn.Linear(lstm_hidden_size * 2, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size // 2)
        self.fc3 = nn.Linear(fc_hidden_size // 2, fc_hidden_size // 4)
        
        # Output layers for different objectives
        self.q_value_head = nn.Linear(fc_hidden_size // 4, action_dim)
        self.value_head = nn.Linear(fc_hidden_size // 4, 1)  # State value for advantage computation
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Enhanced weight initialization"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Forget gate bias initialization
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1)
        
        # Initialize FC layers
        for module in [self.fc1, self.fc2, self.fc3, self.q_value_head, self.value_head]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x, hidden=None):
        """Enhanced forward pass with attention"""
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Initialize hidden states for bidirectional LSTM
        if hidden is None:
            h_0 = torch.zeros(self.lstm_layers * 2, batch_size, self.lstm_hidden_size).to(x.device)
            c_0 = torch.zeros(self.lstm_layers * 2, batch_size, self.lstm_hidden_size).to(x.device)
            hidden = (h_0, c_0)
        
        # LSTM processing
        lstm_out, new_hidden = self.lstm(x, hidden)
        
        # Apply attention mechanism
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection and layer normalization
        lstm_out = self.layer_norm(lstm_out + attended_out)
        
        # Take the last output for Q-value computation
        last_output = lstm_out[:, -1, :]
        
        # Enhanced fully connected processing with skip connections
        x1 = self.leaky_relu(self.fc1(last_output))
        x1 = self.dropout(x1)
        
        x2 = self.leaky_relu(self.fc2(x1))
        x2 = self.dropout(x2)
        
        x3 = self.leaky_relu(self.fc3(x2))
        x3 = self.dropout(x3)
        
        # Dueling DQN architecture
        q_values = self.q_value_head(x3)
        state_value = self.value_head(x3)
        
        # Advantage calculation
        advantages = q_values - q_values.mean(dim=1, keepdim=True)
        final_q_values = state_value + advantages
        
        return final_q_values, new_hidden, attention_weights


class AlibabaRecommenderAgent:
    """
    Enhanced DQN agent specifically designed for Alibaba recommendation system
    with multi-objective optimization and business-aware training
    """
    
    def __init__(self,
                 data_processor: AlibabaDataProcessor,
                 state_encoder: StateEncoder,
                 sequence_length: int = 20,
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.9995,
                 epsilon_min: float = 0.05,
                 batch_size: int = 64,
                 memory_size: int = 50000,
                 target_update_freq: int = 500,
                 priority_replay: bool = True):
        
        self.data_processor = data_processor
        self.state_encoder = state_encoder
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.priority_replay = priority_replay
        
        # State and action dimensions
        self.state_dim = 20  # Fixed for state encoder
        self.action_dim = state_encoder.n_items
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.q_network = AlibabaDQNLSTMRecommender(
            state_dim=self.state_dim,
            action_dim=self.action_dim
        ).to(self.device)
        
        self.target_network = AlibabaDQNLSTMRecommender(
            state_dim=self.state_dim,
            action_dim=self.action_dim
        ).to(self.device)
        
        # Initialize target network
        self.update_target_network()
        
        # Optimizer with learning rate scheduling
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # Loss function
        self.criterion = nn.SmoothL1Loss()
        
        # Enhanced replay buffer
        if priority_replay:
            self.memory = PriorityReplayBuffer(memory_size, self.state_dim, sequence_length)
        else:
            self.memory = ReplayBuffer(memory_size, self.state_dim, sequence_length)
        
        # Training metrics
        self.training_step = 0
        self.episode_count = 0
        self.losses = []
        self.rewards_history = []
        self.business_metrics = {
            'ctr': [],
            'conversion_rate': [],
            'diversity_score': [],
            'coverage': []
        }
        
        # Recommendation tracking
        self.recommended_items = set()
        self.successful_recommendations = defaultdict(int)
    
    def select_action(self, state, user_id: int, candidate_items: List[int] = None, 
                     training: bool = True):
        """
        Enhanced action selection with business constraints
        """
        if training and np.random.random() <= self.epsilon:
            # Exploration with smart sampling
            if candidate_items:
                return np.random.choice(candidate_items)
            else:
                return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values, _, attention_weights = self.q_network(state_tensor)
            q_values = q_values.squeeze(0)
            
            # Apply business constraints
            if candidate_items:
                # Mask Q-values for non-candidate items
                masked_q_values = torch.full_like(q_values, float('-inf'))
                for item_idx in candidate_items:
                    if item_idx < len(masked_q_values):
                        masked_q_values[item_idx] = q_values[item_idx]
                q_values = masked_q_values
            
            # Diversity promotion: reduce Q-values for recently recommended items
            user_recent_items = self._get_recent_recommendations(user_id)
            for item_idx in user_recent_items:
                if item_idx < len(q_values):
                    q_values[item_idx] *= 0.7  # Reduce by 30%
            
            return q_values.argmax().item()
    
    def _get_recent_recommendations(self, user_id: int, window_hours: int = 24) -> List[int]:
        """Get recently recommended items for diversity"""
        # Implementation would track recent recommendations per user
        return []  # Placeholder
    
    def train(self):
        """Enhanced training with multi-objective learning"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        if self.priority_replay:
            batch_data, indices, weights = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = batch_data
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q_values, _, _ = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network for action selection, target for evaluation
        with torch.no_grad():
            next_q_values_main, _, _ = self.q_network(next_states)
            next_actions = next_q_values_main.argmax(1)
            
            next_q_values_target, _, _ = self.target_network(next_states)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute TD errors
        td_errors = target_q_values - current_q_values.squeeze(1)
        
        # Weighted loss for priority replay
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priority replay buffer
        if self.priority_replay and hasattr(self.memory, 'update_priorities'):
            priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
            self.memory.update_priorities(indices, priorities)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        self.losses.append(loss.item())
        return loss.item()
    
    def update_target_network(self):
        """Soft update of target network"""
        tau = 0.005  # Soft update parameter
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def save_model(self, filepath: str):
        """Save comprehensive model checkpoint"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'losses': self.losses,
            'business_metrics': self.business_metrics,
            'state_encoder': {
                'item_to_idx': self.state_encoder.item_to_idx,
                'category_to_idx': self.state_encoder.category_to_idx,
                'n_items': self.state_encoder.n_items,
                'n_categories': self.state_encoder.n_categories
            }
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")


class PriorityReplayBuffer:
    """Priority Experience Replay Buffer for more efficient learning"""
    
    def __init__(self, capacity: int, state_dim: int, sequence_length: int, alpha: float = 0.6):
        self.capacity = capacity
        self.state_dim = state_dim
        self.sequence_length = sequence_length
        self.alpha = alpha
        self.beta = 0.4
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add experience with maximum priority"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """Sample batch with priority-based selection"""
        if self.size == 0:
            return None, None, None
        
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Extract experiences
        batch = [self.buffer[idx] for idx in indices]
        states = torch.stack([torch.FloatTensor(exp[0]) for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.stack([torch.FloatTensor(exp[3]) for exp in batch])
        dones = torch.BoolTensor([exp[4] for exp in batch])
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return self.size


class ReplayBuffer:
    """Standard Experience Replay Buffer"""
    
    def __init__(self, capacity: int, state_dim: int, sequence_length: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.stack([torch.FloatTensor(exp[0]) for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.stack([torch.FloatTensor(exp[3]) for exp in batch])
        dones = torch.BoolTensor([exp[4] for exp in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class AlibabaRecommenderEnvironment:
    """
    Real environment simulator using actual Alibaba dataset
    Simulates user interactions based on historical patterns
    """
    
    def __init__(self, data_processor: AlibabaDataProcessor, 
                 state_encoder: StateEncoder, sequence_length: int = 20,
                 test_split_date: str = "2017-12-01"):
        
        self.data_processor = data_processor
        self.state_encoder = state_encoder
        self.sequence_length = sequence_length
        
        # Split data into training and testing
        self.test_split_date = pd.to_datetime(test_split_date)
        self.train_data = data_processor.data[data_processor.data['timestamp'] < self.test_split_date]
        self.test_data = data_processor.data[data_processor.data['timestamp'] >= self.test_split_date]
        
        # Active users (users with sufficient history)
        user_counts = self.train_data['user_id'].value_counts()
        self.active_users = user_counts[user_counts >= sequence_length].index.tolist()
        
        # Current episode state
        self.current_user = None
        self.current_sequence = []
        self.episode_step = 0
        self.max_episode_steps = 50
        
        # Business metrics tracking
        self.episode_metrics = {
            'recommendations': 0,
            'clicks': 0,
            'carts': 0,
            'purchases': 0,
            'unique_items_recommended': set(),
            'total_reward': 0
        }
        
        print(f"Environment initialized with {len(self.active_users)} active users")
        print(f"Training data: {len(self.train_data)} interactions")
        print(f"Test data: {len(self.test_data)} interactions")
    
    def reset(self, user_id: int = None):
        """Reset environment for new episode"""
        if user_id is None:
            self.current_user = np.random.choice(self.active_users)
        else:
            self.current_user = user_id
        
        # Get user's historical sequence from training data
        user_train_data = self.train_data[self.train_data['user_id'] == self.current_user]
        if len(user_train_data) < self.sequence_length:
            return self.reset()  # Try another user
        
        # Sample a starting point in user's history
        start_idx = np.random.randint(0, max(1, len(user_train_data) - self.sequence_length))
        sequence_data = user_train_data.iloc[start_idx:start_idx + self.sequence_length]
        
        # Convert to sequence format
        self.current_sequence = []
        for _, row in sequence_data.iterrows():
            self.current_sequence.append({
                'item_id': row['item_id'],
                'category_id': row['category_id'],
                'behavior_type': row['behavior_type'],
                'behavior_code': row['behavior_code'],
                'timestamp': row['timestamp']
            })
        
        self.episode_step = 0
        self.episode_metrics = {
            'recommendations': 0,
            'clicks': 0,
            'carts': 0,
            'purchases': 0,
            'unique_items_recommended': set(),
            'total_reward': 0
        }
        
        return self._get_state()
    
    def step(self, action_item_idx: int):
        """Execute recommendation action and simulate user response"""
        self.episode_step += 1
        
        # Convert action index to actual item ID
        idx_to_item = {idx: item_id for item_id, idx in self.state_encoder.item_to_idx.items()}
        recommended_item = idx_to_item.get(action_item_idx, list(idx_to_item.values())[0])
        
        # Simulate user response based on historical patterns and item characteristics
        user_response = self._simulate_user_response(recommended_item)
        
        # Calculate reward
        reward = self.data_processor.calculate_reward(
            recommended_item, user_response, self.current_user
        )
        
        # Update sequence with new interaction
        new_interaction = {
            'item_id': recommended_item,
            'category_id': self.data_processor.item_profiles[recommended_item]['category_id'],
            'behavior_type': user_response,
            'behavior_code': self.data_processor.behavior_mapping[user_response],
            'timestamp': self.current_sequence[-1]['timestamp'] + timedelta(minutes=np.random.randint(1, 60))
        }
        
        self.current_sequence.append(new_interaction)
        if len(self.current_sequence) > self.sequence_length:
            self.current_sequence.pop(0)
        
        # Update episode metrics
        self._update_episode_metrics(recommended_item, user_response, reward)
        
        # Check if episode is done
        done = (self.episode_step >= self.max_episode_steps or 
               user_response == 'buy')  # End episode on purchase
        
        next_state = self._get_state()
        
        return next_state, reward, done, self._get_episode_info()
    
    def _simulate_user_response(self, item_id: int) -> str:
        """
        Simulate user response to recommendation based on:
        1. User's historical preferences
        2. Item characteristics
        3. Contextual factors
        """
        user_profile = self.data_processor.user_profiles.get(self.current_user, {})
        item_profile = self.data_processor.item_profiles.get(item_id, {})
        
        # Base probabilities from user's historical behavior
        base_probs = user_profile.get('behavior_ratios', {
            'pv': 0.7, 'cart': 0.15, 'fav': 0.1, 'buy': 0.05
        })
        
        # Adjust probabilities based on item characteristics
        item_popularity = item_profile.get('popularity_score', 0)
        item_conversion = item_profile.get('buy_conversion_rate', 0)
        
        # Preference matching bonus
        item_category = item_profile.get('category_id', 0)
        user_top_categories = user_profile.get('top_categories', [])
        preference_match = item_category in user_top_categories
        
        # Calculate adjusted probabilities
        probs = base_probs.copy()
        
        if preference_match:
            probs['buy'] *= 2.0
            probs['cart'] *= 1.5
            probs['fav'] *= 1.3
        
        # Item quality effects
        probs['buy'] *= (1 + item_conversion)
        probs['cart'] *= (1 + item_conversion * 0.5)
        
        # Normalize probabilities
        total_prob = sum(probs.values())
        probs = {k: v/total_prob for k, v in probs.items()}
        
        # Sample response
        behaviors = list(probs.keys())
        probabilities = list(probs.values())
        response = np.random.choice(behaviors, p=probabilities)
        
        return response
    
    def _get_state(self):
        """Get current state representation"""
        return self.state_encoder.encode_sequence(
            self.current_sequence, self.current_user, self.sequence_length
        )
    
    def _update_episode_metrics(self, item_id: int, response: str, reward: float):
        """Update episode business metrics"""
        self.episode_metrics['recommendations'] += 1
        self.episode_metrics['unique_items_recommended'].add(item_id)
        self.episode_metrics['total_reward'] += reward
        
        if response == 'pv':
            self.episode_metrics['clicks'] += 1
        elif response == 'cart':
            self.episode_metrics['clicks'] += 1
            self.episode_metrics['carts'] += 1
        elif response == 'fav':
            self.episode_metrics['clicks'] += 1
        elif response == 'buy':
            self.episode_metrics['clicks'] += 1
            self.episode_metrics['purchases'] += 1
    
    def _get_episode_info(self):
        """Get current episode information"""
        recs = self.episode_metrics['recommendations']
        return {
            'ctr': self.episode_metrics['clicks'] / max(1, recs),
            'cart_rate': self.episode_metrics['carts'] / max(1, recs),
            'conversion_rate': self.episode_metrics['purchases'] / max(1, recs),
            'diversity': len(self.episode_metrics['unique_items_recommended']) / max(1, recs),
            'total_reward': self.episode_metrics['total_reward'],
            'episode_step': self.episode_step
        }


def train_alibaba_recommender(data_path: str, 
                            num_episodes: int = 2000,
                            sample_size: int = 100000,
                            sequence_length: int = 20,
                            save_interval: int = 500):
    """
    Train the DQN-LSTM recommender on Alibaba dataset
    """
    print("="*80)
    print("ALIBABA DQN-LSTM RECOMMENDER TRAINING")
    print("="*80)
    
    # Initialize components
    print("\n1. Loading and preprocessing Alibaba dataset...")
    data_processor = AlibabaDataProcessor(data_path, sample_size)
    
    print("\n2. Initializing state encoder...")
    state_encoder = StateEncoder(data_processor)
    
    print("\n3. Setting up environment...")
    env = AlibabaRecommenderEnvironment(data_processor, state_encoder, sequence_length)
    
    print("\n4. Initializing DQN-LSTM agent...")
    agent = AlibabaRecommenderAgent(
        data_processor=data_processor,
        state_encoder=state_encoder,
        sequence_length=sequence_length,
        learning_rate=0.0001,
        gamma=0.99,
        batch_size=64,
        memory_size=50000,
        priority_replay=True
    )
    
    # Training metrics
    episode_rewards = []
    business_metrics = {
        'ctr_history': [],
        'conversion_history': [],
        'diversity_history': [],
        'avg_reward_history': []
    }
    
    print(f"\n5. Starting training for {num_episodes} episodes...")
    print(f"Device: {agent.device}")
    print(f"Action space size: {agent.action_dim}")
    print(f"State dimension: {agent.state_dim}")
    
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        
        # Run episode
        while True:
            # Select action with candidate filtering for efficiency
            # In practice, you might limit recommendations to popular items or user's categories
            candidate_items = None  # Can add business logic here
            
            action = agent.select_action(state, env.current_user, candidate_items)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.memory) >= agent.batch_size:
                loss = agent.train()
                if loss is not None:
                    episode_losses.append(loss)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update metrics
        episode_rewards.append(episode_reward)
        agent.episode_count += 1
        
        # Business metrics
        business_metrics['ctr_history'].append(info['ctr'])
        business_metrics['conversion_history'].append(info['conversion_rate'])
        business_metrics['diversity_history'].append(info['diversity'])
        business_metrics['avg_reward_history'].append(episode_reward)
        
        # Update agent's business metrics
        agent.business_metrics['ctr'].append(info['ctr'])
        agent.business_metrics['conversion_rate'].append(info['conversion_rate'])
        agent.business_metrics['diversity_score'].append(info['diversity'])
        
        # Logging
        if episode % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            recent_ctr = business_metrics['ctr_history'][-100:]
            recent_conversion = business_metrics['conversion_history'][-100:]
            recent_diversity = business_metrics['diversity_history'][-100:]
            
            avg_reward = np.mean(recent_rewards)
            avg_ctr = np.mean(recent_ctr)
            avg_conversion = np.mean(recent_conversion)
            avg_diversity = np.mean(recent_diversity)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            
            print(f"\nEpisode {episode:4d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"CTR: {avg_ctr:.3f} | "
                  f"Conversion: {avg_conversion:.3f} | "
                  f"Diversity: {avg_diversity:.3f}")
            print(f"              | "
                  f"Loss: {avg_loss:.6f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Memory: {len(agent.memory):5d}")
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save_model(f"best_alibaba_recommender.pth")
                print(f"              | New best model saved! (Reward: {best_reward:.2f})")
        
        # Periodic saving
        if episode % save_interval == 0 and episode > 0:
            agent.save_model(f"alibaba_recommender_episode_{episode}.pth")
            
            # Save training progress
            progress_data = {
                'episode_rewards': episode_rewards,
                'business_metrics': business_metrics,
                'training_config': {
                    'num_episodes': num_episodes,
                    'sequence_length': sequence_length,
                    'learning_rate': 0.0001,
                    'gamma': 0.99
                }
            }
            
            with open(f"training_progress_episode_{episode}.pkl", 'wb') as f:
                pickle.dump(progress_data, f)
    
    print(f"\nTraining completed! Best average reward: {best_reward:.2f}")
    
    # Final model save
    agent.save_model("final_alibaba_recommender.pth")
    
    return agent, episode_rewards, business_metrics


def evaluate_alibaba_recommender(agent: AlibabaRecommenderAgent,
                                env: AlibabaRecommenderEnvironment,
                                num_episodes: int = 200):
    """
    Comprehensive evaluation of the trained recommender
    """
    print("\n" + "="*80)
    print("EVALUATING ALIBABA RECOMMENDER SYSTEM")
    print("="*80)
    
    agent.epsilon = 0.0  # No exploration during evaluation
    
    evaluation_metrics = {
        'rewards': [],
        'ctr': [],
        'conversion_rates': [],
        'diversity_scores': [],
        'episode_lengths': [],
        'user_satisfaction': []  # Based on reward
    }
    
    category_performance = defaultdict(list)
    user_type_performance = defaultdict(list)
    
    print(f"Running {num_episodes} evaluation episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        step_count = 0
        
        # Track user characteristics for analysis
        user_profile = env.data_processor.user_profiles.get(env.current_user, {})
        user_type = "high_activity" if user_profile.get('avg_actions_per_day', 0) > 10 else "low_activity"
        
        while True:
            action = agent.select_action(state, env.current_user, training=False)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            state = next_state
            
            if done:
                break
        
        # Collect metrics
        evaluation_metrics['rewards'].append(episode_reward)
        evaluation_metrics['ctr'].append(info['ctr'])
        evaluation_metrics['conversion_rates'].append(info['conversion_rate'])
        evaluation_metrics['diversity_scores'].append(info['diversity'])
        evaluation_metrics['episode_lengths'].append(step_count)
        evaluation_metrics['user_satisfaction'].append(episode_reward / step_count if step_count > 0 else 0)
        
        # User type analysis
        user_type_performance[user_type].append({
            'reward': episode_reward,
            'ctr': info['ctr'],
            'conversion': info['conversion_rate']
        })
    
    # Calculate summary statistics
    print("\nEVALUATION RESULTS:")
    print("-" * 40)
    
    avg_reward = np.mean(evaluation_metrics['rewards'])
    avg_ctr = np.mean(evaluation_metrics['ctr'])
    avg_conversion = np.mean(evaluation_metrics['conversion_rates'])
    avg_diversity = np.mean(evaluation_metrics['diversity_scores'])
    avg_episode_length = np.mean(evaluation_metrics['episode_lengths'])
    
    print(f"Average Reward per Episode: {avg_reward:.2f} ± {np.std(evaluation_metrics['rewards']):.2f}")
    print(f"Click-Through Rate (CTR):   {avg_ctr:.3f} ± {np.std(evaluation_metrics['ctr']):.3f}")
    print(f"Conversion Rate:            {avg_conversion:.3f} ± {np.std(evaluation_metrics['conversion_rates']):.3f}")
    print(f"Diversity Score:            {avg_diversity:.3f} ± {np.std(evaluation_metrics['diversity_scores']):.3f}")
    print(f"Average Episode Length:     {avg_episode_length:.1f} steps")
    
    # Performance by user type
    print(f"\nPERFORMANCE BY USER TYPE:")
    print("-" * 40)
    for user_type, performance_data in user_type_performance.items():
        if performance_data:
            avg_reward_type = np.mean([p['reward'] for p in performance_data])
            avg_ctr_type = np.mean([p['ctr'] for p in performance_data])
            avg_conv_type = np.mean([p['conversion'] for p in performance_data])
            
            print(f"{user_type.upper()}: Reward={avg_reward_type:.2f}, CTR={avg_ctr_type:.3f}, Conv={avg_conv_type:.3f}")
    
    # Business impact estimation
    print(f"\nBUSINESS IMPACT ESTIMATION:")
    print("-" * 40)
    
    baseline_ctr = 0.02  # Assumed baseline CTR
    baseline_conversion = 0.001  # Assumed baseline conversion
    
    ctr_improvement = (avg_ctr - baseline_ctr) / baseline_ctr * 100 if baseline_ctr > 0 else 0
    conv_improvement = (avg_conversion - baseline_conversion) / baseline_conversion * 100 if baseline_conversion > 0 else 0
    
    print(f"CTR Improvement over Baseline:        {ctr_improvement:+.1f}%")
    print(f"Conversion Improvement over Baseline: {conv_improvement:+.1f}%")
    print(f"Recommendation Diversity:             {avg_diversity:.1%}")
    
    return evaluation_metrics


# Example usage and main execution
if __name__ == "__main__":
    # Configuration
    DATA_PATH = "alibaba_user_behavior.csv"  # Path to your Alibaba dataset
    NUM_EPISODES = 1500
    SAMPLE_SIZE = 50000  # Use subset for faster training/testing
    SEQUENCE_LENGTH = 20
    
    print("ALIBABA DEEP REINFORCEMENT LEARNING RECOMMENDER SYSTEM")
    print("Based on: Deep Reinforcement Learning-Based Recommender Algorithm")
    print("Optimization and Intelligent Systems Construction for Business Data Analysis")
    print("\nDataset: Alibaba User Behaviour Dataset from Tianchi")
    
    try:
        # Train the model
        agent, rewards, business_metrics = train_alibaba_recommender(
            data_path=DATA_PATH,
            num_episodes=NUM_EPISODES,
            sample_size=SAMPLE_SIZE,
            sequence_length=SEQUENCE_LENGTH
        )
        
        # Evaluate the model
        env = AlibabaRecommenderEnvironment(
            agent.data_processor, 
            agent.state_encoder, 
            SEQUENCE_LENGTH
        )
        evaluation_results = evaluate_alibaba_recommender(agent, env)
        
        # Generate visualizations if matplotlib available
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Alibaba DQN-LSTM Recommender Training Results', fontsize=16)
            
            # Training reward
            axes[0, 0].plot(rewards)
            axes[0, 0].set_title('Training Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
            
            # CTR progression
            axes[0, 1].plot(business_metrics['ctr_history'])
            axes[0, 1].set_title('Click-Through Rate')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('CTR')
            axes[0, 1].grid(True)
            
            # Conversion rate
            axes[0, 2].plot(business_metrics['conversion_history'])
            axes[0, 2].set_title('Conversion Rate')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Conversion Rate')
            axes[0, 2].grid(True)
            
            # Diversity
            axes[1, 0].plot(business_metrics['diversity_history'])
            axes[1, 0].set_title('Recommendation Diversity')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Diversity Score')
            axes[1, 0].grid(True)
            
            # Training loss
            if agent.losses:
                axes[1, 1].plot(agent.losses)
                axes[1, 1].set_title('Training Loss')
                axes[1, 1].set_xlabel('Training Step')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].grid(True)
            
            # Evaluation metrics distribution
            axes[1, 2].hist(evaluation_results['rewards'], bins=30, alpha=0.7)
            axes[1, 2].set_title('Evaluation Reward Distribution')
            axes[1, 2].set_xlabel('Episode Reward')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].grid(True)
            
            plt.tight_layout()
            plt.savefig('alibaba_recommender_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\nTraining results saved to 'alibaba_recommender_results.png'")
            
        except ImportError:
            print("Matplotlib not available. Skipping visualization.")
        
        print(f"\nAll models and results saved successfully!")
        print(f"Best model: 'best_alibaba_recommender.pth'")
        print(f"Final model: 'final_alibaba_recommender.pth'")
    
    except FileNotFoundError:
        print(f"\nERROR: Dataset file '{DATA_PATH}' not found.")
        print("Please ensure you have downloaded the Alibaba User Behaviour Dataset")
        print("from: https://tianchi.aliyun.com/dataset/dataDetail?dataId=81505")
        print("and update the DATA_PATH variable with the correct file path.")
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Please check your dataset format and try again.")