# Key Algorithms from the Paper

This document provides the core algorithms from "Pay Attention to Network" paper in detail.

## Algorithm 1: Load-Aware TSN Flow Scheduling

### Purpose
Minimize congestion in TSN domain by considering bottleneck resources on network links.

### Algorithm Pseudocode

```
Algorithm 1: Load-aware TSN flow scheduling

Input: Flow set F, Network topology G = (V, E), Cycle time T_cycle
Output: Scheduled flows with paths and GCL entries

1: Initialize link_loads for all links in E
2: Sort flows F by start_offset (ascending)
3: scheduled_flows ← empty list

4: for each flow f in sorted F do
5:     // Step 1: Path Selection
6:     paths ← all_simple_paths(G, f.src, f.dst)
7:     best_path ← None
8:     min_max_load ← ∞
9:     
10:    for each path p in paths do
11:        max_load ← 0
12:        for each link l in p do
13:            current_load ← get_link_load(l, link_loads)
14:            max_load ← max(max_load, current_load)
15:        end for
16:        
17:        if max_load < min_max_load then
18:            min_max_load ← max_load
19:            best_path ← p
20:        end if
21:    end for
22:    
23:    f.path ← best_path
24:    
25:    // Step 2: Time Slot Allocation
26:    current_offset ← f.start_offset
27:    gcl_entries ← empty map
28:    
29:    for each link l in best_path do
30:        // Find available time slot
31:        slot ← find_available_slot(l, current_offset, f, link_loads)
32:        
33:        // Calculate transmission time
34:        trans_time ← (f.size × 8) / bandwidth(l)
35:        
36:        // Create GCL entry
37:        gcl_entries[l] ← {
38:            offset: slot,
39:            duration: trans_time,
40:            queue: priority_queue(f)
41:        }
42:        
43:        // Update offset for next link
44:        current_offset ← slot + trans_time + latency(l)
45:    end for
46:    
47:    f.gcl_entries ← gcl_entries
48:    
49:    // Step 3: Update Link Loads
50:    update_link_loads(f, best_path, gcl_entries, link_loads)
51:    scheduled_flows.append(f)
52: end for

53: return scheduled_flows
```

### Key Functions

#### find_available_slot(link, preferred_offset, flow, link_loads)
```
1: slot ← round_to_slot_boundary(preferred_offset)
2: occupied_slots ← link_loads[link]
3: 
4: while has_conflict(slot, flow, occupied_slots) do
5:     slot ← slot + time_slot_duration
6: end while
7: 
8: return slot
```

#### has_conflict(slot, flow, occupied_slots)
```
1: trans_time ← calculate_transmission_time(flow)
2: 
3: for each existing_slot in occupied_slots do
4:     if slot + trans_time > existing_slot.start AND 
5:        slot < existing_slot.end then
6:         return True
7:     end if
8: end for
9: 
10: return False
```

### Complexity Analysis
- Time Complexity: O(|F| × |P| × |E|), where:
  - |F| = number of flows
  - |P| = average number of paths per flow
  - |E| = number of edges in topology
- Space Complexity: O(|F| × |E|)

---

## Algorithm 2: WiFi DDPG Scheduling

### Purpose
Adaptive scheduling for WiFi domain using attention-based Deep Deterministic Policy Gradient.

### Algorithm Pseudocode

```
Algorithm 2: Wi-Fi DDPG Scheduling

Input: Flow features, Channel SNR per band
Output: Band selection and airtime allocation for each flow

// Initialization
1: Initialize actor network θ^μ (with attention layers)
2: Initialize critic network θ^Q
3: Initialize target networks: θ^μ' ← θ^μ, θ^Q' ← θ^Q
4: Initialize replay buffer B with capacity N
5: Set learning rates α_actor, α_critic
6: Set discount factor γ, soft update rate τ

// Training Loop
7: for episode m = 1 to M do
8:     Reset environment to initial state
9:     
10:    for cycle t = 1 to T do
11:        // Observe State
12:        flow_features ← get_flow_features()
13:        channel_snr ← get_channel_snr()
14:        state s(t) ← encode_state(flow_features, channel_snr)
15:        
16:        // Generate Action (with exploration noise)
17:        action a(t) ← μ(s(t)|θ^μ) + ε
18:        where ε ~ N(0, σ) is exploration noise
19:        
20:        // Execute Action
21:        band_allocation, airtime_allocation ← parse_action(a(t))
22:        apply_scheduling(band_allocation, airtime_allocation)
23:        
24:        // Evaluate Reward
25:        reliability ← calculate_reliability(flows)
26:        deadline_violations ← count_deadline_violations(flows)
27:        reward r(t) ← α × reliability - β × deadline_violations
28:        
29:        // Observe Next State
30:        state s(t+1) ← get_next_state()
31:        
32:        // Store Transition
33:        B.push(<s(t), a(t), r(t), s(t+1)>)
34:        
35:        // Training Step
36:        if |B| ≥ batch_size then
37:            // Sample Mini-batch
38:            batch ← B.sample(batch_size)
39:            
40:            // Update Critic
41:            for each (s, a, r, s') in batch do
42:                a' ← μ(s'|θ^μ')  // Target actor
43:                y ← r + γ × Q(s', a'|θ^Q')  // Target Q-value
44:            end for
45:            
46:            L_critic ← (1/batch_size) Σ (Q(s, a|θ^Q) - y)²
47:            θ^Q ← θ^Q - α_critic × ∇_θQ L_critic
48:            
49:            // Update Actor
50:            L_actor ← -(1/batch_size) Σ Q(s, μ(s|θ^μ)|θ^Q)
51:            θ^μ ← θ^μ - α_actor × ∇_θμ L_actor
52:            
53:            // Soft Update Target Networks
54:            θ^Q' ← (1 - τ) × θ^Q' + τ × θ^Q
55:            θ^μ' ← (1 - τ) × θ^μ' + τ × θ^μ
56:        end if
57:    end for
58: end for

59: return trained policy μ(·|θ^μ)
```

### Neural Network Architecture

#### Actor Network with Attention
```
Input: (batch_size, num_flows, flow_feature_dim)

Layer 1: Linear Embedding
    flow_features → embedded_features (embed_dim)

Layers 2-4: Self-Attention Blocks (repeated 3 times)
    For each block:
        1. Multi-Head Self-Attention
           Q = Linear(embedded_features)
           K = Linear(embedded_features)
           V = Linear(embedded_features)
           
           Attention(Q,K,V) = softmax(QK^T / √d_k) × V
           
        2. Residual Connection + Layer Norm
           x = LayerNorm(x + Attention(x))
           
        3. Feed-Forward Network
           FFN(x) = Linear(ReLU(Linear(x)))
           
        4. Residual Connection
           x = x + FFN(x)

Output Heads:
    Band Selection: Linear(embed_dim → num_bands=3)
    Airtime Allocation: Linear(embed_dim → 1) → Sigmoid

Output: (band_logits, airtime_normalized)
```

#### Critic Network
```
Input: 
    - State: (batch_size, num_flows, flow_feature_dim)
    - Action: (batch_size, num_flows, action_dim)

Process:
    1. Embed state with attention layers
    2. Embed action
    3. Aggregate: state_agg = mean(state_embed)
               action_agg = mean(action_embed)
    4. Concatenate and process through FC layers
    5. Output: Q(s,a) scalar value

Output: Q-value (batch_size, 1)
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| α_actor | 1e-4 | Actor learning rate |
| α_critic | 1e-3 | Critic learning rate |
| γ | 0.99 | Discount factor |
| τ | 0.001 | Soft update rate |
| batch_size | 64 | Mini-batch size |
| buffer_size | 100,000 | Replay buffer capacity |
| embed_dim | 128 | Embedding dimension |
| num_heads | 8 | Number of attention heads |
| num_layers | 3 | Number of attention layers |
| σ | 0.1 → 0.01 | Exploration noise (decaying) |

---

## Algorithm 3: Ternary Band Selection

### Purpose
Efficiently select optimal frequency band for each flow among 3 bands (2.4/5/6 GHz).

### Algorithm Pseudocode

```
Algorithm 3: Ternary Band Selection

Input: Flow f, Channel states for 3 bands, Available airtime per band
Output: Selected band and allocated airtime

1: bands ← [2.4GHz, 5GHz, 6GHz]
2: best_band ← None
3: best_score ← -∞

4: for each band b in bands do
5:     // Calculate minimum airtime needed
6:     data_rate ← calculate_data_rate(SNR[b])
7:     min_airtime ← (f.size × 8) / data_rate
8:     
9:     // Check availability
10:    available_airtime ← get_available_airtime(b)
11:    
12:    if available_airtime < min_airtime then
13:        continue  // Skip this band
14:    end if
15:    
16:    // Calculate reliability
17:    reliability ← calculate_reliability(f.size, SNR[b])
18:    
19:    // Calculate score (multi-objective)
20:    score ← w1 × reliability 
21:          - w2 × (min_airtime / available_airtime)
22:          + w3 × (available_airtime - min_airtime)
23:    
24:    if score > best_score then
25:        best_score ← score
26:        best_band ← b
27:    end if
28: end for

29: if best_band is None then
30:    // Fallback: use least congested band
31:    best_band ← argmax_b(available_airtime[b])
32: end if

33: return best_band
```

### Scoring Function Details

```
Score Components:
1. Reliability: Based on SNR and channel conditions
   reliability(SNR) = 1 - PER
   where PER = 1 - (1 - BER)^(packet_size × 8)
         BER ≈ 0.5 × exp(-0.5 × 10^(SNR/10))

2. Resource Efficiency: Minimize wasted airtime
   efficiency = min_airtime / available_airtime

3. Margin: Prefer bands with more available time
   margin = available_airtime - min_airtime

Weights (empirically tuned):
   w1 = 100  (prioritize reliability)
   w2 = 50   (penalize inefficiency)
   w3 = 10   (bonus for margin)
```

---

## Algorithm 4: Cross-Domain Constraint Validation

### Purpose
Ensure scheduling decisions satisfy TSN-WiFi coordination constraints.

### Algorithm Pseudocode

```
Algorithm 4: Validate Cross-Domain Constraints

Input: Scheduled flow f with TSN path and WiFi allocation
Output: True if valid, False otherwise

// Constraint 1: TSN Domain Constraints
1: for each link l in f.path do
2:     gcl_entry ← f.gcl_entries[l]
3:     
4:     // Check time slot availability
5:     if has_conflict_on_link(l, gcl_entry) then
6:         return False
7:     end if
8:     
9:     // Check queue capacity
10:    if queue_overload(l, gcl_entry.queue) then
11:        return False
12:    end if
13: end for

// Constraint 2: WiFi Domain Constraints
14: band ← f.wifi_band
15: airtime ← f.wifi_airtime

16: // Check airtime availability
17: if airtime > available_airtime(band) then
18:     return False
19: end if

20: // Check minimum airtime requirement
21: data_rate ← get_data_rate(band)
22: min_airtime ← (f.size × 8) / data_rate

23: if airtime < min_airtime then
24:     return False
25: end if

// Constraint 3: Cross-Domain Constraint
26: // Flow can only start in WiFi after TSN completion
27: tsn_end_time ← get_tsn_end_time(f)
28: wifi_start_time ← get_wifi_start_time(f)

29: if wifi_start_time < tsn_end_time then
30:     return False
31: end if

// Constraint 4: Deadline Constraint
32: end_to_end_delay ← tsn_end_time + airtime
33: if end_to_end_delay > f.deadline then
34:     return False  // Optional: may allow violations with penalty
35: end if

36: return True
```

---

## Algorithm 5: State Encoding for DRL

### Purpose
Convert raw environment observations into feature vectors for neural network input.

### Algorithm Pseudocode

```
Algorithm 5: Encode State for DRL Input

Input: Flows F, TSN schedules, WiFi channel states
Output: Feature matrix (num_flows × feature_dim)

1: feature_matrix ← empty matrix
2: bands ← [2.4GHz, 5GHz, 6GHz]

3: for each flow f in F do
4:     features ← empty vector
5:     
6:     // Flow Characteristics (3 features)
7:     features.append(f.size / 1500.0)  // Normalized
8:     features.append(f.start_offset / 1000.0)
9:     features.append(f.deadline / 1000.0)
10:    
11:    // TSN Domain Features (3 features)
12:    if f.path exists then
13:        path_length ← length(f.path)
14:        features.append(path_length / 10.0)  // Normalized
15:        
16:        tsn_end_time ← calculate_tsn_end_time(f)
17:        features.append(tsn_end_time / 1000.0)
18:        
19:        remaining_budget ← f.deadline - tsn_end_time
20:        features.append(remaining_budget / 1000.0)
21:    else
22:        features.append([0, 0, 0])
23:    end if
24:    
25:    // WiFi Channel Features (9 features: 3 per band)
26:    for each band b in bands do
27:        channel ← channel_states[b]
28:        
29:        // SNR (normalized to 0-1)
30:        features.append(channel.snr / 40.0)
31:        
32:        // Channel busy ratio
33:        features.append(channel.busy_ratio)
34:        
35:        // Packet success rate
36:        success_rate ← 1 - channel.packet_loss_rate
37:        features.append(success_rate)
38:    end for
39:    
40:    // Resource Availability (3 features)
41:    for each band b in bands do
42:        avail ← available_airtime(b) / max_airtime
43:        features.append(avail)
44:    end for
45:    
46:    feature_matrix.append(features)
47: end for

48: return feature_matrix  // Shape: (num_flows, 18)
```

### Feature Summary

| Category | Features | Count |
|----------|----------|-------|
| Flow Info | size, offset, deadline | 3 |
| TSN Status | path_length, tsn_end, remaining_budget | 3 |
| Channel State | SNR, busy_ratio, success_rate (×3 bands) | 9 |
| Resource Avail | available_airtime (×3 bands) | 3 |
| **Total** | | **18** |

---

## Performance Metrics Calculations

### Reliability
```
For flow f with packet size S and SNR:
    BER = 0.5 × exp(-0.5 × 10^(SNR/10))
    PER = 1 - (1 - BER)^(S × 8)
    Reliability = 1 - PER

Average Reliability = (1/|F|) × Σ Reliability(f)
```

### Deadline Satisfaction Rate
```
satisfied_flows = 0
for each flow f:
    end_to_end_delay = tsn_completion_time + wifi_airtime
    if end_to_end_delay ≤ f.deadline:
        satisfied_flows += 1

DSR = satisfied_flows / total_flows
```

### Resource Utilization
```
TSN Link Utilization:
    util(link) = allocated_bandwidth / total_bandwidth
    Avg TSN Util = mean(util(l) for all l in links)

WiFi Band Utilization:
    util(band) = allocated_airtime / max_airtime
    Avg WiFi Util = mean(util(b) for all b in bands)
```

### Variance of Link Loads (Load Balance)
```
For each link l:
    load(l) = number of flows using link l

variance = (1/|L|) × Σ (load(l) - mean_load)²
```

Lower variance indicates better load balancing.

---

## Implementation Notes

1. **Numerical Stability**: Normalize all features to [0, 1] range before feeding to neural networks.

2. **Exploration Strategy**: Use decaying Gaussian noise for DDPG exploration:
   ```
   noise_scale(episode) = max(0.01, 0.1 × (1 - episode/total_episodes))
   ```

3. **Batch Processing**: Process all flows in parallel through attention layers for efficiency.

4. **Gradient Clipping**: Clip gradients to [-1, 1] to prevent explosion:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

5. **Target Network Updates**: Use soft updates (τ=0.001) instead of hard copies for stability.

6. **Replay Buffer**: Store transitions with priority based on TD-error for better sample efficiency (optional enhancement).

---

## Algorithm Complexity Summary

| Algorithm | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Load-Aware TSN | O(\|F\| × \|P\| × \|E\|) | O(\|F\| × \|E\|) |
| WiFi DDPG | O(batch_size × model_forward) | O(buffer_size) |
| Ternary Selection | O(3) = O(1) | O(1) |
| Constraint Check | O(\|path\|) | O(1) |
| State Encoding | O(\|F\| × feature_dim) | O(\|F\| × feature_dim) |

Where:
- |F| = number of flows
- |P| = number of paths per flow
- |E| = number of edges/links
- |path| = average path length

