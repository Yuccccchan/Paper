"""
可视化示例：展示 MAIRS 训练过程

这个脚本创建一个简单的可视化，帮助理解智能体如何学习。
"""

import sys
sys.path.insert(0, 'src')

from environment import TSNNetwork, TTFlow
from mairs import MAIRSAgent, MAPPO
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

print("=" * 70)
print("MAIRS 训练过程可视化")
print("=" * 70)
print()

# 创建简单网络
print("步骤 1: 创建网络...")
network = TSNNetwork(slot_duration=0.2, link_bandwidth=100)

# 3个交换机，线性拓扑
for i in range(3):
    network.add_switch(i)

network.add_link(0, 1)
network.add_link(1, 2)

# 创建数据流
flow = TTFlow(0, source=0, destination=2, data_size=1000, period=4.8)
network.add_flow(flow)

print(f"✓ 网络: {network}")
print(f"✓ 数据流: 从交换机0到交换机2")
print()

# 创建智能体
print("步骤 2: 创建智能体...")
agents = {}
for switch_id in network.switches:
    num_neighbors = len(network.switches[switch_id].get_neighbors()) or 1
    agents[switch_id] = MAIRSAgent(
        agent_id=switch_id,
        num_neighbors=num_neighbors,
        max_slots=network.max_slots,
        state_dim=20,
        hidden_dim=64
    )
print(f"✓ 创建了 {len(agents)} 个智能体")
print()

# 训练并记录过程
print("步骤 3: 开始训练（200次迭代）...")
print("这将展示智能体如何从失败到成功...\n")

trainer = MAPPO(agents, alpha=0.25)

# 训练更多迭代来展示学习过程
history = trainer.train(
    network=network,
    flows=[flow],
    num_iterations=200,
    batch_size=10
)

print("\n✓ 训练完成！")
print()

# 创建可视化
print("步骤 4: 生成学习曲线...")

if len(history['success_rates']) > 0:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 图1: 成功率随时间变化
    iterations = range(len(history['success_rates']))
    axes[0].plot(iterations, history['success_rates'], 'b-', linewidth=2)
    axes[0].set_xlabel('训练迭代次数', fontsize=12)
    axes[0].set_ylabel('成功率', fontsize=12)
    axes[0].set_title('智能体学习曲线：成功率提升', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 1.1)
    
    # 添加注释
    if len(history['success_rates']) > 20:
        early_rate = sum(history['success_rates'][:10]) / 10
        late_rate = sum(history['success_rates'][-10:]) / 10
        
        axes[0].axhline(y=early_rate, color='r', linestyle='--', alpha=0.5, label=f'初期平均: {early_rate:.1%}')
        axes[0].axhline(y=late_rate, color='g', linestyle='--', alpha=0.5, label=f'后期平均: {late_rate:.1%}')
        axes[0].legend()
    
    # 图2: 奖励随时间变化
    axes[1].plot(iterations, history['avg_rewards'], 'g-', linewidth=2)
    axes[1].set_xlabel('训练迭代次数', fontsize=12)
    axes[1].set_ylabel('平均奖励', fontsize=12)
    axes[1].set_title('奖励变化：智能体获得的反馈', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/mairs_learning_curve.png', dpi=150, bbox_inches='tight')
    print("✓ 学习曲线已保存到: /tmp/mairs_learning_curve.png")
    print()
    
    # 显示统计信息
    print("=" * 70)
    print("训练统计")
    print("=" * 70)
    
    if len(history['success_rates']) > 20:
        early_success = sum(history['success_rates'][:10]) / 10
        late_success = sum(history['success_rates'][-10:]) / 10
        improvement = ((late_success - early_success) / (early_success + 0.001)) * 100
        
        print(f"初期（前10次）成功率: {early_success:.1%}")
        print(f"后期（后10次）成功率: {late_success:.1%}")
        print(f"提升幅度: {improvement:+.1f}%")
        print()
        
        early_reward = sum(history['avg_rewards'][:10]) / 10
        late_reward = sum(history['avg_rewards'][-10:]) / 10
        
        print(f"初期平均奖励: {early_reward:.3f}")
        print(f"后期平均奖励: {late_reward:.3f}")
        print(f"奖励提升: {late_reward - early_reward:+.3f}")
    
    print()
    print("=" * 70)
    print("学习过程解释")
    print("=" * 70)
    print("""
训练开始时：
- 智能体随机选择动作
- 经常失败（绕圈、选错时间槽）
- 成功率很低

训练过程中：
- 智能体记住哪些动作导致成功
- 逐渐学会避免错误选择
- 成功率逐步提升

训练结束后：
- 智能体学会了最优策略
- 能够稳定地路由数据
- 成功率达到很高水平

这就是强化学习的过程：通过试错来学习！
    """)

# 测试训练后的智能体
print("=" * 70)
print("测试训练后的智能体")
print("=" * 70)

network.reset_schedules()
trajectory, reward, success = trainer.rollout_flow(network, flow)

print(f"✓ 测试结果: {'成功 ✓' if success else '失败 ✗'}")
print(f"  选择的路径: {' -> '.join(f'交换机{s}' for s in flow.route)}")
print(f"  获得奖励: {reward:.3f}")
print()

if success and len(flow.route) > 1:
    print("分配的时间槽:")
    for i in range(len(flow.route) - 1):
        link = (flow.route[i], flow.route[i+1])
        if link in flow.schedule:
            slot = flow.schedule[link]
            print(f"  交换机{link[0]} -> 交换机{link[1]}: 时间槽 {slot}")

print()
print("=" * 70)
print("可视化完成！查看 /tmp/mairs_learning_curve.png 了解学习过程")
print("=" * 70)
