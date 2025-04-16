import networkx as nx

# 初始化图
tetris_graph = nx.Graph()

# 添加固定方块
for x in range(10):
    for y in range(5, 20):  # 假设下半部分已经填满
        tetris_graph.add_node((x, y), occupied=1, height=y)

# 添加当前下落方块 (L 形)
falling_piece = [(4, 4), (5, 4), (6, 4), (6, 3)]  # 4 格 L 形
for node in falling_piece:
    tetris_graph.add_node(node, occupied=2, height=node[1])  # 2 代表是下落方块

# 添加边 (连接相邻方块)
for node in tetris_graph.nodes:
    x, y = node
    neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    for neighbor in neighbors:
        if neighbor in tetris_graph:
            weight = 2 if tetris_graph.nodes[node]['occupied'] == 2 else 1
            tetris_graph.add_edge(node, neighbor, weight=weight)

# 可视化：输出邻接列表
print(tetris_graph.edges(data=True))
