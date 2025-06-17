import numpy as np

# 点在多边形内判定算法
def is_inside_polygon(x, y, poly_x, poly_y):
    n = len(poly_x)
    inside = False
    p1x, p1y = poly_x[0], poly_y[0]
    for i in range(n + 1):
        p2x, p2y = poly_x[i % n], poly_y[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# 向量化计算点到多边形的有符号距离的函数
def compute_signed_distance_vectorized(points, poly_x, poly_y):
    # 确保 poly_x 和 poly_y 的长度相同
    min_len = min(len(poly_x), len(poly_y))
    poly_x = poly_x[:min_len]
    poly_y = poly_y[:min_len]
    
    poly_points = np.column_stack((poly_x, poly_y))
    n = len(poly_points) # 多边形的顶点数
    
    def point_to_line_segment_distance(p, a, b): # 点 p 到线段 ab 的最短距离
        ab = b - a
        ap = p - a
        epsilon = 1e-10
        projection = np.dot(ap, ab) / (np.dot(ab, ab) + epsilon) # p 在线段 ab 上的投影比例
        projection = np.clip(projection, 0, 1) # 将投影比例限制在 [0, 1]
        closest = a + projection * ab
        return np.linalg.norm(p - closest) # 返回点 p 到最近点的欧几里得距离
    
    # 计算每个点到多边形各边的距离, 返回一个形状为 (N,) 的数组，表示每个点到多边形的最短距离（无符号）
    distances = np.array([
        min(point_to_line_segment_distance(p, poly_points[i], poly_points[(i + 1) % n])
            for i in range(n))
        for p in points
    ])
    
    # 一个布尔数组，形状为 (N,)，True 表示点在多边形内部，False 表示外部
    inside = np.array([is_inside_polygon(point[0], point[1], poly_x, poly_y) for point in points])
    
    # 如果 inside[i] == True，返回 distances[i]（正距离）
    # 结果是一个形状为 (N,) 的数组，表示每个点的有符号距离
    return np.where(inside, distances, -distances)

def sample_domain(x, y, method='random', n_samples=500, grid_size=(50, 50), dx = 0.1, dy = 0.1):
    """
    使用随机采样或网格采样对域中的点进行采样。

    Parameters:
    - x, y: Lists or arrays of coordinates defining the domain.
    - method: 'random' or 'grid' to specify the sampling method.
    - n_samples: Number of random samples (used only if method='random').
    - grid_size: Tuple specifying the grid size (used only if method='grid').

    Returns:
    - sample_points: A 2D array of sampled points. 形状为 (N, 2) 的 NumPy 数组，包含采样点的坐标 [x, y]
    - signed_distances: Signed distances of the sampled points.  形状为 (N,) 的 NumPy 数组，包含每个采样点到域边界的带符号距离
    """
    # 扩展边界（通过 dx 和 dy）确保采样区域稍大于原始多边形，包含边界附近区域。
    x_min, x_max = min(x) - dx, max(x) + dx
    y_min, y_max = min(y) - dy, max(y) + dy

    if method == 'random':
        # Random sampling
        x_samples = np.random.uniform(x_min, x_max, n_samples)
        y_samples = np.random.uniform(y_min, y_max, n_samples)
        sample_points = np.column_stack((x_samples, y_samples))

    elif method == 'grid':
        # Grid sampling
        grid_size_x, grid_size_y = grid_size
        x_grid = np.linspace(x_min, x_max, grid_size_x)
        y_grid = np.linspace(y_min, y_max, grid_size_y)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        sample_points = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))

    else:
        raise ValueError("Method must be 'random' or 'grid'.")

    # Compute signed distances using the vectorized function 使用矢量化函数计算有符号距离
    signed_distances = compute_signed_distance_vectorized(sample_points, x, y)

    return sample_points, signed_distances
