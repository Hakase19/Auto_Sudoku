def solver(grid):
    for i in range(9):
        for j in range(9):
            # 寻找空格
            if grid[i][j] != 0:
                continue
            for k in range(1, 10):
                # 检查该选择是否合法
                if valid(grid, i, j, k):
                    grid[i][j] = k
                    if solver(grid):# 判断此选择是否有解
                        return grid
                    grid[i][j] = 0  # 回溯操作
            return   # 若某一格无合法选择，无解
    return grid  # 无数字0，即完成求解


def valid(grid, i, j, c):
    for k in range(9):
        # 检查横纵是否合法
        if grid[i][k] == c or grid[k][j] == c:
            return
    bi = i // 3 * 3
    bj = j // 3 * 3
    for k in range(3):
        # 在每个九宫格内检查是否合法
        if grid[bi + k][bj] == c or grid[bi + k][bj + 1] == c or grid[bi + k][bj + 2] == c:
            return
    return grid
