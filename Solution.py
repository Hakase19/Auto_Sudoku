def solver(grid):
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                continue
            for k in range(1, 10):
                if valid(grid, i, j, k):  # 检查此选择是否合法
                    grid[i][j] = k
                    if solver(grid):  # 判断此选择是否有解
                        return grid
                    grid[i][j] = 0  # 回溯
            return   # 某一格无合法选择，无解
    return grid  # 无 '.' ，完成求解


def valid(grid, i, j, c):
    for k in range(9):  # 检查横纵
        if grid[i][k] == c or grid[k][j] == c:
            return
    bi = i // 3 * 3
    bj = j // 3 * 3
    for k in range(3):  # 检查块
        if grid[bi + k][bj] == c or grid[bi + k][bj + 1] == c or grid[bi + k][bj + 2] == c:
            # if c in board[bi+k][bj:bj+3]:  # 简写
            return
    return grid
