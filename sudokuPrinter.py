import pyautogui as pg


def Print(matrix):
    result = []
    finalresult = []
    for i in range(9):
        result.append(matrix[i])

    for lists in result:
        for num in lists:
            finalresult.append(str(num))

    # 用于计数，满9换行
    counter = []

    for num in finalresult:
        # 控制键盘输入数字
        pg.press(num)
        # 控制键盘右移
        pg.hotkey('right')
        # 满一行后切换到到下一行第一个位置
        counter.append(num)
        if len(counter) % 9 == 0:
            pg.hotkey('down')
            pg.hotkey('left')
            pg.hotkey('left')
            pg.hotkey('left')
            pg.hotkey('left')
            pg.hotkey('left')
            pg.hotkey('left')
            pg.hotkey('left')
            pg.hotkey('left')
