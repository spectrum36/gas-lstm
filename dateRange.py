import pandas as pd

def getDateRange(date, days, forwardBack=False):
    date = date.split("-")
    #I don't get this line but it works
    date = [int(date) for date in date]
    dates = [str(date[0]) + f"-{date[1]:02d}-{date[2]:02d}"]
    if not forwardBack:
        for i in range(days):
            date[2] = date[2] - 1
            if date[2] == 0:
                date[1] = date[1] - 1
                if date[1] in {1, 3, 5, 7, 8, 10, 12}:
                    date[2] = 31
                elif date[1] in {4, 6, 9, 11}:
                    date[2] = 30
                elif date[1] == 2:
                    if date[0] % 4 == 0:
                        date[2] = 29
                    else:
                        date[2] = 28
                else:
                    date[0] = date[0] - 1
                    date[1] = 12
                    date[2] = 31
    else:
        for i in range(days):
            date[2] = date[2] + 1
            if (date[2] == 32 and date[1] in {1, 3, 5, 7, 8, 10, 12}) or (date[2] == 31 and date[1] in {4, 6, 9, 11}) or (date[0] % 4 == 0 and date[1] == 2 and date[2] == 30) or (date[0] % 4 != 0 and date[1] == 2 and date[2] == 29):
                date[1] = date[1] + 1
                date[2] = 1
                if date[1] == 13:
                    date[0] = date[0] + 1
                    date[1] = 1

    return(str(date[0]) + f"-{date[1]:02d}-{date[2]:02d}")
