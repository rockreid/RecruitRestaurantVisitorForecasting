#!/anaconda3/bin/python
import datetime

col_s=datetime.datetime(2016, 7, 1)
last_col_s=datetime.datetime(2016,11,25)

while(col_s<=last_col_s):
    end_date = col_s + datetime.timedelta(days=10)
    col_e=col_s + datetime.timedelta(days=27)
    val_s=col_s + datetime.timedelta(days=28)
    val_e=col_s + datetime.timedelta(days=34)

    print(col_s, col_e, val_s, val_e)
    col_s=col_s + datetime.timedelta(days=7)

