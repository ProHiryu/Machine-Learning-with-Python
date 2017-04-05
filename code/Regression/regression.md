# Regression

- [reference](http://pandas.pydata.org/pandas-docs/stable/indexing.html)
- `dataframe.shift(num)` just move the dataset num times rightwards
- `dataframe.iloc[num]` Selection by Position - is primarily integer position based (from 0 to length-1 of the axis)
- `dataframe.loc[text]` Selection by Label
- python list - `list[-1]` the last element of the list
- python date:

  - time string : `time.ctime()` 'Mon Dec 17 21:02:55 2012'
  - datetime tuple(datetime obj) : `datetime.now()` `datetime.datetime(2012, 12, 17, 21, 3, 44, 139715)`
  - time tuple(time obj) : `time.struct_time(tm_year=2008, tm_mon=11, tm_mday=10, tm_hour=17, tm_min=53, tm_sec=59, tm_wday=0, tm_yday=315, tm_isdst=-1)`
  - timestamp : 时间戳类型:自1970年1月1日(00:00:00 GMT)以来的秒数
  - ![NOT FOUND!](http://s9.sinaimg.cn/large/b09d4602xd10ea8f9ab88&690 "change relation")

- **python iteration `np.nan for _ in range()` means nothing,the variable is not going to be used** - [nan, nan, nan, nan, nan, 781.29480101781269]

- python list : xs*ys means every element in xs times ys by the order of index
