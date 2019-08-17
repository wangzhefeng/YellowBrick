#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-17 21:56:00
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import pandas as pd
import sys
from yellowbrick.features import Rank2D
from yellowbrick.features import JointPlotVisualizer
print("=" * 7)
print("My Env:")
print("=" * 7)
print(sys.version)
print("=" * 40)



# data
data = pd.read_csv("/Users/zfwang/project/YellowBrick/data/bikeshare/bikeshare.csv")
X = data[["season", "month", "hour", "holiday", "weekday", 
          "workingday", "weather", "temp", "feelslike", 
          "humidity", "windspeed"]]
y = data["riders"]


# 探索性特征分析
visualizer = Rank2D(algorithm = "pearson")
visualizer.fit_transform(X)
visualizer.poof()


visualizer = JointPlotVisualizer(feature = "temp", target = "feelslike")
visualizer.fit(X["temp"], X["feelslike"])
visualizer.poof()
