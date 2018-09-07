# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 11:31:04 2018

@author: t-madima
"""

import numpy as np
import matplotlib.pyplot as plt

folder_names = ["dqn__g0.85__lr0.0005__mean__0",
                "dqn__g0.85__lr0.0005__max__0",
                "hra__g0.99__lr0.001__mean__0",
                "hra__g0.99__lr0.001__max__0",
                "hra__g0.99__lr0.001__eps500__max__bellman__0",
                "hra__g0.99__lr0.001__eps500__max__sarsa__0"]
names = ["DQN (mean)", "DQN (max)",
         "HRA (mean)", "HRA (inconsistent max)",
         "HRA (consistent max)", "HRA (online SARSA)"]

handles = []
for i in range(len(folder_names)):
  data = np.genfromtxt(folder_names[i] + '/steps.csv', delimiter=',')
  data = np.delete(data, (0), axis=0)
  h, = plt.plot(data[:, 0], data[:, 1], label=names[i])
  handles.append(h)
plt.xlabel("Episodes")
plt.ylabel("Steps to Success")
plt.legend(handles=handles)
plt.show()

handles = []
for i in range(len(folder_names)):
  data = np.genfromtxt(folder_names[i] + '/scores.csv', delimiter=',')
  data = np.delete(data, (0), axis=0)
  h, = plt.plot(data[:, 0], data[:, 1], label=names[i])
  handles.append(h)
plt.xlabel("Episodes")
plt.ylabel("Episode Score")
plt.legend(handles=handles)
plt.show()