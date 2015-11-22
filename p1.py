#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pdb

''' Script que deriva la constante de Hubble incluyendo su intervalo de
co nfianza al 95 "%" a partir de los datos en data/hubble_original.dat '''


def mostrar_datos(distancia, vel):
    ax, fig = plt.subplots()
    plt.scatter(distancia, vel, label="Datos originales")
    fig.set_title("Datos originales y ajuste lineal")
    fig.set_xlabel("Distancia [Mpc]")
    fig.set_ylabel("Velocidad [Km/s]")
    plt.legend(loc=2)
    plt.show()

# Main
data = np.loadtxt("data/hubble_original.dat")
distancia = data[:, 0]
vel = data[:, 1]
mostrar_datos(distancia, vel)
