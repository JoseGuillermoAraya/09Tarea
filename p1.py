#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import (leastsq, curve_fit)
import pdb

''' Script que deriva la constante de Hubble incluyendo su intervalo de
co nfianza al 95 "%" a partir de los datos en data/hubble_original.dat '''


def mostrar_datos(distancia, vel, H1, H2, H_prom):
    ''' recibe el arreglo de distancias y velocidades y las grafica'''
    ax, fig = plt.subplots()
    plt.scatter(distancia, vel, label="Datos originales")
    fig.plot(distancia, f_modelo_1(H1, distancia), label="modelo con V=D*H")
    fig.plot(f_modelo_2(H2, vel), vel, label="modelo con V/H = D")
    fig.plot(distancia, H_prom*distancia, label="Modelo con H promedio")
    fig.set_title("Datos originales y ajuste lineal")
    fig.set_xlabel("Distancia [Mpc]")
    fig.set_ylabel("Velocidad [Km/s]")
    plt.legend(loc=2)
    plt.savefig("hubble_1.jpg")
    plt.show()


def f_modelo_1(params, x):
    H = params
    return H * x


def f_modelo_2(params, v):
    H=params
    return v / H


def func_a_minimizar_1(x, H):
    params = H
    return f_modelo_1(params, x)


def func_a_minimizar_2(v, H):
    params = H
    return f_modelo_2(params, v)


# Main
data = np.loadtxt("data/hubble_original.dat")
distancia = data[:, 0]
vel = data[:, 1]

a_optimo_1, a_covarianza_1 = curve_fit(func_a_minimizar_1,
                                   distancia, vel, 2)
a_optimo_2, a_covarianza_2 = curve_fit(func_a_minimizar_2,
                                   vel, distancia, 2)
a_prom = (a_optimo_2 + a_optimo_1) / 2
mostrar_datos(distancia, vel, a_optimo_1, a_optimo_2, a_prom)
