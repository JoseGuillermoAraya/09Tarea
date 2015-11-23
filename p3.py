#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import (leastsq, curve_fit)
import pdb
''' Script que deriva la constante de Hubble incluyendo su intervalo de
co nfianza al 95 "%" a partir de los datos en data/DR9Q.dat '''


def mostrar_datos(banda_i, error_i, banda_z, error_z, c):
    '''grafica los datos originales con sus errores asociados,
    grafica el ajuste lineal polyfit'''
    ax, fig = plt.subplots()
    fig.errorbar(banda_i, banda_z, xerr=error_i, yerr=error_z, fmt="o",
                 label="Datos originales y ajuste lineal")
    x = np.linspace(-100, 500, 600)
    fig.plot(x, c[1] + x*c[0], color="r", label="ajuste lineal")
    fig.set_title("Datos originales")
    fig.set_xlabel("Flujo banda i [$10^{-6}Jy$]")
    fig.set_ylabel("Flujo banda z [$10^{-6}Jy$]")
    plt.legend(loc=2)
    plt.savefig("bandas.jpg")
    plt.show()


def mc(banda_i, error_i, banda_z, error_z, c_0):
    '''realiza una simulación de montecarlo para obtener el intervalo de
    confianza del 95%'''
    Nmc = 10000
    cte = np.zeros(Nmc)
    pendiente = np.zeros(Nmc)
    for j in range(Nmc):
        r = np.random.normal(0, 1, size=len(banda_i))
        muestra_i = banda_i + error_i * r
        muestra_z = banda_z + error_z * r
        pendiente[j], cte[j] = np.polyfit(muestra_i, muestra_z, 1)
    ax2, fig2 = plt.subplots()
    fig2.hist(pendiente, bins=30)
    fig2.axvline(c[0], color='r')
    fig2.set_title("Simulacion de Montecarlo")
    fig2.set_xlabel("pendiente [adimensional]")
    fig2.set_ylabel("frecuencia")
    plt.savefig("mc.jpg")
    pendiente = np.sort(pendiente)
    limite_bajo_1 = pendiente[int(Nmc * 0.025)]
    limite_alto_1 = pendiente[int(Nmc * 0.975)]
    print "El intervalo de confianza al 95% es: [{}:{}]".format(limite_bajo_1,
                                                                limite_alto_1)


data = np.loadtxt("data/DR9Q.dat", usecols=(80, 81, 82, 83))
banda_i = data[:, 0] * 3.631
error_i = data[:, 1] * 3.631
banda_z = data[:, 2] * 3.631
error_z = data[:, 3] * 3.631
c = np.polyfit(banda_i, banda_z, 1)
print(c)
mostrar_datos(banda_i, error_i, banda_z, error_z, c)
intervalo_confianza = mc(banda_i, error_i, banda_z, error_z, c)
