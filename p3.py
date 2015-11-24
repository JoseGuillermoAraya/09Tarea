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
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.errorbar(banda_i, banda_z, xerr=error_i, yerr=error_z, fmt="o",
                 label="Datos originales y ajuste lineal")
    x = np.linspace(-100, 500, 600)
    ax1.plot(x, c[1] + x*c[0], color="r", label="ajuste lineal")
    ax1.set_title("Datos originales")
    ax1.set_xlabel("Flujo banda i [$10^{-6}Jy$]")
    ax1.set_ylabel("Flujo banda z [$10^{-6}Jy$]")
    plt.legend(loc=2)
    plt.savefig("bandas.jpg")


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
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.hist(pendiente, bins=30)
    ax2.axvline(c[0], color='r')
    ax2.set_title("Simulacion de Montecarlo (pendientes)")
    ax2.set_xlabel("pendiente [adimensional]")
    ax2.set_ylabel("frecuencia")
    plt.savefig("mc.jpg")

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.hist(pendiente, bins=30)
    ax3.axvline(c[0], color='r')
    ax3.set_title("Simulacion de Montecarlo (coef de posicion)")
    ax3.set_xlabel("coef de posicion [adimensional]")
    ax3.set_ylabel("frecuencia")
    plt.savefig("mc_2.jpg")

    pendiente = np.sort(pendiente)
    cte = np.sort(cte)
    limite_bajo_1 = pendiente[int(Nmc * 0.025)]
    limite_alto_1 = pendiente[int(Nmc * 0.975)]
    limite_bajo_2 = cte[int(Nmc * 0.025)]
    limite_alto_2 = cte[int(Nmc * 0.975)]
    print """El intervalo de confianza al
             95% para la pendiente es: [{}:{}]""".format(limite_bajo_1,
                                                         limite_alto_1)
    print """El intervalo de confianza al
             95% para el coef de posicion es: [{}:{}]""".format(limite_bajo_2,
                                                                limite_alto_2)


data = np.loadtxt("data/DR9Q.dat", usecols=(80, 81, 82, 83))
banda_i = data[:, 0] * 3.631
error_i = data[:, 1] * 3.631
banda_z = data[:, 2] * 3.631
error_z = data[:, 3] * 3.631
c = np.polyfit(banda_i, banda_z, 1)
print(c)
intervalo_confianza = mc(banda_i, error_i, banda_z, error_z, c)
mostrar_datos(banda_i, error_i, banda_z, error_z, c)
plt.show()
