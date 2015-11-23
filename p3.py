#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import (leastsq, curve_fit)
import pdb
''' Script que deriva la constante de Hubble incluyendo su intervalo de
co nfianza al 95 "%" a partir de los datos en data/DR9Q.dat '''


def mostrar_datos(banda_i, error_i, banda_z, error_z):
    '''grafica los datos'''
    ax, fig = plt.subplots()
    plt.errorbar(banda_i, banda_z, xerr=error_i, yerr=error_z, fmt="o",
                 label="Datos originales")
    fig.set_title("Datos originales")
    fig.set_xlabel("Flujo banda i [$10^{-6}Jy$]")
    fig.set_ylabel("Flujo banda z [$10^{-6}Jy$]")
    plt.legend(loc=2)
    plt.savefig("bandas.jpg")
    plt.show()


data = np.loadtxt("data/DR9Q.dat", usecols=(80, 81, 82, 83))
banda_i = data[:, 0] *  3.631
error_i = data[:, 1] *  3.631
banda_z = data[:, 2] *  3.631
error_z = data[:, 3] *  3.631
mostrar_datos(banda_i, error_i, banda_z, error_z)
