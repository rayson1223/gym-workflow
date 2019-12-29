import PyGnuplot as gp
import numpy as np


def main():
    # X = np.arange(10)
    # Y = np.sin(X / (2 * np.pi))
    # Z = Y ** 2.0
    X = Y = np.array(list(range(10)))
    Z = np.array(X ** 3)
    Z2 = np.array(X ** 4)
    gp.s([X, Y, Z, Z2])
    gp.c('splot "tmp.dat" using 1:2:3 with lines')
    gp.c('replot "tmp.dat" u 1:2:4 w lines')
    # gp.c('splot "tmp.dat" using 1:2:3:($3+$4) with zerrorfill')
    # gp.c('replot "tmp.dat" u 1:3 w lp')
    gp.p('myfigure.ps')


if __name__ == "__main__":
    main()
