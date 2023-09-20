import matplotlib.pyplot as p
import numpy as np

def plot(d, ua, toffset, T, b, series, response, nLamb, mV, snr):
    plots = [None] * 9
    Lims = 20  # lims in km
    extent = [-response['D']/2000, response['D']/2000, -response['D']/2000, response['D']/2000]
    
    plots[0] = p.figure(1)
    p.clf()
    p.imshow(series['O1'], extent=extent)
    p.xlabel('Distance [km]')
    p.ylabel('Distance [km]')
    p.xlim([-2*d/1000, 2*d/1000])
    p.ylim([-2*d/1000, 2*d/1000])
    p.gray()
    p.title('Circular Object')

    plots[1] = p.figure(2)
    p.clf()
    p.imshow(series['I1'], extent=extent)
    p.gray()
    p.xlabel('Distance [km]')
    p.ylabel('Distance [km]')
    p.xlim([-Lims, Lims])
    p.ylim([-Lims, Lims])
    p.title('Diffraction pattern, circular object')

    plots[2] = p.figure(3)
    p.clf()
    p.imshow(series['I1s'], extent=extent)
    p.gray()
    p.xlabel('Distance [km]')
    p.ylabel('Distance [km]')
    p.xlim([-Lims, Lims])
    p.ylim([-Lims, Lims])
    p.title('Chromatic diffraction pattern, circular: ' + response['tipo'] + ', Longs. Onda: ' + str(np.round(nLamb)))

    plots[3] = p.figure(4)
    p.clf()
    p.imshow(series['I1n'], extent=extent)
    p.gray()
    p.xlabel('Distance [km]')
    p.ylabel('Distance [km]')
    p.xlim([-Lims, Lims])
    p.ylim([-Lims, Lims])
    p.title('Diffraction pattern, circular, noise mV: ' + str(mV) + ', SNR: ' + str(np.round(snr)))

    plots[4] = p.figure(5)
    p.clf()
    p.plot(series['xc']/1000, series['yc'])
    p.xlim([-Lims, Lims])
    p.xlabel('Distance [km]')
    p.ylabel('Normalized intensity')
    p.title('Circular diffraction profile')

    plots[5] = p.figure(6)
    p.clf()
    p.plot(series['xb']/1000, series['yb'])
    p.xlim([-Lims, Lims])
    p.xlabel('Distance [km]')
    p.ylabel('Normalized intensity')
    p.title('Binary diffraction profile')

    plots[6] = p.figure(7)
    if 'x1' in series:
        p.clf()
        p.plot(series['x1'], series['y1'])
        p.xlabel('Time [s]')
        p.ylabel('Normalized intensity without noise')
        p.xlim([-1, 1])
    
    plots[7] = p.figure(8)
    if 'x2' in series:
        p.clf()
        p.plot(series['x2'], series['y2'], '.-')
        p.xlabel('Time [s]')
        p.ylabel('Normalized intensity with noise')

    plots[8] = response
    return tuple(plots)