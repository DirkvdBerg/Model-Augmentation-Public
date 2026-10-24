"""Noise floor in metres per stage axis (Section V, metrics).

Question: how large is the noise floor RMS(y - y0) of eq:setup_rms, and does it change between records?
Unlike the other scripts in this folder, it reads the final evaluation, not the dataset: the rows with
model 'floor' of Results/24-10-2026/CSV/res_errors.csv (all splits, including the records evaluated with
the multisine added), which hold the floor over the same samples as the model errors.
Output: Results/setup_noise_floor.csv; macros SetupFloor{Min,Max}{X,Y} [m], X over both X actuators.
"""
import csv
import os

import numpy as np

import _setup_common as sc

ERRORS = os.path.join(sc.RESULTS, '24-10-2026', 'CSV', 'res_errors.csv')


def main():
    with open(ERRORS, newline='') as fh:
        rows = [r for r in csv.DictReader(fh) if r['model'] == 'floor']
    x = np.array([[float(r['rms_X1']), float(r['rms_X2'])] for r in rows])
    y = np.array([float(r['rms_Y']) for r in rows])
    sc.write_csv('noise_floor', ['split', 'record', 'rms_X1', 'rms_X2', 'rms_Y'],
                 [[r['split'], r['record'], r['rms_X1'], r['rms_X2'], r['rms_Y']] for r in rows])
    print('%d records; X %.3e to %.3e m; Y %.3e to %.3e m' % (len(rows), x.min(), x.max(), y.min(), y.max()))
    sc.set_macros(os.path.splitext(os.path.basename(__file__))[0], {
        'SetupFloorMinX': (sc.sci(x.min()), 'smallest noise floor RMS(y - y0) on X1 and X2, all records [m]'),
        'SetupFloorMaxX': (sc.sci(x.max()), 'largest noise floor RMS(y - y0) on X1 and X2, all records [m]'),
        'SetupFloorMinY': (sc.sci(y.min()), 'smallest noise floor RMS(y - y0) on Y, all records [m]'),
        'SetupFloorMaxY': (sc.sci(y.max()), 'largest noise floor RMS(y - y0) on Y, all records [m]'),
    })


if __name__ == '__main__':
    main()
