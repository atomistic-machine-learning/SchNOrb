#!/usr/bin/env python
import argparse
import os
import tempfile
from base64 import b64decode
from shutil import rmtree

import numpy as np
from ase.db import connect
from tqdm import tqdm

import schnetpack as spk
import schnorb as stl


def get_number_orbitals(database):
    basis_def = database.metadata['basisdef']
    basis_def = np.array(basis_def)
    n_orbitals = np.zeros(basis_def.shape[0], dtype=np.int)

    for i in range(basis_def.shape[0]):
        n_orbitals[i] = int(np.count_nonzero(basis_def[i, :, 2]))

    return n_orbitals


def get_average_energies(database, n_orbitals):
    atype_entries = n_orbitals.shape[0]

    mean_orb_energies = np.zeros((atype_entries, np.max(n_orbitals)))
    atype_count = np.zeros(atype_entries)

    for row in tqdm(database.select(), ncols=120):
        # Convert from binary back to float
        shape = row.data['_shape_hamiltonian']
        dtype = row.data['_dtype_hamiltonian']
        ham = np.frombuffer(b64decode(row.data['hamiltonian']), dtype=dtype)
        ham = ham.reshape(shape)
        energies = np.diag(ham)

        # Get atom types
        atypes = row.numbers
        pos = 0
        for atom in atypes:
            inc = n_orbitals[atom]
            mean_orb_energies[atom, :inc] += energies[pos:pos + inc]
            atype_count[atom] += 1
            pos += inc

    for i in range(atype_entries):
        count = atype_count[i]
        if count > 0:
            mean_orb_energies[i] /= count

    return mean_orb_energies


def sub_mean(src, dst, has_forces):
    if has_forces:
        dst = spk.data.AtomsData(dst,
                                 available_properties=['hamiltonian', 'overlap',
                                                       'energy', 'forces'])
    else:
        dst = spk.data.AtomsData(dst,
                                 available_properties=['hamiltonian', 'overlap',
                                                       'energy'])

    print('Calculate statistics')
    loader = spk.data.AtomsLoader(src, batch_size=100,
                                  num_workers=4, pin_memory=True)
    print(src.available_properties)
    mean, stddev = loader.get_statistics('energy', True)
    mean = float(mean['energy'].numpy())

    with connect(src.dbpath) as database:
        n_orbitals = get_number_orbitals(database)
        mean_orb_energies = get_average_energies(database, n_orbitals)

    print('Write final DB')
    for i in tqdm(range(len(src))):
        ats, props = src.get_properties(i)
        H = props['hamiltonian'].numpy()
        S = props['overlap'].numpy()
        E = props['energy'].numpy()
        if has_forces:
            F = props['forces'].numpy()

            new_props = {
                'hamiltonian': H, 'overlap': S, 'energy': E - mean, 'forces': F,
            }
        else:
            new_props = {
                'hamiltonian': H, 'overlap': S, 'energy': E - mean
            }
        dst.add_system(ats, **new_props)

    with connect(src.dbpath) as conn1:
        md = conn1.metadata

    md['mean_energy'] = mean
    md['orbital_energies'] = mean_orb_energies.tolist()

    with connect(dst.dbpath) as conn2:
        pass
    conn2.metadata = md


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=str, help='Path to ASE database.')
    parser.add_argument('outputs', type=str,
                        help='Path to fhi-AIMS calculation directories or ORCA output files.')
    parser.add_argument('--basisdef', type=str, default=None,
                        help='Path to ASE database.')
    parser.add_argument('--orbital_energies', default=None,
                        help='Path to single atom orbital energies')
    parser.add_argument('--noout', action='store_true',
                        help='No output files available.')
    parser.add_argument('--check_convergence', action='store_true',
                        help='Filter non converged calculations.')
    parser.add_argument('--mindist', default=None, type=float,
                        help='Screen structures for short distances')
    parser.add_argument('--format', default='aims', type=str, choices=['aims', 'orca'],
                        help='Input file format.')
    parser.add_argument('--forces', action='store_true',
                        help='Extract forces. Currently only implemented for ORCA.')
    parser.add_argument('--energy_offset', type=float, default=None,
                        help='Remove offset from molecule energies.')
    args = parser.parse_args()

    tmpdir = tempfile.mkdtemp(dir='/tmp')
    tmpsrc = os.path.join(tmpdir, 'tmpdb.db')

    outputs = [os.path.join(args.outputs, d) for d in os.listdir(args.outputs)]

    if args.basisdef is None:
        if args.format == 'aims':
            basisdef = stl.data.extract_basis_definition_aims(outputs)
        elif args.format == 'orca':
            basisdef = stl.data.extract_basis_definition_orca(outputs)
        else:
            raise NotImplementedError(
                'Unrecognized reference data format {:s}'.format(args.format)
            )
    else:
        basisdef = np.load(args.basisdef)

    if args.orbital_energies is None:
        orbital_energies = None
    else:
        orbital_energies = np.load(args.orbital_energies)

    if args.format == 'aims':
        data = stl.data.AimsHamiltonianParser(tmpsrc, basisdef,
                                              check_convergence=args.check_convergence,
                                              min_dist=args.mindist,
                                              orbital_energies=orbital_energies,
                                              noout=args.noout,
                                              forces=args.forces,
                                              energy_offset=args.energy_offset)

    elif args.format == 'orca':
        data = stl.data.OrcaHamiltonianParser(tmpsrc, basisdef,
                                              check_convergence=args.check_convergence,
                                              min_dist=args.mindist,
                                              orbital_energies=orbital_energies,
                                              forces=args.forces,
                                              energy_offset=args.energy_offset)
    else:
        raise NotImplementedError(
            'Unrecognized reference data format {:s}'.format(args.format))

    data.parse_directories(outputs)

    # update statistics
    atoms_data = data.atomsdata
    subtract_mean = sub_mean(atoms_data, args.database, args.forces)

    rmtree(tmpdir)
