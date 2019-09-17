import logging
import os

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.units import Ha, Bohr
from tqdm import tqdm

import schnetpack as spk
from schnorb import SchNOrbProperties
from schnorb.rotations import OrcaRotator, rand_rotation_matrix
from schnorb.utils import check_nan_np

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def extract_basis_definition_aims(output_dirs):
    """
    Format: orb_idx,type,n,l,m

    :param geometry:
    :return:
    """

    for dir in output_dirs:
        if os.path.isdir(dir):
            break

    basis_indices = os.path.join(dir, 'basis-indices.out')
    geometry = os.path.join(dir, 'geometry.in')

    atoms = read(geometry, format='aims')
    basis_types = {'atomic': 0, 'ionic': 1, 'hydro': 2}

    # load raw data, transform strings into bool flags and skip first two lines
    basis_definition = np.genfromtxt(basis_indices, dtype=np.int,
                                     skip_header=2, encoding=None,
                                     converters={1: lambda x: np.int(
                                         basis_types['atomic'])})
    basis_definition[:, 0] -= 1
    basis_definition[:, 2] -= 1

    # add atomic basis function index
    uidx, first, count = np.unique(basis_definition[:, 2], return_counts=True,
                                   return_index=True)
    offsets = np.repeat(first, count)
    aidx = basis_definition[:, 0] - offsets

    basis_def = np.zeros((np.max(atoms.numbers) + 1, np.max(aidx) + 1, 5),
                         dtype=np.int)
    for i, z in enumerate(atoms.numbers):
        zidx = basis_definition[:, 2] == i
        bdi = basis_definition[zidx]
        # print(i,z, bdi, basis_def.shape)
        if basis_def[z, 0, 1] == 0:
            basis_def[z, :bdi.shape[0], 1:] = bdi[:, np.array([1, 3, 4, 5])]
            basis_def[z, :, 0] = np.arange(np.max(aidx) + 1)
        else:
            assert np.all(basis_def[z, :bdi.shape[0], 1:] == bdi[:, np.array(
                [1, 3, 4, 5])])

    return basis_def


def extract_basis_definition_orca(output_files):
    """
    Construct basis set definition for orca

    Args:
        output_files (list(str)): List of output files. Uses first file with
            .log extension.

    Returns:
        np.array: Basis set definitions in the form
            n_atom_types x n_basis x (idx, type, n, l, m)

    """
    for file in output_files:
        if os.path.exists(file) and os.path.splitext(file)[-1] == '.log':
            break

    # Get ORCA geometry and basis set definition (list of ls per atom)
    basis_parser = OrcaDataParser(properties=['atoms', 'basis'])
    basis_parser.parse_file(file)
    data = basis_parser.get_parsed()

    # Construct atoms object
    atoms = Atoms(*data['atoms'])

    nlm, coefficients = data['basis']

    # Find maximum basis length
    max_atom_basis = 0
    for element in nlm.keys():
        max_atom_basis = max(nlm[element].shape[0], max_atom_basis)

    # Construct and populate output array
    basis_def = np.zeros((np.max(atoms.numbers) + 1, max_atom_basis, 5), dtype=np.int)

    for atom_type in nlm.keys():
        n_entries = nlm[atom_type].shape[0]
        basis_def[atom_type, :n_entries, 2:] = np.array(nlm[atom_type])
        basis_def[atom_type, :n_entries, 0] = np.arange(n_entries)

    return basis_def, coefficients


class HamiltonianParserException(Exception):
    pass


class HamiltonianParser:

    def __init__(self, dbpath, basis_definition, orbital_energies=None,
                 check_convergence=False, check_files=False, min_dist=None,
                 minimal_basis=False, forces=False, energy_offset=None):

        properties = ['hamiltonian', 'overlap', 'energy']

        self.dbpath = dbpath
        self.check_convergence = check_convergence
        self.min_dist = min_dist
        self.basis_definition = basis_definition
        self.orbital_energies = orbital_energies
        self.minimal_basis = minimal_basis
        self.atomsdata = spk.data.AtomsData(
            dbpath, available_properties=properties
        )
        self.forces = forces
        self.energy_offset = energy_offset

    def parse_directories(self, data_dirs, append=False, buffer_size=1000):
        at_buffer = []
        prop_buffer = []

        for path in tqdm(sorted(data_dirs), ncols=120):

            if os.path.exists(path):
                logging.debug('Reading data: ' + path)
                atoms, properties = self.parse_molecule(path)

                if properties is not None:
                    at_buffer.append(atoms)
                    prop_buffer.append(properties)

                    if len(at_buffer) >= buffer_size:
                        self.atomsdata.add_systems(at_buffer, prop_buffer)
                        at_buffer = []
                        prop_buffer = []

        if len(at_buffer) != 0:
            self.atomsdata.add_systems(at_buffer, prop_buffer)

        # Set minimal basis conventions for Quambos
        if self.minimal_basis:
            self.basis_definition = self.basis_definition[:, :5, :]
            self.basis_definition[1, 1:, :] = 0.0

            if self.orbital_energies is not None:
                self.orbital_energies = self.orbital_energies[:, :5]

        if type(self.basis_definition) == tuple:
            basis_def, coefficents = self.basis_definition
            metadata = {
                'basisdef': basis_def.tolist(),
                'basiscoeff': {k: coefficents[k].tolist() for k in coefficents}
            }
        else:
            metadata = {
                'basisdef': self.basis_definition.tolist()
            }

        if self.energy_offset is not None:
            metadata['energy_offset'] = self.energy_offset

        if self.orbital_energies is not None:
            metadata['orbital_energies'] = self.orbital_energies.tolist()

        self.atomsdata.set_metadata(metadata)

    def parse_molecule(self, path):

        # check whether files exist and are the proper format.
        if not self._check_files(path):
            return None, None

        # read geometry data
        atoms = self._parse_geometry(path)

        # check if input files are ok
        if not self._check_input(atoms, path):
            return atoms, None

        if self.check_convergence:
            if not self._check_convergence(path):
                return atoms, None

        # TODO: Check for energy
        energy = self._parse_energy(path)

        if energy is None:
            return atoms, None

        if self.energy_offset is not None:
            energy -= self.energy_offset

        # parse everything
        matrices = self._parse_matrices(path, atoms)
        H, S = matrices[0], matrices[1]

        properties = {
            'hamiltonian': H.astype(np.float32),
            'overlap': S.astype(np.float32),
            'energy': energy.astype(np.float64)
        }

        if self.forces:
            forces = self._parse_forces(path)
            properties['forces'] = forces.astype(np.float32)

        return atoms, properties

    def _check_input(self, atoms, path):
        try:
            # If requested check for structural sanity
            if self.min_dist is not None:
                self._check_structure(atoms)
            # Check for convergence of calculations
            if self.check_convergence:
                self._check_convergence(path)
        except HamiltonianParserException as e:
            logging.warning(str(e))
            return False
        return True

    def _check_structure(self, atoms):
        distances = atoms.get_all_distances()
        min_dist = np.min(distances[distances != 0])
        if min_dist < self.min_dist:
            print(self.min_dist)
            raise HamiltonianParserException(
                'Distance of {:f} below threshold of {:f} detected.'.format(
                    min_dist, self.min_dist))

    def _check_files(self, path):
        raise NotImplementedError

    def _parse_geometry(self, path):
        raise NotImplementedError

    def _check_convergence(self, path):
        raise NotImplementedError

    def _parse_matrices(self, path, atoms):
        raise NotImplementedError

    def _parse_energy(self, path):
        raise NotImplementedError

    def _parse_forces(self, path):
        raise NotImplementedError


class AimsHamiltonianParser(HamiltonianParser):
    # Base names of FHI aims output files
    files = {
        'hamiltonian': 'hamiltonian.out',
        'geometry': 'geometry.in',
        'overlap': 'overlap-matrix.out',
        'energy': ['total_energy.dat'],
        'outfile': ['output', 'OUTPUT', 'aims.out']
        # Since output names are inconsistent
    }

    def __init__(self, dbpath, basis_definition, orbital_energies=None,
                 check_convergence=False, min_dist=None,
                 outfile=None, noout=False, minimal_basis=False, forces=False,
                 energy_offset=None):
        super(AimsHamiltonianParser, self).__init__(
            dbpath, basis_definition,
            orbital_energies=orbital_energies,
            check_convergence=check_convergence,
            min_dist=min_dist, minimal_basis=minimal_basis, forces=forces,
            energy_offset=energy_offset)

        # Determine output file for convergence check
        if noout:
            self.outfile = None
        elif outfile is None:
            self.outfile = self.files['outfile']
        else:
            self.outfile = [outfile]

    def _check_files(self, path):
        """
        Check whether all fhi-AIMS files are present in the directory.
        Stupid exceptions...

        :param path:
        :return:
        """
        if not os.path.isdir(path):
            return False

        for key, file in self.files.items():
            if key == 'outfile':
                if self.outfile is not None:
                    file = self.outfile
                else:
                    continue
            if key == 'energy' and self.outfile is not None:
                continue

            if type(file) is list:
                exists = False

                for option in file:
                    outfile = os.path.join(path, option)
                    if os.path.exists(outfile):
                        exists = True
                if not exists:
                    return False
            else:
                file = os.path.join(path, file)
                if not os.path.exists(file):
                    return False

        return True

    def _parse_geometry(self, path):
        """
        Read in FHI aims geometry from 'geometry.in' file via ase and store

        Args:
            wdir: Directory with FHI aims files

        Returns:
            atoms: Atoms object of current molecule
        """
        geometry_path = os.path.join(path, self.files['geometry'])

        # This should not raise an error, as empty directories might be present. _check_files will just skip them.
        # if not os.path.exists(geometry_path):
        #    raise FileNotFoundError('Could not open {:s}'.format(geometry_path))

        atoms = read(geometry_path, format='aims')
        return atoms

    def _check_convergence(self, path):
        """
        Check whether calculation is converged.
        """
        output_paths = [os.path.join(path, i) for i in self.outfile]

        outfile = None
        for file in output_paths:
            if os.path.exists(file):
                outfile = file

        # Once again, should have been checked with _check_files before
        # if outfile is None:
        #    raise FileNotFoundError('AIMS output not found in outfile options.')

        flag = open(outfile).readlines()[-2].strip()

        if not flag == 'Have a nice day.':
            return False
        else:
            return True

    def _parse_matrices(self, path, atoms):
        """
        mu/nu format: row/col,atom,orb_idx,type,n,l,m

        :param path:
        :param atoms:
        :return:
        """
        hpath = os.path.join(path, self.files['hamiltonian'])
        spath = os.path.join(path, self.files['overlap'])

        Hraw = np.loadtxt(hpath)
        Sraw = np.loadtxt(spath)
        size = np.max(Hraw[:, 0].astype(np.int))

        mu = Hraw[:, 0].astype(np.int) - 1
        nu = Hraw[:, 1].astype(np.int) - 1

        H = np.zeros((size, size))
        S = np.zeros((size, size))

        H[mu, nu] = Hraw[:, 2]
        H[nu, mu] = Hraw[:, 2]
        S[mu, nu] = Sraw[:, 2]
        S[nu, mu] = Sraw[:, 2]
        return H, S

    def _parse_energy(self, path):
        read_dat = False

        if self.outfile is not None:
            energy_files = self.outfile
        else:
            energy_files = self.files['energy']
            read_dat = True

        output_paths = [os.path.join(path, i) for i in energy_files]
        outfile = None
        for file in output_paths:
            if os.path.exists(file):
                outfile = file

        if outfile is None:
            raise HamiltonianParserException(
                'No output file for energy found in {:f}'.format(path))
        else:
            if read_dat:
                energy = np.loadtxt(outfile)
                energy = np.array([energy]).astype(
                    np.float64) / Ha
            else:
                energy = [line for line in open(outfile, 'r').readlines()
                          if
                          'Total energy of the DFT / Hartree-Fock s.c.f. calculation      :' in line]
                try:
                    energy = np.array([energy[-1].split()[-2]]).astype(
                        np.float64) / Ha
                except:
                    energy = None

        return energy

    def _parse_forces(self, path):

        force_files = self.outfile
        output_paths = [os.path.join(path, i) for i in force_files]
        outfile = None
        for file in output_paths:
            if os.path.exists(file):
                outfile = file

        if outfile is None:
            raise HamiltonianParserException(
                'No output file for energy found in {:f}'.format(path))

        forces = []
        read = False
        with open(outfile, 'r') as of:
            for line in of:
                line = line.strip()
                if line.startswith(
                        'Total atomic forces (unitary forces cleaned) [eV/Ang]:'):
                    forces = []
                    read = True
                elif not line:
                    read = False
                elif read:
                    forces.append([float(i) for i in line.split()[2:5]])

        # Convert to Hartree / Angs
        forces = np.array(forces) / Ha
        return forces


class OrcaHamiltonianParser(HamiltonianParser):
    file_endings = ['.log']

    def __init__(self, dbpath, basis_definition, orbital_energies=None,
                 check_convergence=False, min_dist=None,
                 outfile=None, minimal_basis=False, forces=False, energy_offset=None):
        super(OrcaHamiltonianParser, self).__init__(dbpath, basis_definition,
                                                    orbital_energies,
                                                    check_convergence,
                                                    min_dist,
                                                    minimal_basis=minimal_basis,
                                                    forces=forces,
                                                    energy_offset=energy_offset)

        self.orca_parser = OrcaDataParser()

    def _check_files(self, path):
        ending = os.path.splitext(path)[-1]
        if ending in self.file_endings:
            return True
        else:
            return False

    def _parse_geometry(self, path):
        """
        Get geometries from ORCA output. Since ORCA writes only one file,
        """
        if not os.path.exists(path):
            raise FileNotFoundError('Could not open {:s}'.format(path))

        # Read file and populate parser
        self.orca_parser.parse_file(path)

        atypes, coords = self.orca_parser.get_parsed()['atoms']
        atoms = Atoms(atypes, coords)

        return atoms

    def _check_convergence(self, path):
        """
        Check whether calculation is converged.
        """
        outfile = None
        if os.path.exists(path):
            outfile = path

        if outfile is None:
            raise FileNotFoundError(
                'AIMS output not found in outfile options.')

        flag = open(outfile).readlines()[-2].strip()

        if not flag == '****ORCA TERMINATED NORMALLY****':
            return False
        else:
            return True

    def _parse_matrices(self, path, atoms):
        data = self.orca_parser.get_parsed()
        H = data['hamiltonian']
        S = data['overlap']
        return H, S

    def _parse_energy(self, path):
        data = self.orca_parser.get_parsed()
        energy = data['energy']
        return energy

    def _parse_forces(self, path):
        data = self.orca_parser.get_parsed()
        forces = data['forces']
        return forces


class OrcaOutputParser:
    """
    Basic Orca output parser class. Parses an Orca output file according to the parsers specified in the 'parsers'
    dictionary. Parsed data is stored in an dictionary, using the same keys as the parsers. If a list of formatters is
    provided to a parser, a list of the parsed entries is stored in the ouput dictionary.

    Args:
        parsers (dict[str->callable]): dictionary of OrcaPropertyParser, each with their own OrcaFormatter.
    """

    def __init__(self, parsers):
        self.parsers = parsers
        self.parsed = None

    def parse_file(self, path):
        """
        Open the file and iterate over its lines, applying all parsers. In the end, all data is collected in a
        dictionary.

        Args:
            path (str): path to Orca output file.
        """
        # Reset for new file
        for parser in self.parsers:
            self.parsers[parser].reset()

        with open(path, 'r') as f:
            for line in f:
                for parser in self.parsers:
                    self.parsers[parser].parse_line(line)

        self.parsed = {}

        for parser in self.parsers:
            self.parsed[parser] = self.parsers[parser].get_parsed()

    def get_parsed(self):
        """
        Get parsed data.

        Returns:
            dict[str->list]: Dictionary of data entries according to parser keys.
        """
        return self.parsed


class OrcaFormatter:
    """
    Format raw Orca data collected by an OrcaPropertyParser. Behavior is determined by the datatype option.

    Args:
        position (int): Position to start formatting. If no stop is provided returns only value at position, otherwise
                        all values between position and stop are returned. (Only used for 'vector' mode)
        stop (int, optional): Stop value for range. (Only used for 'vector' mode)
        datatype (str, optional): Change formatting behavior. The possible options are:
                                  'vector': Formats data between position and stop argument, if provided converting it
                                            to the type given in the converter.
                                  'matrix': Formats collected matrix data into the shape of a square, symmetric
                                            numpy.ndarray. Ignores other options.
                                  'basis': Formats basis set data and creates a dictionary with atom types as indices
                                           and the ls of the present basis functions as entries.
        converter (type, optional): Convert data to type. (Only used for 'vector' mode)
    """
    # ls for basis set definitions
    basis_l = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4}

    def __init__(self, position, stop=None, datatype='vector',
                 converter=float, unit=None):
        self.position = position
        self.stop = stop
        self.datatype = datatype
        self.converter = converter
        self.matrix_dim = None
        self.unit = unit

    def format(self, parsed):
        if parsed is None:
            return None
        elif self.datatype == 'vector':
            formatted = self._format_vector(parsed)
        elif self.datatype == 'matrix':
            formatted = self._format_matrix(parsed)
        elif self.datatype == 'basis':
            formatted = self._format_basis(parsed)
        else:
            raise NotImplementedError(
                'Unrecognized data type {:s}'.format(self.datatype))

        if self.unit is not None:
            formatted *= self.unit

        return formatted

    def _format_vector(self, parsed):
        vector = []
        for line in parsed:
            line = line.split()
            if self.stop is None:
                vector.append(self.converter(line[self.position]))
            else:
                vector.append(
                    [self.converter(x) for x in line[self.position:self.stop]])
        vector = np.array(vector)
        return vector

    def _format_matrix(self, parsed):

        n_entries = len(parsed[1].split())

        self.matrix_dim = int(parsed[-1].split()[0]) + 1

        subdata = [parsed[i:i + self.matrix_dim + 1] for i in
                   range(0, len(parsed), self.matrix_dim + 1)]

        matrix = [[] for _ in range(self.matrix_dim)]

        for block in subdata:
            for i, entry in enumerate(block[1:]):
                matrix[i] += [float(x) for x in entry.split()[1:]]

        matrix = np.array(matrix)
        return matrix

    def _format_basis(self, parsed):

        basis_key = None
        basis = {}
        basis_coefficients = {}

        # Keep track of current basis block
        basis_idx = -1

        # Keep track of maximum primitives combined
        max_basis = 0

        for line in parsed:
            if line.startswith('# Basis set for element'):
                basis_key = atomic_numbers[line.split()[6]]
                basis[basis_key] = []
                basis_idx = -1
                basis_coefficients[basis_key] = {}
            else:
                if basis_key is None or line.startswith('New'):
                    continue
                else:
                    line = line.split()
                    # Get angular momentum
                    if len(line) == 2:
                        basis[basis_key].append(self.basis_l[line[0]])
                        basis_idx += 1

                    # Extract basis set information
                    elif len(line) == 3:
                        coeff = float(line[1])
                        expon = float(line[2])

                        if basis_idx not in basis_coefficients[basis_key]:
                            basis_coefficients[basis_key][basis_idx] = [(coeff, expon)]
                        else:
                            basis_coefficients[basis_key][basis_idx].append(
                                (coeff, expon))

                        max_basis = max(max_basis,
                                        len(basis_coefficients[basis_key][basis_idx]))

        # Prepare everything

        lmn_data = {}
        basis_data = {}

        for elem in basis:

            current_ls = basis[elem]

            basis_block = []
            current_lmn = []

            # Get main quantum numbers, current_ls need to be added to get shell right:
            unique, count = np.unique(current_ls, return_counts=True)
            n = np.hstack([np.arange(count[u]) for u in unique]) + 1 + current_ls

            # Extract basis set information and l and m quantum numbers
            for idx, current_l in enumerate(current_ls):
                basis_info = np.zeros((2 * current_l + 1, max_basis, 2))

                data = np.array(basis_coefficients[elem][idx])

                # Assign the data to the zero-padded array
                n_primitives = data.shape[0]
                basis_info[:, :n_primitives, 0] = data[None, ..., 0]
                basis_info[:, :n_primitives, 1] = data[None, ..., 1]

                basis_block.append(basis_info)

                # Generate quantum numbers...
                m = m_range(current_l)
                current_lmn += list(zip((2 * current_l + 1) * [n[idx]],
                                        (2 * current_l + 1) * [current_l], m))

            basis_block = np.vstack(basis_block)
            current_lmn = np.array(current_lmn)

            lmn_data[elem] = current_lmn
            basis_data[elem] = basis_block

        return lmn_data, basis_data


def m_range(l):
    """
    Routine to get magnetic quantum numbers in Orca format
    """
    m = [0]
    for i in range(l):
        m += [(i + 1), -(i + 1)]
    return m


class OrcaDataParser(OrcaOutputParser):
    """
    Orca parser for hamiltonians and related data. Instance of the OrcaOutputParser class with predefined starts,
    stops and formatters for hamiltonians, overlaps, atoms and basis information.

    Args:
        properties (list(str)): list of properties to be collected. Possible are:
            'hamiltionian': Fock matrix data. Requires 'Print[P_Iter_F] 1' flag in Orca input.
            'overlap':      Overlap matrix. Requires 'Print[P_Overlap] 1' flag in Orca input.
            'basis':        Basis set information. Requires 'Print[P_AtomBasis] 1' and 'Print[P_Basis] 2' flags
                            in Orca input.
            'atoms':        Atom types and structure.
    """
    starts = {
        'hamiltonian': 'Fock matrix for operator 0',
        'overlap': 'OVERLAP MATRIX',
        'atoms': 'CARTESIAN COORDINATES (ANGSTROEM)',
        'basis': 'BASIS SET IN INPUT FORMAT',
        'energy': 'FINAL SINGLE POINT ENERGY',
        'forces': 'CARTESIAN GRADIENT'
    }

    stops = {
        'hamiltonian': ['***Gradient check signals convergence***',
                        '***Energy convergence achieved***',
                        '***RMSP convergence achieved***',
                        '***MAXP convergence achieved***',
                        '***Gradient convergence achieved***',
                        '***Orbital Rotation convergence achieved***',
                        '**** Energy Check signals convergence ****'],
        'overlap': ['DFT GRID GENERATION', 'INITIAL GUESS: MOREAD'],
        'atoms': 'CARTESIAN COORDINATES (A.U.)',
        'basis': ['ORCA GTO INTEGRAL CALCULATION', 'AUXILIARY BASIS SET INFORMATION',
                  'Checking for AutoStart:'],
        'forces': 'Difference to translation invariance:',
        'energy': None
    }

    formatters = {
        'hamiltonian': OrcaFormatter(None, datatype='matrix'),
        'overlap': OrcaFormatter(None, datatype='matrix'),
        'atoms': (
            OrcaFormatter(0, converter=str), OrcaFormatter(1, stop=4)
        ),
        'basis': OrcaFormatter(None, datatype='basis'),
        'forces': OrcaFormatter(3, stop=6, datatype='vector', unit=-1 / Bohr),
        'energy': OrcaFormatter(4)
    }

    def __init__(self,
                 properties=['hamiltonian', 'overlap', 'atoms', 'energy', 'forces']):
        parsers = {p: OrcaPropertyParser(self.starts[p], self.stops[p],
                                         formatters=self.formatters[p]) for p
                   in
                   properties}
        super(OrcaDataParser, self).__init__(parsers)


class OrcaPropertyParser:
    """
    Basic property parser for ORCA output files. Takes a start flag and a stop flag/list of stop flags and collects
    the data entries in between. If a formatter is provided, the data is formatted accordingly upon retrieval. Operates
    in a line-wise fashion.

    Args:
        start (str): begins to collect data starting from this string
        stop (str/list(str)): stops data collection if any of these strings is encounteres
        formatters (object): OrcaFormatter to convert collected data
    """

    def __init__(self, start, stop, formatters=None):
        self.start = start
        self.stop = stop
        self.formatters = formatters

        self.read = False
        self.parsed = None

    def parse_line(self, line):
        """
        Parses a line in the output file and updates Parser.

        Args:
            line (str): line of Orca output file
        """
        line = line.strip()
        if line.startswith("---------") or len(line) == 0:
            pass
        elif line.startswith(self.start):
            # Avoid double reading and restart for multiple files and repeated instances of data.
            self.parsed = []
            self.read = True
            # For single line output
            if self.stop is None:
                self.parsed.append(line)
                self.read = False
        elif self.read:
            # Check for stops
            if isinstance(self.stop, list):
                for stop in self.stop:
                    if self.read and line.startswith(stop):
                        self.read = False
                if self.read:
                    self.parsed.append(line)
            else:
                if line.startswith(self.stop):
                    self.read = False
                else:
                    self.parsed.append(line)

    def get_parsed(self):
        """
        Returns data, if formatters are specified in the corresponding format.
        """
        if self.formatters is None:
            return self.parsed
        elif hasattr(self.formatters, '__iter__'):
            return [formatter.format(self.parsed) for formatter in
                    self.formatters]
        else:
            return self.formatters.format(self.parsed)

    def reset(self):
        """
        Reset state of parser
        """
        self.read = False
        self.parsed = None


class SchNOrbAtomsData(spk.data.AtomsData):

    def __init__(self, *args, add_rotations=True, rotator_cls=OrcaRotator,
                 **kwargs):
        super(SchNOrbAtomsData, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.add_rotations = add_rotations
        self.basisdef = np.array(self.get_metadata('basisdef'))
        self.rotator_cls = rotator_cls
        self.rotator = rotator_cls(self.basisdef)

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]

        return SchNOrbAtomsData(*self.args,
                                add_rotations=self.add_rotations,
                                rotator_cls=self.rotator_cls,
                                subset=subidx,
                                **self.kwargs)

    def __getitem__(self, idx):
        at, properties = self.get_properties(idx)

        if self.add_rotations:
            H = properties[SchNOrbProperties.ham_prop].numpy()
            S = properties[SchNOrbProperties.ov_prop].numpy()

            isnan = True
            while isnan:
                rnd_rot = rand_rotation_matrix()

                if SchNOrbProperties.f_prop in properties.keys():
                    Hrot, Srot, pos_rot, force_rot = self.rotator.transform(
                        rnd_rot, H, S, at.numbers, at.positions,
                        properties[SchNOrbProperties.f_prop].numpy()
                    )
                    isnan = check_nan_np(Hrot, Srot, pos_rot, force_rot)
                else:
                    Hrot, Srot, pos_rot = self.rotator.transform(
                        rnd_rot, H, S, at.numbers, at.positions
                    )
                    isnan = check_nan_np(Hrot, Srot, pos_rot)

            at.set_positions(pos_rot)
            properties[SchNOrbProperties.R] = torch.FloatTensor(pos_rot)
            properties[SchNOrbProperties.ham_prop] = torch.FloatTensor(Hrot)
            properties[SchNOrbProperties.ov_prop] = torch.FloatTensor(Srot)
            if SchNOrbProperties.f_prop in properties.keys():
                properties[SchNOrbProperties.f_prop] = torch.FloatTensor(force_rot)

        # get atom environment
        nbh_idx, offsets = self.environment_provider.get_environment(at)

        properties[SchNOrbProperties.neighbors] = torch.LongTensor(
            nbh_idx.astype(np.int))
        properties[SchNOrbProperties.cell_offset] = torch.FloatTensor(
            offsets.astype(np.float32))
        properties['_idx'] = torch.LongTensor(np.array([idx], dtype=np.int))

        if self.collect_triples:
            nbh_idx_j, nbh_idx_k = spk.environment.collect_atom_triples(nbh_idx)
            properties[SchNOrbProperties.neighbor_pairs_j] = torch.LongTensor(
                nbh_idx_j.astype(np.int))
            properties[SchNOrbProperties.neighbor_pairs_k] = torch.LongTensor(
                nbh_idx_k.astype(np.int))

        return properties
