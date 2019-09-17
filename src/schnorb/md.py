from schnetpack.md.calculators import MDCalculator
from schnetpack.md.utils import MDUnits
from schnetpack import Properties


class SchNOrbMD(MDCalculator):

    def __init__(self, model,
                 required_properties=['energy', 'forces', 'hamiltonian', 'overlap'],
                 force_handle='forces',
                 position_conversion=1.0 / MDUnits.angs2bohr,
                 force_conversion=1.0 / MDUnits.angs2bohr):
        super(SchNOrbMD, self).__init__(required_properties, force_handle,
                                        position_conversion=position_conversion,
                                        force_conversion=force_conversion)
        self.model = model

    def calculate(self, system):
        inputs = self._generate_input(system)
        H, S, E, F = self.model(inputs)
        self.results = {
            'energy': E,
            'forces': F,
            'hamiltonian': H,
            'overlap': S
        }
        self._update_system(system)

    def _generate_input(self, system):
        positions, atom_types, atom_masks = self._get_system_molecules(system)
        neighbors, neighbor_mask = self._get_system_neighbors(system)

        inputs = {
            Properties.R: positions,
            Properties.Z: atom_types,
            Properties.atom_mask: atom_masks,
            Properties.cell: None,
            Properties.cell_offset: None,
            Properties.neighbors: neighbors,
            Properties.neighbor_mask: neighbor_mask
        }

        return inputs
