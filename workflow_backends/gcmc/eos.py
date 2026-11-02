"""Resolve reservoir phi before FLAMES constructs its cached fugacity."""
import math

from workflow_backends.gcmc.upstream import load_official


def reservoir_phi(parameters, *, temperature_k, pressure_bar, guest):
    if parameters.eos_kind != 'pr':
        return parameters.phi
    load_official()
    from flames.eos import PengRobinsonEOS
    eos = PengRobinsonEOS(temperature=float(temperature_k), pressure=float(pressure_bar)*1e5,
                         criticalTemperature=parameters.tc_k, criticalPressure=parameters.pc_bar*1e5,
                         acentricFactor=parameters.omega, molarMass=float(guest.get_masses().sum()))
    phi = float(eos.get_fugacity_coefficient())
    if not math.isfinite(phi) or phi <= 0:
        raise ValueError('official PR returned nonfinite/nonpositive fugacity coefficient')
    return phi
