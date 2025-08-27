"""
Define constants used commonly
"""

NA = 6.02214179e23  # Avogadro's number in mol^-1
kb = 1.380649e-23  # Boltzmann constant in J/K
R = 8.314472  # Ideal gas constant in J/(mol*K)
Rd = 287.052874 # J/(kg*K), specific gas constant for dry air
Rv = 461.525 # J/(kg*K), specific gas constant for water vapor
EPSILON = Rd/Rv # ratio of the gas constant for dry air to that for water vapor (Rd/Rv) = 0.622 approx.
g = 9.80665 # m/s^2, acceleration due to gravity
M_dry = 0.02897  # kg/mol, molar mass of dry air

# molar masses in g/mol of common atmospheric gases
molar_masses = {'co2': 44.0095,
                'ch4': 16.0425,
                'o3': 47.9982,
                'no2': 46.0055,
                'so2': 64.0638,
                'co': 28.0101,
                'h2o': 18.01528,
                'dry_air': 28.964}
