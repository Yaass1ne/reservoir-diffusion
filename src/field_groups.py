"""HDF5 group each dataset field lives in, for the 780_c5.h5 layout."""
FIELD_GROUP = {
    # inputs/ group (rock + injection controls)
    "Perm": "inputs", "Por": "inputs", "Rate": "inputs", "Bhp": "inputs", "Boundary": "inputs",
    # outputs/ group (fluids + geomechanics)
    "sat": "outputs", "pressure": "outputs", "Displacement": "outputs", "Vstrain": "outputs",
}
