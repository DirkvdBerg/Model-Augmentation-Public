from model_augmentation.fit_systems.interconnect import *

def test_Interconnect():
    nu = 1
    ny = 2
    interconnect = Interconnect(nu, ny)

    assert interconnect.nu == nu
    assert interconnect.ny == ny

    assert interconnect.nu != ny
    assert interconnect.ny != nu

