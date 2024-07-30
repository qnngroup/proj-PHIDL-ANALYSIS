import phidlfem.analysis as pfa
import phidl.geometry as pg

if __name__ == "__main__":
    D = pg.optimal_step(start_width=10, end_width=20, anticrowding_factor=1)
    # D = pg.flagpole(size=(10,30), stub_size=(4,40), shape='p', taper_type='straight')
    sq = pfa.get_squares(D, 0.5)
    print(f"squares = {sq}")
    pfa.visualize_laplace(D, 0.5)
