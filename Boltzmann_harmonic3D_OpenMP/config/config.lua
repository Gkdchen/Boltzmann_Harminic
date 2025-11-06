config = {
    grid = {
        nx = 100,
        ny = 100,
        nz = 100,
        np = 100
    },
    solver ={
        solver_name = "3d",
        save_iter = 1000000,
        display_iter = 100,
        max_iter = 10000000
    },
    simulator = {
        max_iter = 30000000,
        save_iter = 1000000,
        display_iter = 100
    },
    flow = {
        source = 1.0,
        lamda = 1.0/3.0,
        tau_f = 1.0
    },
    experiments = {
        name = "harmoinc",
    }
}
