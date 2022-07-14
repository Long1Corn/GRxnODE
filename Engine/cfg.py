class CFG:
    def __init__(self):
        self.device = "cuda"
        self.lr_rate = 0.005
        self.lr_RTD = 0.01
        self.lr_warmup = 1
        self.weight_decay = 1e-4
        self.lr_step_size = [600, ]
        self.lr_gamma = 0.1

        self.itr = 800

        self.print_freq = 10
        self.test_freq = 50

        self.ode_rtol = 0.001
        self.ode_atol = 0.0001
        self.ode_method = "adaptive_heun"
        self.ode_tol_step = 200
        self.ode_tol_gamma = 0.1

        self.train_batch_size = 0
        self.test_batch_size = 0
        self.train_grad_clip = 5
        self.save_filename = 'model.pth'
        self.save_freq = 100

        self.model_in_var = 3
        self.model_hidden_var = 24
        self.model_out_var = 3


        self.RTD_N = 100
        self.RTD_max_tau = 4

