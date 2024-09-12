class WarmupScheduler:
    def __init__(self, optimizer, target_lr,initial_lr=1e-7,warmup_steps=25):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.current_step = 0
    
    def step(self):
        if self.current_step < self.warmup_steps:
            # Linearly increase the learning rate
            lr = (self.target_lr - self.initial_lr) * (self.current_step / self.warmup_steps) + self.initial_lr
            # Apply the learning rate to the optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        # Increment the step count
        self.current_step += 1
